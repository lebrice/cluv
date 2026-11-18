"""Read-only status faults must not discard healthy queued allocations."""

import asyncio
from datetime import datetime, timezone
import importlib
import shlex
import subprocess

import pytest

from cluv.cache import Job
from cluv import slurm

dispatch = importlib.import_module("cluv.cli.submit")


class StatusRemote:
    def __init__(self, name, job_id, answers):
        self.hostname, self.job_id, self.answers = name, job_id, answers
        self.polls = 0
        self.current = None

    async def get_output(self, command, **kwargs):
        inner = shlex.split(command)[-1]
        if inner.startswith("sacct "):
            self.current = self.answers[min(self.polls, len(self.answers) - 1)]
            self.polls += 1
            if isinstance(self.current, BaseException):
                raise self.current
            if self.current == "fallback_timeout":
                return f"{self.job_id}|PENDING\n"
            return "\n".join(f"{self.job_id}+{i}|{self.current}" for i in (0, 1))
        assert inner.startswith("scontrol show job -o ")
        assert self.current == "fallback_timeout"
        await asyncio.sleep(1)  # The real scontrol asyncio.timeout cancels this read.
        raise AssertionError("timeout must interrupt the actual awaited read")


async def run_poll(monkeypatch, tmp_path, remotes, initial_states=None):
    async def prepared(*args, **kwargs):
        return []

    monkeypatch.setattr(dispatch, "sync_and_submit_jobs_to_cluster", prepared)
    rows = {}
    for name, remote in remotes.items():
        job = Job(
            cluster=name,
            remote=remote,
            job_script=tmp_path / "job.sh",
            sbatch_args={"job-name": "owned-race", "__het_components__": [{}]},
            program_args=[],
            sbatch_command="fixture",
            n_chunks=None,
            git_commit="pin",
            job_id=remote.job_id,
            submitted_at=datetime.now(timezone.utc),
        )
        rows[name] = [
            dispatch.SubmissionProgress(
                job, tmp_path / f"{name}.log", state=(initial_states or {}).get(name, "PENDING")
            )
        ]
    winner = await dispatch.wait_for_first_running_job(
        rows,
        cluster_to_remote=remotes,
        found_running_job=asyncio.Event(),
        _skip_sync=True,
        sync_datasets=False,
        fetch_results=False,
        initial_delay=0,
        max_wait_time_seconds=0,
    )
    return winner, rows


async def test_actual_scontrol_timeout_then_fresh_recovery(monkeypatch, tmp_path):
    original_timeout = asyncio.timeout
    monkeypatch.setattr(slurm.asyncio, "timeout", lambda _: original_timeout(0.001))
    remote = StatusRemote("site", 101, ["fallback_timeout", "RUNNING"])
    winner, _ = await run_poll(monkeypatch, tmp_path, {"site": remote})
    assert winner.job_id == 101 and remote.polls == 2


async def test_persistent_one_site_fault_keeps_healthy_pending_age_and_winner(
    monkeypatch, tmp_path
):
    messages = []
    monkeypatch.setattr(dispatch.console, "log", lambda message: messages.append(str(message)))
    bad = StatusRemote("bad", 101, [ConnectionResetError("read connection reset")])
    healthy = StatusRemote("healthy", 201, ["PENDING"] * 4 + ["RUNNING"])
    winner, rows = await run_poll(
        monkeypatch, tmp_path, {"bad": bad, "healthy": healthy}, {"bad": "RUNNING"}
    )
    assert winner.job_id == 201 and healthy.polls == bad.polls == 5
    assert rows["bad"][0].state == "UNKNOWN"  # Never select its stale RUNNING.
    output = "\n".join(messages)
    assert "DEGRADED" in output and "bad" in output and "101" in output
    assert "failures 3:" in output


async def test_all_relevant_read_failures_are_bounded(monkeypatch, tmp_path):
    remotes = {
        name: StatusRemote(name, i, [TimeoutError("read unavailable")])
        for name, i in [("first", 101), ("second", 201)]
    }
    with pytest.raises(RuntimeError, match="all remaining candidate clusters") as error:
        await run_poll(monkeypatch, tmp_path, remotes)
    assert all(remote.polls == 3 for remote in remotes.values())
    assert "first" in str(error.value) and "second" in str(error.value)


async def test_preempted_attempt_is_read_again_and_can_requeue(monkeypatch, tmp_path):
    requeued = StatusRemote("requeued", 101, ["PREEMPTED", "RUNNING"])
    pending = StatusRemote("pending", 201, ["PENDING"])
    winner, _ = await run_poll(monkeypatch, tmp_path, {"requeued": requeued, "pending": pending})
    assert winner.job_id == 101 and requeued.polls == 2


@pytest.mark.parametrize("failure", [ValueError("unsafe parsed status"), asyncio.CancelledError()])
async def test_safety_errors_and_user_cancellation_are_not_retried(monkeypatch, tmp_path, failure):
    remote = StatusRemote("site", 101, [failure])
    with pytest.raises(type(failure)):
        await run_poll(monkeypatch, tmp_path, {"site": remote})
    assert remote.polls == 1


@pytest.mark.parametrize(
    "message,retry",
    [
        ("Connection reset by peer", True),
        ("Host key verification failed", False),
        ("Permission denied (publickey).\nConnection closed by UNKNOWN port 65535", False),
        ("REMOTE HOST IDENTIFICATION HAS CHANGED!\nConnection reset by peer", False),
    ],
)
async def test_only_identified_ssh_transport_losses_retry(monkeypatch, tmp_path, message, retry):
    failure = subprocess.CalledProcessError(255, ["ssh", "site", "sacct"], stderr=message)
    remote = StatusRemote("site", 101, [failure, "RUNNING"])
    if retry:
        winner, _ = await run_poll(monkeypatch, tmp_path, {"site": remote})
        assert winner.job_id == 101 and remote.polls == 2
    else:
        with pytest.raises(subprocess.CalledProcessError):
            await run_poll(monkeypatch, tmp_path, {"site": remote})
        assert remote.polls == 1
