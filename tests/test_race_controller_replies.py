"""Regressions from the actual 2026-11-18 Rorqual/Killarney race replies.

Fixture fields are verbatim excerpts of
/tmp/lsj-ready-pair-20261118T004507Z/rorqual-loser-component-confirmation.txt.
The bare sacct row is projected to the JobID,State fields cluv requests.
"""

import datetime
import importlib
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluv.cache import Job
from cluv.slurm import get_job_states_with_sacct, parse_scontrol_het_state

dispatch = importlib.import_module("cluv.cli.submit")
CONTROL = (Path(__file__).parent / "fixtures/rorqual-never-allocated-scontrol.txt").read_text()
NAME = "cluv-lsj-ready-pair-20261118T004507Z"


@pytest.mark.parametrize(
    "change,expected",
    [
        ("unchanged", "CANCELLED"),
        ("missing", "UNKNOWN"),
        ("wrong_leader", "UNKNOWN"),
        ("wrong_set", "UNKNOWN"),
        ("wrong_name", "UNKNOWN"),
        ("running_sibling", "INCOMPLETE"),
        ("conflicting_duplicate", "UNKNOWN"),
        ("nonconsecutive", "CANCELLED"),
        ("missing_set", "UNKNOWN"),
    ],
)
def test_authoritative_controller_requires_exact_whole_het_identity(change, expected):
    text = CONTROL
    if change == "missing":
        text = text.split("\n\n")[0]
    elif change == "wrong_leader":
        text = text.replace("HetJobId=27087172", "HetJobId=999")
    elif change == "wrong_set":
        text = text.replace("27087172-27087173", "27087172-27087174")
    elif change == "wrong_name":
        text = text.replace(NAME, "another-race")
    elif change == "running_sibling":
        first, second = text.split("\n\n")
        text = first + "\n\n" + second.replace("JobState=CANCELLED", "JobState=RUNNING")
    elif change == "conflicting_duplicate":
        text += "\n" + text.split("\n\n")[1].replace("JobState=CANCELLED", "JobState=RUNNING")
    elif change == "nonconsecutive":
        text = text.replace("27087172-27087173", "27087172,27087179").replace(
            "JobId=27087173 ", "JobId=27087179 "
        )
    elif change == "missing_set":
        text = text.replace("HetJobIdSet=27087172-27087173", "")
    assert parse_scontrol_het_state(text, 27087172, 2, NAME) == expected


async def test_captured_missing_accounting_sibling_confirms_all_cancelled(
    tmp_path,
):
    calls = []

    class Remote:
        hostname = "rorqual"

        async def get_output(self, command, **kwargs):
            calls.append(command)
            assert command.startswith("bash --login -c ")
            inner = shlex.split(command)[-1]
            if inner.startswith("scancel "):
                assert (
                    inner == "scancel 27087172"
                )  # Never touch the successful Killarney allocation.
                return ""
            if inner.startswith("sacct "):
                return "27087172|CANCELLED by 3098083\n"
            assert inner == "scontrol show job -o 27087172"
            return CONTROL

    remote = Remote()
    loser = Job(
        cluster="rorqual",
        remote=remote,
        job_script=tmp_path / "job.sh",
        sbatch_args={"job-name": NAME, "__het_components__": [{}]},
        program_args=[],
        sbatch_command="captured",
        n_chunks=None,
        git_commit="pin",
        job_id=27087172,
        submitted_at=datetime.datetime.now(),
    )
    row = dispatch.SubmissionProgress(loser, tmp_path / "receipt", state="PENDING")
    await dispatch.wait_for_jobs_to_cancel(
        [row], {"rorqual": remote}, max_wait_time_seconds=1, initial_delay=0.001
    )
    assert row.state == "CANCELLED"
    assert len(calls) == 3


async def test_login_control_commands_are_used_on_local_cluster_too(monkeypatch, tmp_path):
    import cluv.slurm as slurm

    commands = []

    async def run(argv, **kwargs):
        commands.append(argv)
        assert argv[:3] == ("bash", "--login", "-c")
        inner = argv[3]
        if inner.startswith("scancel "):
            return SimpleNamespace(stdout="")
        return SimpleNamespace(
            stdout="27087172|CANCELLED\n" if inner.startswith("sacct ") else CONTROL
        )

    monkeypatch.setattr(dispatch, "run", run)
    monkeypatch.setattr(slurm, "run", run)
    row = SimpleNamespace(job_id=27087172, job=SimpleNamespace(remote=None))
    await dispatch.run_scancel([row])
    result = await get_job_states_with_sacct(None, [27087172], {27087172: 2}, {27087172: NAME})
    assert result == ["CANCELLED"] and len(commands) == 3


async def test_controller_missing_sibling_remains_unknown():
    class Remote:
        async def get_output(self, command, **kwargs):
            return CONTROL.split("\n\n")[0] if "scontrol" in command else "27087172|CANCELLED\n"

    assert await get_job_states_with_sacct(Remote(), [27087172], {27087172: 2}) == ["UNKNOWN"]


async def test_controller_fallback_timeout_is_surfaced_not_terminal(monkeypatch):
    import asyncio
    import cluv.slurm as slurm

    timeout = asyncio.timeout
    monkeypatch.setattr(slurm.asyncio, "timeout", lambda _: timeout(0.01))

    class Remote:
        async def get_output(self, command, **kwargs):
            if "scontrol" in command:
                await asyncio.sleep(30)
            return "27087172|CANCELLED\n"

    with pytest.raises(TimeoutError):
        await get_job_states_with_sacct(Remote(), [27087172], {27087172: 2})


async def test_real_reply_race_returns_winner_after_confirmed_unallocated_loser_cleanup(
    tmp_path, monkeypatch
):
    import asyncio
    import subprocess
    from unittest.mock import AsyncMock
    from cluv.cache import Submission
    from cluv.config import SubmitPolicy

    cancelled = []

    class Remote:
        def __init__(self, name):
            self.hostname = name

        async def run(self, command, **kwargs):
            assert "sbatch --parsable" in command
            return subprocess.CompletedProcess(
                command, 0, "27087172" if self.hostname == "rorqual" else "7739062", ""
            )

        async def get_output(self, command, **kwargs):
            assert command.startswith("bash --login -c ")
            inner = shlex.split(command)[-1]
            if inner.startswith("scancel "):
                assert self.hostname == "rorqual" and inner == "scancel 27087172"
                cancelled.append(27087172)
                return ""
            if self.hostname == "killarney":
                assert inner.startswith("sacct ")
                return "7739062+0|RUNNING\n7739062+1|RUNNING\n"
            if inner.startswith("sacct "):
                return "27087172|CANCELLED by 3098083\n" if cancelled else "27087172|PENDING\n"
            return (
                CONTROL if cancelled else CONTROL.replace("JobState=CANCELLED", "JobState=PENDING")
            )

    remotes = {name: Remote(name) for name in ("rorqual", "killarney")}

    async def specs(cluster, remote, **kwargs):
        return [
            Submission(
                cluster=cluster,
                remote=remote,
                job_script=tmp_path / "job.sh",
                sbatch_args={"job-name": NAME, "__het_components__": [{}]},
                program_args=[],
                sbatch_command="sbatch --parsable fixture",
                n_chunks=None,
                git_commit="pin",
            )
        ]

    monkeypatch.setattr(
        dispatch, "get_cluv_config", lambda: SimpleNamespace(submit=SubmitPolicy())
    )
    monkeypatch.setattr(dispatch, "get_cluster_to_remote", AsyncMock(return_value=remotes))
    monkeypatch.setattr(dispatch, "get_submissions", specs)
    monkeypatch.setattr(dispatch, "ensure_clean_git_state", lambda **_: "pin")
    monkeypatch.setattr(dispatch, "get_submission_log_dir", lambda: tmp_path)
    saved = []
    monkeypatch.setattr(dispatch, "save_job", saved.append)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _: real_sleep(0))
    winner = await dispatch.submit("first", None, [], [], _skip_sync=True)
    assert winner.job_id == 7739062 and saved == [winner]
    assert cancelled == [27087172]
