import datetime
import shlex
from pathlib import Path

import pytest

import cluv.cli.submit as submit_module
from cluv.cache import Submission
from cluv.cli.submit_utils import dry_run
from cluv.cli.submit_utils.dry_run import (
    TestOnlyResult,
    clean_sbatch_error,
    get_test_only_command,
    parse_test_only_output,
    render_test_only_table,
)
from cluv.config import get_cluv_config

FORECAST = (
    "sbatch: Job 123 to start at 2026-11-03T17:00:00 using 4 processors on nodes cn-a001 "
    "in partition main\n"
)


def test_parse_forecast() -> None:
    assert parse_test_only_output(FORECAST) == {
        "start": datetime.datetime(2026, 11, 3, 17, 0, 0),
        "cpus": 4,
        "nodes": "cn-a001",
        "partition": "main",
    }


def test_parse_forecast_among_other_lines_and_minimal_form() -> None:
    out = (
        "sbatch: warning: something\n"
        "sbatch: Job 7 to start at 2026-11-03T17:00:00 using 1 processors on nodes cn[1-2] "
        "in partition gpu,long\n"
    )
    parsed = parse_test_only_output(out)
    assert parsed is not None
    assert parsed["nodes"] == "cn[1-2]" and parsed["partition"] == "gpu,long"
    minimal = parse_test_only_output("Job 7 to start at 2026-11-03T17:00:00")
    assert minimal == {
        "start": datetime.datetime(2026, 11, 3, 17),
        "cpus": None,
        "nodes": None,
        "partition": None,
    }


def test_parse_returns_none_without_forecast() -> None:
    assert parse_test_only_output("sbatch: error: Invalid account\n") is None
    assert parse_test_only_output("") is None


def test_clean_sbatch_error() -> None:
    out = "sbatch: error: Batch job submission failed: Invalid account\n"
    assert clean_sbatch_error(out) == "Batch job submission failed: Invalid account"


def test_command_has_test_only_no_parsable_and_no_script() -> None:
    command = get_test_only_command({"account": "def-x", "time": "0:10:00", "gres": "gpu:1"})
    (inner,) = shlex.split(command)[3:]
    assert inner == 'sbatch --test-only "--account=def-x" "--time=0:10:00" "--gres=gpu:1"'
    assert "--parsable" not in command


def test_table_shows_accepted_and_rejected() -> None:
    now = datetime.datetime(2026, 11, 3, 15, 0)
    results = [
        TestOnlyResult(
            cluster="mila",
            config="account=a",
            start=datetime.datetime(2026, 11, 3, 17, 0),
            cpus=4,
            nodes="cn-a001",
            partition="main",
        ),
        TestOnlyResult(cluster="fir", config="account=b", error="Invalid account"),
    ]
    table = render_test_only_table(results, now=now)
    from rich.console import Console

    console = Console(width=140, record=True)
    console.print(table)
    text = console.export_text()
    assert "in 2h00m" in text and "cn-a001" in text and "main" in text
    assert "REJECTED: Invalid account" in text


class _FakeRemote:
    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode, self.stderr = returncode, stderr
        self.calls: list[tuple[str, str | None]] = []

    async def run(self, command, *, input=None, **kwargs):
        import subprocess

        self.calls.append((command, input))
        return subprocess.CompletedProcess(command, self.returncode, "", self.stderr)


def _submission(tmp_path: Path, remote) -> Submission:
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --gres=gpu:1\necho hi\n")
    return Submission(
        cluster="mila",
        remote=remote,
        job_script=script,
        sbatch_args={"account": "a", "time": "0:10:00"},
        program_args=["x"],
        sbatch_command="unused",
        n_chunks=None,
        git_commit="c",
    )


async def test_submission_sends_script_on_stdin_and_parses(tmp_path: Path) -> None:
    remote = _FakeRemote(0, FORECAST)
    result = await dry_run.test_only_submission(_submission(tmp_path, remote))
    assert (
        result.accepted
        and result.partition == "main"
        and result.config == "account=a time=0:10:00"
    )
    ((command, stdin),) = remote.calls
    assert "--test-only" in command and "--parsable" not in command
    assert stdin is not None and "#SBATCH --gres=gpu:1" in stdin


async def test_submission_reports_rejection(tmp_path: Path) -> None:
    remote = _FakeRemote(1, "sbatch: error: Batch job submission failed: Invalid account\n")
    result = await dry_run.test_only_submission(_submission(tmp_path, remote))
    assert not result.accepted
    assert result.error == "Batch job submission failed: Invalid account"


async def test_exit_status_is_1_when_a_config_is_rejected(tmp_path: Path) -> None:
    ok = _submission(tmp_path, _FakeRemote(0, FORECAST))
    bad = _submission(tmp_path, _FakeRemote(1, "sbatch: error: nope\n"))
    with pytest.raises(SystemExit) as exc:
        await dry_run.test_only([ok, bad])
    assert exc.value.code == 1
    assert len(await dry_run.test_only([ok])) == 1


async def test_submit_test_only_does_not_sync_commit_or_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`submit(test_only=True)` expands the configurations but never syncs, commits or saves."""
    sub = _submission(tmp_path, _FakeRemote(0, FORECAST))
    seen = {}

    async def fake_get_cluster_to_remote(cluster):
        return {"mila": sub.remote}

    async def fake_get_submissions(**kwargs):
        seen.update(kwargs)
        return [sub, sub]

    async def fail(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(submit_module, "get_cluster_to_remote", fake_get_cluster_to_remote)
    monkeypatch.setattr(submit_module, "get_submissions", fake_get_submissions)
    monkeypatch.setattr(submit_module, "sync_common_part", fail)
    monkeypatch.setattr(submit_module, "ensure_clean_git_state", fail)
    monkeypatch.setattr(submit_module, "save_job", fail)
    get_cluv_config.cache_clear()

    assert await submit_module.submit("mila", None, [], ["x"], test_only=True) is None
    assert seen["cluster"] == "mila"
    assert len(sub.remote.calls) == 2


def test_parse_real_cluster_variants() -> None:
    """Outputs seen on Mila: a stray token after the time, and no nodes/partition for het jobs."""
    odd = parse_test_only_output(
        "sbatch: Job 1 to start at 2026-11-03T17:06:26 a using 1 processors on nodes cn-f002 "
        "in partition long-cpu\n"
    )
    assert odd is not None and odd["cpus"] == 1 and odd["nodes"] == "cn-f002"
    assert odd["partition"] == "long-cpu"
    het = parse_test_only_output(
        "sbatch: 0: # Using default long partition\n"
        "sbatch: Job 2 to start at 2026-11-03T17:24:27 using 2 processors on cn-c036\n"
    )
    assert het is not None and het["cpus"] == 2 and het["nodes"] == "cn-c036"
    assert het["partition"] is None


def test_clean_sbatch_error_drops_ansi_banner() -> None:
    out = (
        "sbatch: error: \x1b[00;36m-----------\n"
        "sbatch: error: # Using default long partition\n"
        "sbatch: error: -----------\x1b[00m\n"
        "allocation failure: Invalid account or account/partition combination specified\n"
    )
    assert clean_sbatch_error(out) == (
        "allocation failure: Invalid account or account/partition combination specified"
    )
