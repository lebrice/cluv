"""Fake scheduler workflows: no SSH, sync, downloads or allocations."""

import asyncio
import importlib
import shlex
import subprocess
from unittest.mock import AsyncMock

import pytest

import cluv.__main__ as cli
import cluv.slurm as slurm
from cluv.cli.submit import merge_job_resources, submit, wait_for_jobs_to_cancel
from cluv.sbatch_args import sbatch_args_from_list, sbatch_args_to_list
from tests.test_submit import fake_home, no_active_remotes, project_dir  # noqa: F401

dispatch = importlib.import_module("cluv.cli.submit")


@pytest.mark.parametrize(
    "arguments", [["submit", "first", "--help"], ["submit", "first", "job.sh", "-h"]]
)
def test_help_after_positionals_has_no_dispatch(monkeypatch, capsys, arguments):
    call = AsyncMock(side_effect=AssertionError("No submit preparation allowed"))
    monkeypatch.setattr(cli, "submit", call)
    cli.main(arguments)
    assert "eligible-clusters" in capsys.readouterr().out
    call.assert_not_called()


def test_program_help_is_forwarded_and_selection_flags_rescued(monkeypatch):
    call = AsyncMock()
    monkeypatch.setattr(cli, "submit", call)
    cli.main(
        [
            "submit",
            "first",
            "--eligible-clusters",
            "fir,rorqual",
            "--eligible-accounts=def-user_gpu",
            "--",
            "--help",
        ]
    )
    assert call.call_args.kwargs["eligible_clusters"] == "fir,rorqual"
    assert call.call_args.kwargs["eligible_accounts"] == "def-user_gpu"
    assert call.call_args.kwargs["program_args"] == ["--help"]


@pytest.mark.parametrize(
    "states,expected",
    [
        ("99+0|RUNNING\n99+1|PENDING", "INCOMPLETE"),
        ("99+0|RUNNING", "UNKNOWN"),
        ("99+0|COMPLETED\n99+1|RUNNING", "INCOMPLETE"),
        ("99+0|CANCELLED\n99+1|COMPLETING", "INCOMPLETE"),
        ("99+0|COMPLETED\n99+1|CANCELLED", "CANCELLED"),
        ("99+0|RUNNING\n99+1|RUNNING", "RUNNING"),
    ],
)
def test_heterogeneous_allocation_state_requires_every_component(states, expected):
    assert slurm.parse_sacct_job_states(states, {99: 2}) == {99: expected}


async def test_account_matrix_het_resources_and_loser_cancellation(
    monkeypatch,
    project_dir,  # noqa: F811
    no_active_remotes,  # noqa: F811
):
    (project_dir / "pyproject.toml").write_text("""[tool.cluv]
results_path = "results"
[tool.cluv.submit]
exclude_accounts = ["cc-debug*"]
[tool.cluv.sbatch_args]
time = "00:20:00"
[tool.cluv.clusters.fir]
sbatch_args = [{account="cc-debug_gpu"},{account="def-user_gpu"},{account="rrg-user_gpu"},{account="unapproved_gpu"}]
""")
    script = project_dir / "job.sh"
    script.write_text(
        "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --gpus-per-node=h100:2\n#SBATCH hetjob\n#SBATCH --nodes=1\n#SBATCH --gpus-per-node=h100:4\necho training\n"
    )
    resolve = AsyncMock(return_value={"fir": None})
    monkeypatch.setattr(dispatch, "get_cluster_to_remote", resolve)
    monkeypatch.setattr(dispatch, "ensure_clean_git_state", lambda **_: "frozen-pin")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _: real_sleep(0))
    submitted = []
    cancelled = []
    polls = 0

    async def fake_run(argv, **kwargs):
        nonlocal polls
        command = shlex.join(argv)
        if "sbatch --parsable" in command:
            assert "cc-debug" not in command and "unapproved" not in command
            assert command.count("--time=00:20:00") == 2
            account = "def-user_gpu" if "def-user_gpu" in command else "rrg-user_gpu"
            assert command.count(f"--account={account}") == 2
            assert "h100:2" in command and "h100:4" in command and "#SBATCH hetjob" in command
            assert "frozen-pin" in command and "/train.parquet" in command
            submitted.append(account)
            output = "100" if account == "def-user_gpu" else "200"
        elif "sacct -j" in command:
            polls += 1
            ids = command.split("sacct -j ")[1].split()[0].split(",")
            lines = []
            for job_id in ids:
                if job_id == "100":
                    states = ["RUNNING", "PENDING"] if polls < 2 else ["RUNNING", "RUNNING"]
                else:
                    # First cancellation poll must not accept only the leader's terminal state.
                    states = (
                        ["CANCELLED", "RUNNING"]
                        if len(cancelled) == 1
                        else (["CANCELLED"] * 2 if cancelled else ["PENDING"] * 2)
                    )
                lines += [f"{job_id}+{i}|{s}" for i, s in enumerate(states)]
            output = "\n".join(lines)
        elif command == "scancel 200":
            cancelled.append(200)
            output = ""
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(dispatch, "run", fake_run)
    monkeypatch.setattr(slurm, "run", fake_run)
    result = await submit(
        "first",
        script,
        [],
        ["--train-data", "/train.parquet"],
        eligible_clusters="fir",
        eligible_accounts="def-user_gpu,rrg-user_gpu",
        _skip_sync=True,
    )
    resolve.assert_awaited_once_with(["fir"])
    assert result.job_id == 100
    assert set(submitted) == {"def-user_gpu", "rrg-user_gpu"}
    assert cancelled == [200, 200]


@pytest.mark.parametrize("failure", ["cancel", "missing_component", "hung_query"])
async def test_cancellation_deadline_is_bounded_and_failure_is_visible(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    class Row:
        state = "RUNNING"
        job_id = 99
        cluster = "fir"
        job = SimpleNamespace(job_id=99, remote=None, sbatch_args={"__het_components__": [{}]})

    async def cancel(_):
        if failure == "cancel":
            raise RuntimeError("scheduler denied cancellation")

    async def query(*args, **kwargs):
        if failure == "hung_query":
            await asyncio.sleep(30)
        return ["UNKNOWN" if failure == "missing_component" else "RUNNING"]

    monkeypatch.setattr(dispatch, "run_scancel", cancel)
    monkeypatch.setattr(dispatch, "get_job_states_with_sacct", query)
    with pytest.raises(TimeoutError, match="Loser cancellation unconfirmed") as error:
        await asyncio.wait_for(
            wait_for_jobs_to_cancel(
                [Row()], {"fir": None}, max_wait_time_seconds=0.03, initial_delay=0.001
            ),
            timeout=1,
        )
    if failure == "cancel":
        assert "scheduler denied cancellation" in str(error.value)


def test_per_component_overrides_and_common_account_time(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\n#SBATCH --nodes=12\n#SBATCH --gpus-per-node=rtx:8\n#SBATCH hetjob\n#SBATCH --nodes=21\n#SBATCH --gpus-per-node=b200:8\necho go\n"
    )
    flags = [
        "--nodes=1",
        "--time=00:10:00",
        ":",
        "--nodes=2",
        "--time=00:15:00",
        "--account=other_gpu",
    ]
    merged = merge_job_resources(script, {"account": "def-user_gpu", "time": "00:20:00"}, flags)
    rendered = sbatch_args_to_list(merged)
    assert sbatch_args_from_list(rendered) == merged
    first, second = " ".join(rendered).split(" : ")
    assert "--account=def-user_gpu" in first and "--time=00:10:00" in first and "rtx:8" in first
    assert "--account=other_gpu" in second and "--time=00:15:00" in second and "b200:8" in second


def test_real_shell_passes_het_stdin_and_program_args_through_original_sbatch(tmp_path):
    from pathlib import PurePosixPath

    from cluv.cli.submit import get_sbatch_command

    binary = tmp_path / "sbatch"
    binary.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$TRACE_ARGS"\ncat > "$TRACE_SCRIPT"\nprintf "%s\\n" "$PATH" > "$TRACE_PATH"\necho 123\n'
    )
    binary.chmod(0o755)
    script = '#!/bin/bash\n#SBATCH --nodes=12\n#SBATCH hetjob\n#SBATCH --nodes=21\necho "$UNEXPANDED_MODEL"\n'
    args = {
        "job-name": "cluv-race-unique-123",
        "nodes": "1",
        "gpus-per-node": "h100:2",
        "account": "ordinary",
        "time": "00:20:00",
        "__het_components__": [
            {"nodes": "1", "gpus-per-node": "h100:4", "account": "ordinary", "time": "00:20:00"}
        ],
    }
    command = get_sbatch_command(
        PurePosixPath("canonical.sh"),
        args,
        ["--train-data", "/data with spaces/train.parquet", "--", "--max-steps", "8"],
        dict(
            PATH=f"{tmp_path}:$PATH",
            GIT_COMMIT="frozen",
            TRACE_ARGS=str(tmp_path / "args"),
            TRACE_SCRIPT=str(tmp_path / "script"),
            TRACE_PATH=str(tmp_path / "path"),
        ),
        script_contents=script,
    )
    result = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "123"
    actual_args = (tmp_path / "args").read_text().splitlines()
    assert actual_args == [
        "--parsable",
        "--job-name=cluv-race-unique-123",
        "/dev/stdin",
        "--train-data",
        "/data with spaces/train.parquet",
        "--",
        "--max-steps",
        "8",
    ]
    body = (tmp_path / "script").read_text()
    assert body.count("#SBATCH hetjob") == 1
    assert body.count("--job-name=cluv-race-unique-123") == 2
    assert body.count("--account=ordinary") == body.count("--time=00:20:00") == 2
    assert "h100:2" in body and "h100:4" in body and "--nodes=12" not in body
    assert 'echo "$UNEXPANDED_MODEL"' in body
    assert (tmp_path / "path").read_text().startswith(f"{tmp_path}:")


@pytest.mark.parametrize("selector", ["", "not-configured"])
async def test_invalid_cluster_selection_stops_before_connection_or_git(monkeypatch, selector):
    connect = AsyncMock(side_effect=AssertionError("No connection allowed"))
    monkeypatch.setattr(dispatch, "get_cluster_to_remote", connect)
    monkeypatch.setattr(dispatch, "ensure_clean_git_state", lambda **_: pytest.fail("No git prep"))
    with pytest.raises(ValueError, match="eligible|Eligible"):
        await submit("first", None, [], [], eligible_clusters=selector)
    connect.assert_not_called()


async def test_preparation_is_once_per_cluster_and_shared_checkout_is_serialized(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from cluv.cli.submit import sync_and_submit_jobs_to_cluster

    active = 0
    preparing = set()
    calls = []
    submitting = []

    async def prepare(remote, **kwargs):
        nonlocal active
        active += 1
        assert active == 1
        calls.append(remote)
        preparing.add(remote)
        await asyncio.sleep(0.005)
        preparing.remove(remote)
        active -= 1

    async def submit_variant(submission, path):
        # The preparation lock must be released before account variant submission.
        assert submission.rsplit("-", 1)[0] not in preparing
        submitting.append(submission)
        raise dispatch.JobSubmissionFailed("fake rejection")

    monkeypatch.setattr(dispatch, "sync_per_cluster_part", prepare)
    monkeypatch.setattr(dispatch, "submit_job", submit_variant)
    lock = asyncio.Lock()
    event = asyncio.Event()
    groups = [
        [
            SimpleNamespace(state="SYNCING", log_path=tmp_path / f"{c}-{a}.log", job=f"{c}-{a}")
            for a in range(2)
        ]
        for c in ("trillium", "trillium-gpu")
    ]
    await asyncio.gather(
        *[
            sync_and_submit_jobs_to_cluster(c, c, rows, event, preparation_lock=lock)
            for c, rows in zip(("trillium", "trillium-gpu"), groups, strict=True)
        ]
    )
    assert calls == ["trillium", "trillium-gpu"]
    assert len(submitting) == 4


def test_library_default_does_not_embed_project_account_policy():
    from types import SimpleNamespace

    from cluv.config import SubmitPolicy

    assert SubmitPolicy().exclude_accounts == []
    candidate = SimpleNamespace(sbatch_args={"account": "cc-debug_gpu"})
    assert dispatch.eligible_submission(candidate, None, [])
    assert not dispatch.eligible_submission(candidate, ["cc-debug_gpu"], ["cc-debug*"])


@pytest.mark.parametrize("cluster,expected", [("fir", "fir"), ("first", ["fir", "rorqual"])])
async def test_test_only_respects_explicit_single_cluster_policy(monkeypatch, cluster, expected):
    from types import SimpleNamespace
    from cluv.config import SubmitPolicy

    monkeypatch.setattr(
        dispatch,
        "get_cluv_config",
        lambda: SimpleNamespace(
            submit=SubmitPolicy(eligible_clusters=["fir", "rorqual"]),
            clusters_names=["fir", "rorqual"],
        ),
    )
    estimate = AsyncMock()
    monkeypatch.setattr(dispatch, "submit_test_only", estimate)
    await submit(cluster, None, [], [], test_only=True)
    assert estimate.call_args.args[0] == expected


def submission(tmp_path, account="ordinary"):
    from cluv.cache import Submission

    return Submission(
        cluster="fir",
        remote=None,
        job_script=tmp_path / "job.sh",
        sbatch_args={"account": account},
        program_args=[],
        sbatch_command="sbatch --parsable job.sh",
        n_chunks=None,
        git_commit="pin",
    )


@pytest.mark.parametrize(
    "output,code,known",
    [
        ("site notice\n123456;fir\n", 0, 123456),
        ("123456\n654321\n", 0, None),
        ("lost response", 0, None),
        ("123456;fir\n", 255, 123456),
        ("", 255, None),
    ],
)
async def test_raw_submission_receipt_precedes_parsing_and_uncertainty_is_visible(
    tmp_path, monkeypatch, output, code, known
):
    async def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, code, output, "raw stderr")

    monkeypatch.setattr(dispatch, "run", run)
    receipt = tmp_path / "receipt"
    if code == 0 and known:
        job = await dispatch.submit_job(submission(tmp_path), receipt)
        assert job.job_id == known
    else:
        with pytest.raises(dispatch.SubmissionOutcomeUnknown) as error:
            await dispatch.submit_job(submission(tmp_path), receipt)
        assert (error.value.job.job_id if error.value.job else None) == known
    raw = receipt.read_text()
    assert "status: RECEIVED" in raw and f"returncode: {code}" in raw
    assert output in raw and "raw stderr" in raw


@pytest.mark.parametrize("failure", ["transport", "timeout"])
async def test_transport_uncertainty_is_durable_and_bounded(tmp_path, monkeypatch, failure):
    async def run(*args, **kwargs):
        if failure == "timeout":
            await asyncio.sleep(30)
        raise ConnectionError("response lost after request")

    monkeypatch.setattr(dispatch, "run", run)
    monkeypatch.setattr(dispatch, "SUBMIT_TIMEOUT_SECONDS", 0.01)
    receipt = tmp_path / "receipt"
    with pytest.raises(dispatch.SubmissionOutcomeUnknown, match="do not retry"):
        await dispatch.submit_job(submission(tmp_path), receipt)
    assert "UNKNOWN submission transport outcome" in receipt.read_text()


@pytest.mark.parametrize("failed_write", ["RECEIVED", "SUBMITTED", "both"])
async def test_submit_receipt_io_failure_preserves_returned_candidate_id(
    tmp_path, monkeypatch, failed_write
):
    async def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "123456;fir\n", "site notice")

    monkeypatch.setattr(dispatch, "run", run)
    original_write = dispatch.write_submission_log

    def write(path, **fields):
        status = fields.get("status", "SUBMITTED")
        if failed_write in (status, "both"):
            raise OSError("receipt storage unavailable")
        return original_write(path, **fields)

    monkeypatch.setattr(dispatch, "write_submission_log", write)
    with pytest.raises(dispatch.SubmissionOutcomeUnknown) as error:
        await dispatch.submit_job(submission(tmp_path), tmp_path / "receipt")
    assert error.value.job.job_id == 123456
    assert any("receipt storage unavailable" in note for note in error.value.__notes__)
    assert any("123456;fir" in note for note in error.value.__notes__)


@pytest.mark.parametrize("receipt_failure", [False, True])
async def test_later_successful_sibling_is_recorded_before_unexpected_error(
    tmp_path, monkeypatch, receipt_failure
):
    import datetime
    from cluv.cache import Job

    rows = [
        dispatch.SubmissionProgress(submission(tmp_path, account), tmp_path / account)
        for account in ("first", "second")
    ]

    async def submit_variant(spec, path):
        if spec.sbatch_args["account"] == "first":
            raise ValueError("post-submit parse failure")
        return Job(**vars(spec), job_id=123456, submitted_at=datetime.datetime.now())

    monkeypatch.setattr(dispatch, "submit_job", submit_variant)
    if receipt_failure:
        from pathlib import Path

        original_open = Path.open

        def open_receipt(path, *args, **kwargs):
            if path == rows[0].log_path:
                raise OSError("receipt disk unavailable")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_receipt)
    with pytest.raises(ExceptionGroup, match="reconciliation"):
        await dispatch.wait_for_first_running_job(
            {"fir": rows},
            cluster_to_remote={"fir": None},
            found_running_job=asyncio.Event(),
            _skip_sync=True,
            sync_datasets=False,
            fetch_results=False,
            initial_delay=0,
        )
    assert rows[1].job_id == 123456
    assert rows[0].state == "UNKNOWN (submission outcome)"
    if not receipt_failure:
        assert "post-submit parse failure" in rows[0].log_path.read_text()


async def test_observed_winner_drains_delayed_sbatch_before_returning_ids(tmp_path, monkeypatch):
    import dataclasses
    import datetime
    from cluv.cache import Job

    rows = {
        name: [
            dispatch.SubmissionProgress(
                dataclasses.replace(submission(tmp_path), cluster=name), tmp_path / name
            )
        ]
        for name in ("fir", "rorqual")
    }
    completed = []

    async def variant(spec, path):
        if spec.cluster == "rorqual":
            await asyncio.sleep(0.005)
        completed.append(spec.cluster)
        return Job(
            **vars(spec),
            job_id=10 if spec.cluster == "fir" else 20,
            submitted_at=datetime.datetime.now(),
        )

    async def states(remote, ids, **kwargs):
        return ["RUNNING"] * len(ids)

    monkeypatch.setattr(dispatch, "submit_job", variant)
    monkeypatch.setattr(dispatch, "get_job_states_with_sacct", states)
    winner = await dispatch.wait_for_first_running_job(
        rows,
        cluster_to_remote={"fir": None, "rorqual": None},
        found_running_job=asyncio.Event(),
        _skip_sync=True,
        sync_datasets=False,
        fetch_results=False,
        initial_delay=0,
    )
    assert winner.job_id == 10
    assert rows["rorqual"][0].job_id == 20
    assert completed == ["fir", "rorqual"]


@pytest.mark.parametrize("state", ["BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"])
async def test_all_terminal_failures_end_the_race(tmp_path, monkeypatch, state):
    import datetime
    from cluv.cache import Job

    row = dispatch.SubmissionProgress(submission(tmp_path), tmp_path / "receipt")

    async def submit_variant(spec, path):
        return Job(**vars(spec), job_id=123, submitted_at=datetime.datetime.now())

    async def states(*args, **kwargs):
        return [state]

    monkeypatch.setattr(dispatch, "submit_job", submit_variant)
    monkeypatch.setattr(dispatch, "get_job_states_with_sacct", states)
    result = await asyncio.wait_for(
        dispatch.wait_for_first_running_job(
            {"fir": [row]},
            cluster_to_remote={"fir": None},
            found_running_job=asyncio.Event(),
            _skip_sync=True,
            sync_datasets=False,
            fetch_results=False,
            initial_delay=0,
        ),
        timeout=1,
    )
    assert result is None and row.state == state


@pytest.mark.parametrize("failure", ["parse", "poll", "cleanup"])
async def test_race_failure_cleans_all_known_siblings_without_reporting_success(
    monkeypatch,
    project_dir,  # noqa: F811
    failure,
):
    (project_dir / "pyproject.toml").write_text(
        '[tool.cluv]\nresults_path="results"\n[tool.cluv.clusters.fir]\nsbatch_args=[{account="first"},{account="second"}]\n'
    )
    script = project_dir / "job.sh"
    script.write_text("#!/bin/sh\necho work\n")
    receipts = project_dir / "receipts"
    receipts.mkdir()
    monkeypatch.setattr(dispatch, "get_submission_log_dir", lambda: receipts)
    monkeypatch.setattr(dispatch, "ensure_clean_git_state", lambda **_: "pin")
    monkeypatch.setattr(dispatch, "get_cluster_to_remote", AsyncMock(return_value={"fir": None}))
    cancelled = []
    polled = False
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _: real_sleep(0))

    async def run(argv, **kwargs):
        nonlocal polled
        command = shlex.join(argv)
        if "sbatch --parsable" in command:
            output = (
                ("unresolved response" if failure != "poll" else "100")
                if "--account=first" in command
                else "200"
            )
        elif "scancel" in command:
            cancelled.extend(int(value) for value in argv[1:])
            if failure == "cleanup":
                raise RuntimeError("cleanup denied")
            output = ""
        else:
            if failure == "poll" and not polled:
                polled = True
                raise ConnectionError("poll failed")
            ids = command.split("sacct -j ")[1].split()[0].split(",")
            output = "\n".join(
                f"{value}|{'CANCELLED' if int(value) in cancelled and failure != 'cleanup' else 'RUNNING'}"
                for value in ids
            )
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(dispatch, "run", run)
    monkeypatch.setattr(slurm, "run", run)
    real_cleanup = dispatch.wait_for_jobs_to_cancel

    async def bounded_cleanup(rows, remotes):
        return await real_cleanup(rows, remotes, max_wait_time_seconds=0.03, initial_delay=0.001)

    monkeypatch.setattr(dispatch, "wait_for_jobs_to_cancel", bounded_cleanup)
    monkeypatch.setattr(
        dispatch, "save_job", lambda _: pytest.fail("Must not report/save race success")
    )
    with pytest.raises((ExceptionGroup, ConnectionError)) as error:
        await submit("fir", script, [], [], _skip_sync=True)
    assert 200 in cancelled
    if failure == "poll":
        assert 100 in cancelled
    if failure == "cleanup":
        assert "Known-job cleanup also failed" in str(error.value.__notes__)
    assert all(job_id in (100, 200) for job_id in cancelled)
