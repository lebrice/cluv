"""Exercise the submit CLI and actual resource/argv rendering without SSH."""

import asyncio
import copy
import importlib
import shlex
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cluv.__main__ as cli
from cluv.config import get_cluv_config
from cluv.site_overrides import load_site_overrides

dispatch = importlib.import_module("cluv.cli.submit")
SITES = ["rorqual", "nibi", "tamia", "trillium-gpu"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    text = '[project]\nname="overlay-fixture"\nversion="1.0.0"\n[tool.cluv]\nresults_path="$SCRATCH/logs"\nproject_dir="$HOME/repos/fixture"\n[tool.cluv.submit]\nexclude_accounts=["cc-debug*"]\n'
    for name in SITES:
        accounts = '[{account="ordinary",mem="32G"},{account="cc-debug_gpu",mem="32G"}]'
        text += (
            f'[tool.cluv.clusters.{name}]\njob_script_path="{name}.sh"\nsbatch_args={accounts}\n'
        )
        (tmp_path / f"{name}.sh").write_text(
            '#!/bin/bash\n#SBATCH -N 2 -n 2 -G 2 --mem-per-cpu=4G --time=00:10:00\n#SBATCH hetjob\n#SBATCH -N 1 -n 2 --gpus-per-node=h100:2 --mem-per-cpu=4G --time=00:10:00\nprintf "%s\\0" "$@"\n'
        )
    (tmp_path / "pyproject.toml").write_text(text)
    get_cluv_config.cache_clear()
    return tmp_path


def write_overlay(path):
    text = ""
    for site in SITES:
        first = 2 if site in ["rorqual", "nibi"] else 4
        root = "/scratch/n/normandf" if site == "tamia" else "/scratch/normandf"
        text += f'[sites.{site}]\nprogram_args_prefix=["--model","{root}/models/Qwen3-8B"]\n'
        for gpu, tasks, mem in [(first, 1, "64G"), (4, 4, "128G")]:
            memory = f'"{mem}"' if site in ["rorqual", "nibi"] else "false"
            text += f'[[sites.{site}.resources]]\nnodes=1\nntasks-per-node={tasks}\ngpus-per-node="h100:{gpu}"\ncpus-per-task=8\nmem={memory}\ntime="00:20:00"\n'
    path.write_text(text)
    return path


def script_groups(command):
    inner = shlex.split(command)[-1]
    groups = []
    for line in inner.splitlines():
        if line.startswith("#SBATCH") and line != "#SBATCH hetjob":
            groups.append(
                dispatch.sbatch_args_from_list(shlex.split(line.removeprefix("#SBATCH")))
            )
    return groups


async def test_mixed_shapes_all_headers_accounts_copies_and_literal_argv(project):
    path = write_overlay(project / "race.toml")
    overrides = load_site_overrides(path, SITES)
    original = copy.deepcopy(overrides.sites)
    common = ["--environment=gsm8k", "--tensor-parallel-size", "2", "--", "--max-steps", "128"]
    for site in SITES:
        jobs = await dispatch.get_submissions(
            site,
            SimpleNamespace(hostname=site),
            job_script=None,
            sbatch_args=["--job-name=one-race", "--time=00:20:00"],
            program_args=common,
            chunking=None,
            git_commit="pinned",
            overrides=overrides,
        )
        assert len(jobs) == 2
        assert [j.sbatch_args["account"] for j in jobs] == ["ordinary", "cc-debug_gpu"]
        assert [j for j in jobs if dispatch.eligible_submission(j, None, ["cc-debug*"])] == [
            jobs[0]
        ]
        groups = script_groups(jobs[0].sbatch_command)
        assert len(groups) == 2
        for index, group in enumerate(groups):
            assert group["nodes"] == "1" and group["ntasks-per-node"] == str(
                1 if index == 0 else 4
            )
            assert (
                group["gpus-per-node"]
                == f"h100:{(2 if site in ['rorqual', 'nibi'] else 4) if index == 0 else 4}"
            )
            assert group["cpus-per-task"] == "8" and group["time"] == "00:20:00"
            assert group["account"] == "ordinary" and group["job-name"] == "cluv-one-race"
            assert not {"n", "G", "ntasks", "gpus", "gres", "mem-per-cpu"} & group.keys()
            if site in ["rorqual", "nibi"]:
                assert group["mem"] == ("64G" if index == 0 else "128G")
            else:
                assert "mem" not in group
        assert jobs[0].program_args[:2] == [
            "--model",
            ("/scratch/n/normandf" if site == "tamia" else "/scratch/normandf")
            + "/models/Qwen3-8B",
        ]
        jobs[0].sbatch_args["__het_components__"][0]["nodes"] = 999
        jobs[0].program_args.append("MUTATED")
        assert jobs[1].sbatch_args["__het_components__"][0]["nodes"] == 1
        assert "MUTATED" not in jobs[1].program_args
    assert overrides.sites == original and common[-1] == "128"


def test_cli_reads_once_private_snapshot_and_old_default(project, monkeypatch):
    path = write_overlay(project / "race.toml")
    raw = path.read_bytes()
    receipts = project / "receipts"
    receipts.mkdir()
    monkeypatch.setattr(dispatch, "get_submission_log_dir", lambda: receipts)
    monkeypatch.setattr(dispatch, "ensure_clean_git_state", lambda **_: "pin")
    monkeypatch.setattr(dispatch, "sync_common_part", AsyncMock())

    async def connect(_):
        path.write_text("INVALID AFTER SNAPSHOT")
        return {"rorqual": SimpleNamespace(hostname="rorqual")}

    monkeypatch.setattr(dispatch, "get_cluster_to_remote", connect)
    captured = []

    async def wait(rows, **kwargs):
        captured.extend(row for values in rows.values() for row in values)
        return None

    monkeypatch.setattr(dispatch, "wait_for_first_running_job", wait)
    call = AsyncMock(side_effect=dispatch.submit)
    monkeypatch.setattr(cli, "submit", call)
    cli.main(
        [
            "submit",
            "first",
            "--site-overrides",
            str(path),
            "--eligible-clusters=rorqual",
            "--no-sync-datasets",
            "--no-fetch-results",
            "--",
            "--environment=gsm8k",
        ]
    )
    assert len(captured) == 1 and captured[0].job.sbatch_args["gpus-per-node"] == "h100:2"
    snapshots = list(receipts.glob("site-overrides-*.toml"))
    assert len(snapshots) == 1
    assert snapshots[0].read_bytes() == raw and stat.S_IMODE(snapshots[0].stat().st_mode) == 0o600
    assert "site_overrides_sha256:" in captured[0].log_path.read_text()
    assert str(snapshots[0]) in captured[0].log_path.read_text()
    ordinary = asyncio.run(
        dispatch.get_submissions(
            "rorqual",
            SimpleNamespace(hostname="rorqual"),
            job_script=None,
            sbatch_args=[],
            program_args=["plain"],
            chunking=None,
            git_commit="pin",
        )
    )
    assert ordinary[0].program_args == ["plain"] and ordinary[0].sbatch_args["nodes"] == "2"


@pytest.mark.parametrize(
    "bad",
    [
        "[sites.unknown]\nprogram_args_prefix=[]",
        '[sites.rorqual]\nenv={GIT_COMMIT="spoof"}',
        "[sites.rorqual]\nunknown_field=true",
        '[sites.rorqual]\nprogram_args_prefix=["--"]',
        '[[sites.rorqual.resources]]\naccount="cc-debug_gpu"',
        '[[sites.rorqual.resources]]\npartition="debug"',
        "[[sites.rorqual.resources]]\nnodes=1",
    ],
)
async def test_invalid_overlay_fails_before_transport_or_project_mutation(
    project, monkeypatch, bad
):
    path = project / "bad.toml"
    path.write_text(bad)
    remote = AsyncMock(side_effect=AssertionError("must not contact transport"))
    monkeypatch.setattr(dispatch, "get_cluster_to_remote", remote)
    monkeypatch.setattr(
        dispatch, "ensure_clean_git_state", lambda **_: pytest.fail("must not commit")
    )
    with pytest.raises(ValueError):
        await dispatch.submit("first", None, [], [], site_overrides=path)
    remote.assert_not_called()


async def test_excluded_account_cannot_be_reintroduced_before_transport(project, monkeypatch):
    path = write_overlay(project / "race.toml")
    remote = AsyncMock(side_effect=AssertionError("must not contact transport"))
    monkeypatch.setattr(dispatch, "get_cluster_to_remote", remote)
    with pytest.raises(ValueError, match="No eligible"):
        await dispatch.submit("rorqual", None, ["--account=cc-debug_gpu"], [], site_overrides=path)
    remote.assert_not_called()


def test_cli_explicit_resource_layer_removes_inherited_aliases(project):
    site = load_site_overrides(write_overlay(project / "race.toml"), SITES).for_site("rorqual")
    resources = dispatch.merge_job_resources(
        project / "rorqual.sh",
        {"account": "ordinary"},
        ["-n", "4", "-G", "4", "--mem-per-cpu=8G"],
        site.resources,
    )
    for group in [resources, *resources["__het_components__"]]:
        assert group["ntasks"] == "4" and group["gpus"] == "4" and group["mem-per-cpu"] == "8G"
        assert not {"n", "G", "gpus-per-node", "ntasks-per-node", "mem"} & group.keys()


async def test_prefix_is_literal_in_executed_stdin_script(project):
    import json
    import subprocess

    marker = project / "must-not-exist"
    literal = ["--model", f"a path's model; $(touch {marker})", "--value=a&b"]
    path = project / "literal.toml"
    path.write_text("[sites.rorqual]\nprogram_args_prefix=" + json.dumps(literal))
    jobs = await dispatch.get_submissions(
        "rorqual",
        SimpleNamespace(hostname="rorqual"),
        job_script=None,
        sbatch_args=[],
        program_args=["common", "--", "trainer"],
        chunking=None,
        git_commit="pin",
        overrides=load_site_overrides(path, SITES),
    )
    inner = shlex.split(jobs[0].sbatch_command)[-1]
    script = inner.split("\n", 1)[1].rsplit("\n", 1)[0]
    result = subprocess.run(["bash"], input=script.encode(), capture_output=True, check=True)
    assert result.stdout.split(b"\0")[:-1] == [
        arg.encode() for arg in [*literal, "common", "--", "trainer"]
    ]
    assert not marker.exists()
