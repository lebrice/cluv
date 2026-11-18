"""Exercise selective rsync and staged uv locally; all remote endpoints are fakes."""

import importlib
import os
import shlex
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cluv.cache import ProjectStateOnCluster
from cluv.config import CluvConfig, PartialClusterConfig

sync = importlib.import_module("cluv.cli.sync")


@pytest.mark.parametrize("fetch_results", [True, False])
def test_sync_cli_forwards_result_fetch_policy_to_preparation(monkeypatch, fetch_results):
    import cluv.__main__ as cli

    remote = SimpleNamespace(hostname="fir")
    monkeypatch.setattr(sync, "get_disabled_clusters", lambda: {})
    monkeypatch.setattr(sync, "current_cluster", lambda: None)
    monkeypatch.setattr(sync, "get_cluster_to_remote", AsyncMock(return_value={"fir": remote}))
    monkeypatch.setattr(sync, "sync_common_part", AsyncMock())
    prepare = AsyncMock()
    monkeypatch.setattr(sync, "sync_per_cluster_part", prepare)
    flags = [] if fetch_results else ["--no-fetch-results"]
    cli.main(["sync", "fir", *flags])
    prepare.assert_awaited_once_with(
        remote, sync_datasets=True, extras=None, all_extras=False, sync_results=fetch_results
    )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync required")
async def test_mila_controller_selective_data_precedes_portable_uv_without_site_update(
    tmp_path, monkeypatch
):
    source, target = tmp_path / "source", tmp_path / "target"
    uv = source / "runtime/uv/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        '#!/bin/sh\nif [ "$1" = --version ]; then echo "uv 0.13.8"; else printf "%s\\n" "$UV_PYTHON" "$UV_CACHE_DIR" "$UV_OFFLINE" > "$SCRATCH/uv-sync-env"; fi\n'
    )
    uv.chmod(0o755)
    (source / "models/Qwen3-0.6B").mkdir(parents=True)
    (source / "models/Qwen3-0.6B/weight").write_text("required weight")
    (source / "robot-checkpoints").mkdir()
    (source / "robot-checkpoints/secret").write_text("must not transfer")
    rules = tmp_path / "filter.rules"
    rules.write_text("+ /runtime/***\n+ /models/\n+ /models/Qwen3-0.6B/***\n- /***\n")
    config = CluvConfig(
        results_path=str(target / "results"),
        data_source=f"mila:{source}",
        datasets_path=str(target),
        manage_uv=False,
        datasets_rsync_filter=str(rules),
        env={
            "PATH": "$SCRATCH/runtime/uv/bin:$PATH",
            "UV_PYTHON": "$SCRATCH/runtime/python/bin/python3.12",
            "UV_CACHE_DIR": "$SCRATCH/runtime/cache",
            "UV_OFFLINE": "1",
        },
        clusters={
            "mila": PartialClusterConfig(datasets_path=str(tmp_path / "wrong-mirror")),
            "fir": PartialClusterConfig(project_dir=str(target / "repo")),
        },
    )
    monkeypatch.setattr(sync, "get_cluv_config", lambda: config)
    monkeypatch.setattr(sync, "current_cluster", lambda: "mila")
    monkeypatch.setattr(sync, "read_cache", lambda: SimpleNamespace(project_states={}))
    monkeypatch.setattr(sync, "save_project_state", lambda *_: None)
    installer = AsyncMock(side_effect=AssertionError("Never install/update site uv"))
    monkeypatch.setattr(sync, "install_uv", installer)
    monkeypatch.setattr(sync, "create_results_dir_with_symlink_to_scratch", AsyncMock())
    order = []

    async def clone(*args, **kwargs):
        assert (target / "runtime/uv/bin/uv").is_file()
        assert not (target / "robot-checkpoints").exists()
        order.append("clone")

    monkeypatch.setattr(sync, "clone_project", clone)

    async def run(argv, **kwargs):
        assert argv[0] == "rsync" and argv[-2] == f"{source}/"
        assert argv[argv.index("--filter") + 1] == f"merge {rules}"
        local = [*argv[:-1], argv[-1].removeprefix("fir:")]
        result = subprocess.run(local, capture_output=True, text=True, check=True)
        order.append("data")
        return result

    monkeypatch.setattr(sync, "run", run)
    target_env = dict(os.environ, SCRATCH=str(target))

    class Remote:
        hostname = "fir"

        async def run(self, command, **kwargs):
            result = subprocess.run(
                command, shell=True, env=target_env, capture_output=True, text=True, check=True
            )
            if " sync --quiet" in command:
                order.append("uv-sync")
            return result

        async def get_output(self, command, **kwargs):
            result = await self.run(command, **kwargs)
            if "command -v uv" in command:
                assert str(target / "runtime/uv/bin/uv") in result.stdout
                order.append("validate-uv")
            return result.stdout.strip()

    await sync.sync_per_cluster_part(Remote(), sync_results=False)
    assert order == ["data", "clone", "validate-uv", "uv-sync"]
    assert (target / "uv-sync-env").read_text().splitlines() == [
        str(target / "runtime/python/bin/python3.12"),
        str(target / "runtime/cache"),
        "1",
    ]
    assert (target / "models/Qwen3-0.6B/weight").read_text() == "required weight"
    installer.assert_not_called()


@pytest.mark.parametrize("problem", ["filter", "scratch"])
async def test_unresolved_controller_inputs_fail_before_git_push_or_transfer(
    tmp_path, monkeypatch, problem
):
    rules = tmp_path / "filter"
    rules.write_text("- /***\n")
    config = CluvConfig(
        results_path="results",
        data_source="mila:/required",
        datasets_path="$SCRATCH/data",
        datasets_rsync_filter="$UNKNOWN_FILTER/rules" if problem == "filter" else str(rules),
    )
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.delenv("UNKNOWN_FILTER", raising=False)
    monkeypatch.setattr(sync, "get_cluv_config", lambda: config)
    monkeypatch.setattr(sync, "current_cluster", lambda: None)
    monkeypatch.setattr(sync, "get_datasets_path", lambda: "$SCRATCH/data")
    push = AsyncMock(side_effect=AssertionError("No git push"))
    pull = AsyncMock(side_effect=AssertionError("No data transfer"))
    monkeypatch.setattr(sync, "run_git_push_if_needed", push)
    monkeypatch.setattr(sync, "pull_datasets_if_needed", pull)
    with pytest.raises(RuntimeError, match="controller|Controller"):
        await sync.sync_common_part([])
    push.assert_not_called()
    pull.assert_not_called()


async def test_source_host_does_not_copy_dataset_to_itself(tmp_path, monkeypatch):
    config = CluvConfig(
        results_path="results",
        data_source=f"mila:{tmp_path}",
        datasets_path=str(tmp_path / "different-target"),
    )
    monkeypatch.setattr(sync, "current_cluster", lambda: "mila")
    transfer = AsyncMock(side_effect=AssertionError("No self-copy"))
    monkeypatch.setattr(sync, "_push_datasets_to_remote", transfer)
    await sync.push_datasets_if_needed(
        SimpleNamespace(hostname="mila"), config, ProjectStateOnCluster()
    )
    transfer.assert_not_called()


async def test_desktop_does_not_push_its_mirror_back_over_authoritative_source(
    tmp_path, monkeypatch
):
    config = CluvConfig(
        results_path="results",
        data_source=f"mila:{tmp_path}",
        datasets_path=str(tmp_path / "desktop-mirror"),
        clusters={"mila": PartialClusterConfig(datasets_path="$SOURCE_ASSETS")},
    )
    monkeypatch.setattr(sync, "current_cluster", lambda: None)
    transfer = AsyncMock(side_effect=AssertionError("Never overwrite authoritative source"))
    monkeypatch.setattr(sync, "_push_datasets_to_remote", transfer)
    remote = SimpleNamespace(hostname="mila", get_output=AsyncMock(return_value=str(tmp_path)))
    await sync.push_datasets_if_needed(remote, config, ProjectStateOnCluster())
    transfer.assert_not_called()


async def test_filter_applies_to_pull_and_prevents_stale_root_mtime_skip(tmp_path, monkeypatch):
    import datetime

    rules = tmp_path / "filter"
    rules.write_text("+ /required/***\n- /***\n")
    config = CluvConfig(
        results_path="results",
        datasets_path=str(tmp_path / "target"),
        datasets_rsync_filter=str(rules),
        clusters={"fir": PartialClusterConfig()},
    )
    monkeypatch.setattr(sync, "get_cluv_config", lambda: config)
    run = AsyncMock()
    monkeypatch.setattr(sync, "run", run)
    remote = SimpleNamespace(hostname="fir", run=AsyncMock())
    await sync._pull_datasets(remote, "/selected/source", tmp_path / "mirror")
    args = run.call_args.args[0]
    assert args[args.index("--filter") + 1] == f"merge {rules}"
    state = ProjectStateOnCluster(
        last_pushed_datasets=datetime.datetime.now() + datetime.timedelta(days=1)
    )
    await sync._push_datasets_to_remote(tmp_path / "mirror", remote, config, state)
    assert run.await_count == 2
    args = run.call_args.args[0]
    assert args[args.index("--filter") + 1] == f"merge {rules}"


async def test_local_controller_uses_staged_checkout_and_preserves_older_checkout(
    tmp_path, monkeypatch
):
    from cluv.cli.submit import get_submissions

    submit_module = importlib.import_module("cluv.cli.submit")

    old, staged = tmp_path / "older-home", tmp_path / "staged"
    trees = []
    for path, text in ((old, "older-body"), (staged, "reviewed-body")):
        path.mkdir()
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        blob = subprocess.run(
            ["git", "-C", str(path), "hash-object", "-w", "--stdin"],
            input=text,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "update-index",
                "--add",
                "--cacheinfo",
                f"100644,{blob},scripts/shared.sh",
            ],
            check=True,
        )
        trees.append(
            subprocess.run(
                ["git", "-C", str(path), "write-tree"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    (staged / "pyproject.toml").write_text('[tool.cluv]\nresults_path="results"\n')
    (staged / "header.sh").write_text(
        '#!/bin/sh\n#SBATCH --nodes=1\n#SBATCH hetjob\n#SBATCH --nodes=1\nset -eu\npayload=$(git show "${GIT_COMMIT}:scripts/shared.sh")\nprintf "%s" "$payload" > "$TRACE_PROOF"\n'
    )
    binary = tmp_path / "sbatch"
    binary.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$TRACE_ARGS"\npwd > "$TRACE_CWD"\ncat > "$TRACE_SCRIPT"\nsh "$TRACE_SCRIPT"\n'
    )
    binary.chmod(0o755)
    config = CluvConfig(
        results_path=str(tmp_path / "results"),
        project_dir=str(old),
        job_script_path="header.sh",
        env={
            "PATH": f"{tmp_path}:$PATH",
            **{
                f"TRACE_{name.upper()}": str(tmp_path / name)
                for name in ("args", "cwd", "script", "proof")
            },
        },
        clusters={"mila": PartialClusterConfig()},
    )
    monkeypatch.setattr(submit_module, "get_cluv_config", lambda: config)
    monkeypatch.chdir(staged)
    jobs = await get_submissions(
        "mila",
        None,
        job_script=None,
        sbatch_args=[],
        program_args=["--train-data", "/data/train.parquet"],
        chunking=None,
        vram=None,
        git_commit=trees[1],
    )
    result = subprocess.run(
        shlex.split(jobs[0].sbatch_command), capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "proof").read_text() == "reviewed-body"
    assert (tmp_path / "cwd").read_text().strip() == str(staged)
    assert f"--chdir={staged}" in (tmp_path / "args").read_text().splitlines()
    assert (
        subprocess.run(
            ["git", "-C", str(old), "show", f"{trees[0]}:scripts/shared.sh"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        == "older-body"
    )
    assert (
        subprocess.run(
            ["git", "-C", str(old), "cat-file", "-e", trees[1]], capture_output=True
        ).returncode
        != 0
    )


def test_home_relative_desktop_transport_never_expands_remote_paths_locally(tmp_path, monkeypatch):
    from pathlib import Path
    from cluv.config import load_cluv_config

    config_module = importlib.import_module("cluv.config")

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(config_module, "current_cluster", lambda: None)
    project = tmp_path / "pyproject.toml"
    project.write_text("""[tool.cluv]
results_path = "$SCRATCH/results"
data_source = "mila:/authoritative/source"
datasets_path = "$SCRATCH/datasets/large-scale-jobs"
datasets_rsync_filter = "$HOME/.cache/large-scale-jobs/portable-sync-filter.rules"
manage_uv = false
[tool.cluv.local.env]
SCRATCH = "$HOME/.cache/large-scale-jobs"
[tool.cluv.env]
UV_PYTHON = "$SCRATCH/runtime/python/bin/python3.12"
PATH = "$SCRATCH/runtime/uv/bin:$PATH"
[tool.cluv.clusters.fir]
""")
    cfg = load_cluv_config(project)
    home = os.environ["HOME"]
    assert os.environ["SCRATCH"] == f"{home}/.cache/large-scale-jobs"
    assert (
        cfg.get_cluster_config("fir").datasets_path.as_posix()
        == "$SCRATCH/datasets/large-scale-jobs"
    )
    assert cfg.env["UV_PYTHON"] == "$SCRATCH/runtime/python/bin/python3.12"
    original_is_file = Path.is_file
    expected_filter = Path(home) / ".cache/large-scale-jobs/portable-sync-filter.rules"
    monkeypatch.setattr(
        Path, "is_file", lambda path: path == expected_filter or original_is_file(path)
    )
    assert sync.dataset_filter_args(cfg) == ["--filter", f"merge {expected_filter}"]
    target = str(tmp_path / "actual-target-scratch")
    command = sync.target_environment(cfg.env) + 'printf "%s" "$UV_PYTHON"'
    result = subprocess.run(
        ["bash", "--login", "-c", command],
        env=dict(os.environ, SCRATCH=target),
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == f"{target}/runtime/python/bin/python3.12"
    assert os.environ["HOME"] == home
