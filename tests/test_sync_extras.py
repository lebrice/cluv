"""Unit tests for the `--extra` / `--all-extras` flags of `cluv sync`, which are passed down to
the `uv sync` run on each cluster."""

from __future__ import annotations

import importlib
from pathlib import PurePosixPath
from unittest import mock

import pytest

import cluv.__main__
from cluv.cache import ProjectStateOnCluster
from cluv.remote import Remote

sync_module = importlib.import_module("cluv.cli.sync")


@pytest.mark.parametrize(
    ("argv", "expected_kwargs"),
    [
        ([], {"extras": None, "all_extras": False}),
        (["--extra", "foo"], {"extras": ["foo"], "all_extras": False}),
        (["--extra", "foo", "--extra", "bar"], {"extras": ["foo", "bar"], "all_extras": False}),
        (["--all-extras"], {"extras": None, "all_extras": True}),
    ],
)
def test_cli_passes_extras_to_sync(
    argv: list[str], expected_kwargs: dict, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cluv.__main__, "sync", mock_sync := mock.AsyncMock())
    cluv.__main__.main(["sync", "mila", *argv])
    mock_sync.assert_awaited_once_with(clusters=["mila"], sync_datasets=True, **expected_kwargs)


def test_cli_extra_and_all_extras_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cluv.__main__, "sync", mock_sync := mock.AsyncMock())
    with pytest.raises(SystemExit):
        cluv.__main__.main(["sync", "--extra", "foo", "--all-extras"])
    mock_sync.assert_not_awaited()


@pytest.mark.parametrize(
    ("kwargs", "expected_suffix"),
    [
        ({}, ""),
        ({"extras": ["foo"]}, " --extra foo"),
        ({"extras": ["foo", "bar-baz"]}, " --extra foo --extra bar-baz"),
        ({"all_extras": True}, " --all-extras"),
    ],
)
async def test_run_uv_sync_passes_extras(kwargs: dict, expected_suffix: str):
    remote = mock.AsyncMock(Remote, hostname="foo")
    state = ProjectStateOnCluster()
    await sync_module.run_uv_sync(remote, PurePosixPath("/home/me/proj"), state, **kwargs)
    remote.run.assert_awaited_once_with(
        f"bash --login -c 'uv --directory=/home/me/proj sync --quiet{expected_suffix}'"
    )
    assert state.last_uv_sync_extras_flags == expected_suffix


async def test_run_uv_sync_reruns_when_extras_change():
    """`uv sync` already ran for this commit, but with other extras: it must run again (and then
    be skipped when nothing changed)."""
    remote = mock.AsyncMock(Remote, hostname="foo")
    path = PurePosixPath("/home/me/proj")
    state = ProjectStateOnCluster()

    await sync_module.run_uv_sync(remote, path, state)
    await sync_module.run_uv_sync(remote, path, state)
    assert remote.run.await_count == 1  # Same commit, same (no) extras: skipped.

    await sync_module.run_uv_sync(remote, path, state, extras=["foo"])
    assert remote.run.await_count == 2
    await sync_module.run_uv_sync(remote, path, state, extras=["foo"])
    assert remote.run.await_count == 2

    await sync_module.run_uv_sync(remote, path, state)  # Back to none: removes the extra.
    assert remote.run.await_count == 3


async def test_run_uv_sync_cache_from_before_extras_support_is_not_resynced():
    """A cache written before this field existed (None) means `uv sync` ran without extras."""
    remote = mock.AsyncMock(Remote, hostname="foo")
    state = ProjectStateOnCluster(
        last_uv_sync_git_commit=sync_module.subprocess.getoutput("git rev-parse HEAD").strip()
    )
    await sync_module.run_uv_sync(remote, PurePosixPath("/proj"), state)
    remote.run.assert_not_awaited()


@pytest.mark.parametrize("bad_name", ["foo'; rm -rf ~; echo '", "foo bar", "-foo", ""])
def test_invalid_extra_names_are_rejected(bad_name: str):
    with pytest.raises(ValueError, match="Invalid extra name"):
        sync_module._uv_sync_extras_flags([bad_name], all_extras=False)


async def test_sync_passes_extras_to_each_cluster(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sync_module, "get_disabled_clusters", mock.Mock(return_value={}))
    monkeypatch.setattr(sync_module, "current_cluster", mock.Mock(return_value=None))
    monkeypatch.setattr(
        sync_module,
        "get_cluster_to_remote",
        mock.AsyncMock(return_value={"a": Remote(hostname="a"), "b": Remote(hostname="b")}),
    )
    monkeypatch.setattr(sync_module, "sync_common_part", mock.AsyncMock())
    monkeypatch.setattr(
        sync_module, "sync_per_cluster_part", mock_per_cluster := mock.AsyncMock(return_value=[])
    )

    await sync_module.sync(["a", "b"], extras=["foo"])

    assert mock_per_cluster.await_count == 2
    for call in mock_per_cluster.await_args_list:
        assert call.kwargs["extras"] == ["foo"]
        assert call.kwargs["all_extras"] is False


async def test_sync_rejects_bad_extra_before_touching_clusters(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sync_module, "get_disabled_clusters", mock_disabled := mock.Mock())
    with pytest.raises(ValueError):
        await sync_module.sync(["a"], extras=["no good"])
    mock_disabled.assert_not_called()
