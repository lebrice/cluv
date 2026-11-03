"""`cluv submit --test-only`: ask Slurm when each job configuration would start, submit nothing.

`sbatch --test-only` validates a job against the cluster (partition, account, QoS, limits) and
prints a forecast to *stderr*, without queueing anything. There is no job id on stdout, which is
why this can't go through `submit_job`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import re
import shlex
import sys

import rich.box
import rich.table
import rich.text

from cluv.cache import Submission
from cluv.remote import run
from cluv.sbatch_args import SbatchArgs, sbatch_args_to_list
from cluv.utils import console

# sbatch: Job 123 to start at 2026-11-03T17:00:00 using 4 processors on nodes cn-a001 in partition main
# Observed variants: Mila prints a stray token after the time ("...T17:06:26 a using 1 ..."), and
# for heterogeneous jobs there is neither "nodes" nor a partition ("using 2 processors on cn-c036").
_FORECAST_RE = re.compile(
    r"Job (?P<job_id>\d+) to start at (?P<start>\S+)"
    r"(?:.*?using (?P<cpus>\d+) processors?)?"
    r"(?: on (?:nodes? )?(?P<nodes>\S+))?"
    r"(?: in partition (?P<partition>\S+))?"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclasses.dataclass(frozen=True)
class TestOnlyResult:
    """What `sbatch --test-only` said about one job configuration."""

    __test__ = False  # not a pytest test class

    cluster: str
    config: str
    """Short description of what distinguishes this configuration (account, GPU type, ...)."""
    start: datetime.datetime | None = None
    cpus: int | None = None
    nodes: str | None = None
    partition: str | None = None
    error: str | None = None
    """Slurm's error message if the configuration is rejected (or its output, if unparsable)."""

    @property
    def accepted(self) -> bool:
        return self.error is None


def parse_test_only_output(output: str) -> dict[str, object] | None:
    """Parse the forecast line printed by `sbatch --test-only`, or return None if there is none.

    >>> parse_test_only_output(
    ...     "sbatch: Job 1 to start at 2026-11-03T17:00:00 using 4 processors on nodes cn-a001 "
    ...     "in partition main"
    ... )
    {'start': datetime.datetime(2026, 11, 3, 17, 0), 'cpus': 4, 'nodes': 'cn-a001', 'partition': 'main'}
    >>> parse_test_only_output("sbatch: error: Batch job submission failed") is None
    True
    """
    for line in output.splitlines():
        if match := _FORECAST_RE.search(line):
            try:
                start = datetime.datetime.fromisoformat(match["start"])
            except ValueError:
                continue
            return {
                "start": start,
                "cpus": int(match["cpus"]) if match["cpus"] else None,
                "nodes": match["nodes"],
                "partition": match["partition"],
            }
    return None


def clean_sbatch_error(output: str) -> str:
    """Strip the `sbatch: error:` prefixes from sbatch's error output.

    >>> clean_sbatch_error("sbatch: error: Batch job submission failed: Invalid account\\n")
    'Batch job submission failed: Invalid account'
    >>> clean_sbatch_error("sbatch: error: \x1b[00;36m-----\x1b[00m\\nallocation failure: Invalid account\\n")
    'allocation failure: Invalid account'
    """
    lines = [
        re.sub(r"^sbatch: (error: )?", "", _ANSI_RE.sub("", line)).strip()
        for line in output.splitlines()
    ]
    # Drop the banner some clusters (Mila) print around the real message.
    lines = [line for line in lines if line and not re.fullmatch(r"-+|# .*", line)]
    return "; ".join(lines)


def describe_config(sbatch_args: SbatchArgs) -> str:
    """Short description of the flags that tell the configurations of one cluster apart."""
    keys = ("account", "gres", "gpus", "partition", "constraint", "qos", "time")
    return " ".join(f"{k}={sbatch_args[k]}" for k in keys if k in sbatch_args) or "-"


def get_test_only_command(sbatch_args: SbatchArgs) -> str:
    """The command that runs `sbatch --test-only` on the cluster.

    The job script is not named: `sbatch` reads it from stdin (see `test_only_submission`), so
    nothing has to be synced to the cluster first. Program args are left out too, they have no
    effect on the forecast.
    """
    from cluv.cli.submit import double_quote  # avoid a circular import

    flags = " ".join(map(double_quote, sbatch_args_to_list(sbatch_args)))
    return f"bash --login -c {shlex.quote(f'sbatch --test-only {flags}')}"


async def test_only_submission(submission: Submission) -> TestOnlyResult:
    """Run `sbatch --test-only` for this submission, feeding it the local job script on stdin."""
    command = get_test_only_command(submission.sbatch_args)
    script = submission.job_script.read_text()
    kwargs = dict(cluster=submission.cluster, config=describe_config(submission.sbatch_args))
    if submission.remote is not None:
        result = await submission.remote.run(
            command, input=script, display=False, warn=True, hide=True
        )
    else:
        result = await run(
            tuple(shlex.split(command)), input=script, warn=True, hide=True, _display=False
        )
    output = f"{result.stderr}\n{result.stdout}"
    if result.returncode == 0 and (forecast := parse_test_only_output(output)):
        return TestOnlyResult(**kwargs, **forecast)  # type: ignore[arg-type]
    if result.returncode == 0:
        return TestOnlyResult(
            **kwargs, error=f"accepted, but no start time in sbatch's output: {output.strip()!r}"
        )
    return TestOnlyResult(
        **kwargs, error=clean_sbatch_error(output) or f"exit {result.returncode}"
    )


def format_delta(delta: datetime.timedelta) -> str:
    """
    >>> format_delta(datetime.timedelta(seconds=30))
    'now'
    >>> format_delta(datetime.timedelta(hours=2, minutes=5))
    'in 2h05m'
    >>> format_delta(datetime.timedelta(days=1, hours=3))
    'in 1d03h'
    """
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "now"
    minutes, _ = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"in {days}d{hours:02d}h"
    if hours:
        return f"in {hours}h{minutes:02d}m"
    return f"in {minutes}m"


def render_test_only_table(
    results: list[TestOnlyResult], now: datetime.datetime | None = None
) -> rich.table.Table:
    now = now or datetime.datetime.now()
    table = rich.table.Table(
        "Cluster",
        "Configuration",
        "Estimated start",
        "Nodes",
        "Partition",
        title="sbatch --test-only (nothing was submitted)",
        box=rich.box.ROUNDED,
        show_lines=True,
        header_style="bold white on #1a1a2e",
        title_style="bold cyan",
    )
    for r in results:
        if r.error is not None:
            table.add_row(
                r.cluster, r.config, rich.text.Text(f"REJECTED: {r.error}", style="red"), "", ""
            )
            continue
        assert r.start is not None
        start = f"{r.start:%Y-%m-%d %H:%M:%S} ({format_delta(r.start - now)})"
        nodes = f"{r.nodes} ({r.cpus} cpus)" if r.cpus is not None else (r.nodes or "-")
        table.add_row(r.cluster, r.config, start, nodes, r.partition or "-")
    return table


async def test_only(submissions: list[Submission]) -> list[TestOnlyResult]:
    """Run `sbatch --test-only` for every submission, in parallel, and print the table.

    Exits with status 1 if any configuration is rejected, so this can gate a CI job.
    """
    results = list(await asyncio.gather(*map(test_only_submission, submissions)))
    console.print(render_test_only_table(results))
    if not all(r.accepted for r in results):
        sys.exit(1)
    return results
