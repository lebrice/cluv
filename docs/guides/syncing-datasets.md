# Syncing datasets across clusters

## Portable tool bundles and selective inputs

Remote data sources pass through a controller-local mirror. Keep
`data_source = "mila:/absolute/source/root"` authoritative, and bound the mirror
with reviewed filters and a capacity check. An existing controller running on
the source host instead uses that actual path and skips self-copy. Target paths
remain per-cluster `datasets_path`; this is not a direct remote-to-remote service.

```toml
[tool.cluv]
data_source = "mila:/absolute/source/root"
datasets_path = "$SCRATCH/datasets/project"
datasets_rsync_filter = "$HOME/.cache/project/portable-sync-filter.rules"
datasets_copy_links = false
manage_uv = false

[tool.cluv.local.env]
SCRATCH = "$HOME/.cache/project"
```

`datasets_rsync_filter` is a controller-local rsync merge-rules file, applied to
both pull and push. Use reviewed allow rules and a final exclusion rule to select
only required models/data/runtime/manifests; otherwise ordinary dataset sync copies
the whole root. Validate the selected files and symlink closure independently.
Filtered pushes run rsync even when the source root mtime is unchanged, since nested
files or filter contents may have changed. Missing/unresolved filters fail before
preparation. No delete or whole-root mirror is implied.

`datasets_copy_links` defaults to `true`, retaining the usual dataset behavior of
following source links. Set it to `false` for a portable bundle whose manifest
requires symlink types and exact relative link text: rsync archive mode then
preserves links on both pull and push. Verify that the selected relative links
are closed within the bundle before using this mode; cluv does not invent that proof.

The existing `local.env` applies only off-cluster: the example resolves the
desktop mirror under `$HOME/.cache/project/datasets/project`, while remote
`$SCRATCH` paths still expand in each target login shell. It does not change
`HOME` or export the controller's `SCRATCH` to targets. An unresolved controller
path fails before git push or transfer. A local-current-cluster submission uses
the controller's actual checkout for shell cwd, Slurm `--chdir` and script lookup,
so an older configured remote checkout is left untouched.

With `manage_uv = false`, selective data is copied before validating `uv` through
the configured target environment. cluv never installs or self-updates site uv.
Configure the staged tool's `PATH`, `UV_PYTHON`, `UV_CACHE_DIR` and offline settings
in the cluster environment; that exact environment is also used for `uv sync`.
Variables expand in the target login shell, not on the controller. The default
`manage_uv = true` retains the ordinary installer and login-node cache-warming
behavior. A staged bundle still needs real per-target relocation/runtime validation
before a GPU race; local fake-endpoint tests do not establish that readiness.

Use `cluv sync --no-fetch-results <cluster> ...` to prepare selected inputs and
runtime without copying old result/checkpoint trees back to the controller.
The default still fetches results. After reviewed preparation and per-target
validation, `cluv submit first --no-sync-datasets --no-fetch-results -- ...`
keeps the ordinary race while avoiding redundant data and result transfers.

[`cluv sync`](../commands.md#cluv-sync) can also replicate datasets to every cluster listed in your Cluv config.

## Configure dataset sync in `pyproject.toml`

Add `data_source` and `datasets_path` under `[tool.cluv]`:

```toml title="pyproject.toml"
[tool.cluv]
# Source cluster and source path (`hostname:/path`).
data_source = "mila:/network/datasets/cifar10.var/cifar10_torchvision"

# Destination path used on each cluster (can use env vars like $SCRATCH).
datasets_path = "$SCRATCH/datasets/cifar10"
```

- `data_source` is where Cluv pulls the dataset from.
- `datasets_path` is where Cluv stores it on each cluster, except the source cluster (see
  [Run sync](#run-sync)).

You can override `datasets_path` per cluster:

```toml title="pyproject.toml"
[tool.cluv.clusters.killarney]
datasets_path = "$HOME/datasets/cifar10"
```

### Using a local path as the data source

If your dataset is already on the machine where you run `cluv sync`, you can point
`data_source` directly at a local path (no `hostname:` prefix):

```toml
[tool.cluv]
# Local path — no hostname prefix.
data_source = "/data/datasets/cifar10"

# Destination path on each cluster.
datasets_path = "$SCRATCH/datasets/cifar10"
```

Cluv will skip the remote-pull step and push the local directory straight to every
configured cluster.


## Login to clusters (including the source)

Before syncing, create reusable SSH connections:

```console
cluv login
```

If your source is `mila:...`, make sure the `mila` connection exists (for example `cluv login mila`).

## Run sync

```console
cluv sync
```

With dataset sync enabled (default), Cluv will:

1. Pull data from `data_source` to the `datasets_path` of the machine where you run Cluv (remote
   source only).
2. Push that dataset to each target cluster at its configured `datasets_path`.

Cluv skips the copies that aren't needed:

- It doesn't push the data to the source cluster. Jobs there read the data at the `data_source`
  path directly: on the source cluster, `cluv.job.get_datasets_path()` returns that path instead of
  `datasets_path`.
- It only pulls the data when a target other than the source cluster needs it. The cluster where you
  run Cluv counts as a target when you name it, when you run `cluv sync` without naming any cluster,
  or when you run `cluv submit first`.

For example, with `data_source = "mila:/network/datasets/imagenet"`:

- `cluv submit mila`, run from your laptop or from narval, copies no data.
- `cluv submit narval`, run on narval, pulls the data to narval.
- `cluv sync narval`, run on mila, pushes the data from `/network/datasets/imagenet` to narval.

## Optional: skip dataset replication for one run

If you need to skip the dataset replication step, specify:
```console
cluv sync --no-sync-datasets
```
