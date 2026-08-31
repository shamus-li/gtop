# gtop

`gtop` shows live SLURM GPU usage with node, user, job, and partition detail.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A login node where `sinfo` and `sacct` can read the cluster state

The parser is tested against captured Cornell Unicorn and Empire AI Slurm output.

## Install

```bash
uv tool install git+https://github.com/shamus-li/gtop.git
```

If `gtop` is not on `PATH`, let `uv` add its tool bin directory and restart your shell:

```bash
uv tool update-shell
exec "$SHELL" -l
```

The binary is installed in `$(uv tool dir --bin)` (usually `~/.local/bin`).

## Usage

```bash
gtop
```

The default view groups free capacity by GPU family. `--nodes` expands those
groups into nodes, `--jobs` shows active jobs, and `-U` shows the top 25 users.

```bash
gtop -v
gtop --nodes
```

Show the node-by-node view.

```bash
gtop -m
gtop -u wl757
gtop -u wl757 abc123
gtop -m -v
```

Filter to your own usage or one or more users. Use `-j -m` to see your jobs.

```bash
gtop -U
```

Show the top 25 users.

```bash
gtop -j
gtop -j -m
gtop -j -p monakhova
```

Jobs always include running, pending, and requeued records.

```bash
gtop -p monakhova gpu
gtop -C gpu gpu-high
gtop -s
gtop --json
```

`-p` selects nodes in one or more partitions and includes all usage on those
nodes, even from jobs submitted through other partitions. For example,
`gtop -j -p monakhova` also shows jobs on those nodes from
`monakhova-interactive` or `gpu`. Pending jobs without assigned nodes are
filtered by their requested partitions. Add `-m` to limit usage to your jobs.
`-C` requires every listed node feature.
`-s` switches capacity and top-user views to shard counts. `--json` emits a
compact, view-specific schema.

## Help

```bash
gtop --help
```

The help includes every option and the display legend.
