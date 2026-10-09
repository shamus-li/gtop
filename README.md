# gtop

`gtop` shows live SLURM GPU usage with node, user, job, and partition detail.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A login node where `sinfo` and `sacct` can read the cluster state

gtop is built for the Cornell Unicorn cluster, where partitions have priority
tiers and nodes carry `gpu-high`, `gpu-mid` or `gpu-low` features.

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

For each partition you can submit to: GPUs free now, and how many more you
would get by preempting jobs in lower-priority partitions. A node's GPUs count
as free when one job could take them plus 4 CPUs and 16G RAM on that node;
idle GPUs on nodes without that much CPU or RAM are listed as short. Use
`-t high`, `-t mid` or `-t low` to show only one GPU tier, and `-C` to require
node features: `gtop -C 'nvlink,ampere|ada|hopper|blackwell'` shows modern GPUs
with NVLink. `gtop nodes` takes the same filters.

```bash
gtop nodes
gtop nodes nikola-compute-14
```

Free GPU, CPU and memory on every node. Name one or more nodes to see their
state (with any drain reason), the partitions that include them, the jobs
running there and the jobs queued for them. Jobs queued in the node's lab
partitions are listed; shared queues such as `gpu` are only counted.

```bash
gtop users
gtop users --week
gtop users --month
gtop users --year
gtop users -n 50
```

Who is using GPUs now, or GPU-hours per user and lab account over the last 7,
30 or 365 days, from `sreport`. Results are cached. The week and month caches
refresh in the background (hourly and every 6 hours); the year cache only
updates when you add `--refresh`, which takes about 80 seconds. The list shows
the top 25 users; `-n N` shows N.

```bash
gtop jobs
gtop jobs -m
```

Running, pending and requeued jobs.

Every command takes `-p PARTITION` and `--json`; `nodes`, `users` and `jobs`
also take `-m` (your usage) or `-u USER`. Give several partitions, users or
features as a comma-separated list: `gtop jobs -u alice,bob -p gpu,default`.

## Help

```bash
gtop --help
```

The help includes every option and the display legend.
