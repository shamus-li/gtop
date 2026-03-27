# gtop

`gtop` shows live SLURM GPU usage on Cornell-style clusters, with a fast GPU-type summary by default and deeper job or node views when you need them.

## Install

```bash
uv tool install git+ssh://git@github.com/shamus-li/gtop.git
```

If `gtop` is not found after install, add uv's tool bin directory to your shell `PATH`:

```bash
uv tool update-shell
exec "$SHELL" -l
```

The installed executable is placed in `$(uv tool dir --bin)` (typically `~/.local/bin`).

Upgrade or remove:

```bash
uv tool upgrade gtop
uv tool uninstall gtop
```

## Default View

```bash
gtop
```

Plain `gtop` shows one row per GPU type. It is the high-level view: how many GPUs are free, how usage is split across `priority`, `gpu`, and `default` partitions, and how many nodes are in each GPU family.

Legend from `gtop --help`:

```text
magenta = priority   cyan = gpu   blue = default   dim = free
counts are free/total, then priority/gpu/default
```

## Common Commands

```bash
gtop -v
```

Show the node-by-node breakdown instead of the GPU-type summary.

```bash
gtop --me
gtop -u wl757
```

Filter the view to one or more users. This becomes a usage view rather than a free-capacity view.

```bash
gtop --me -v
```

Show the filtered node breakdown and the jobs you are running on each node.

```bash
gtop -U
gtop -U 10
```

Show only the largest cluster users. The default is the top 25 users.

```bash
gtop --jobs
gtop --jobs --me
gtop --jobs --partition monakhova
```

Show the active jobs view. Jobs are grouped by node, with parsed `GPU`, `CPU`, and `MEM` columns instead of raw `AllocTRES`.

```bash
gtop --constraint gpu-high
gtop --sort name
```

Filter nodes by SLURM constraint expression or change the sort order.

```bash
gtop -s
```

Switch the display from whole GPUs to shard counts on sharded nodes.

```bash
gtop --json
```

Emit JSON instead of the Rich terminal view.

## Jobs View

`gtop --jobs` is meant to replace the common `sacct | grep ... | sort` workflow for active jobs.

- It defaults to `RUNNING,PENDING,REQUEUED`.
- `--partition` matches both the SLURM partition name and the node name, so `gtop --jobs --partition monakhova` also catches jobs running on `monakhova-*` nodes from other partitions.
- Job rows show parsed resources and time limits, while the node header shows used GPUs and the GPU type for that node when available.

## Help And Debugging

```bash
gtop --help
```

Low-level flags such as `--timeout`, `--sinfo-command`, `--sacct-command`, `--debug`, and `--no-parallel` are grouped under `Debug options` in the help output.
