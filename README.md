# gtop

`gtop` shows live SLURM GPU usage with node and partition detail. It preserves the Cornell-style priority/gpu/default view when those partitions exist, and otherwise adapts to the cluster's actual partition names.

## Install

```bash
uv tool install git+ssh://git@github.com/shamus-li/gtop.git
```

For a local checkout:

```bash
uv tool install --force --editable .
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

Default output is a GPU-family summary: free GPUs, partition split, and node count.

```bash
gtop -v
```

Show the node-by-node view.

```bash
gtop --me
gtop -u wl757
gtop --me -v
gtop --partition cornell
gtop --mine
```

Filter to one or more users, scope the view to specific partitions, or auto-detect your accessible partitions. Filtered views show usage rather than free capacity. `-v` also includes per-node jobs.

```bash
gtop -U
gtop -U 10
```

Show only the top users. Default is 25.

```bash
gtop --jobs
gtop --jobs --me
gtop --jobs -p monakhova
gtop --jobs --mine
```

Show active jobs grouped by node with parsed `GPU`, `CPU`, `MEM`, and time columns.

```bash
gtop --constraint gpu-high
gtop --sort name
gtop -s
gtop --json
```

Use `--constraint` everywhere, switch to shard counts with `-s`, or emit JSON with `--json`.

## Help

```bash
gtop --help
```

Legend:

```text
orchid = priority   teal = gpu   cornflower = default   dim = free
counts are free/total, then priority/gpu/default
```
