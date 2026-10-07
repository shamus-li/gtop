from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .models import ServerState
from .render import _display_gpu_type
from .render_cluster import _gpu_capability_rank
from .scheduling import Partition, obtainable


@dataclass(frozen=True)
class GpuTypeRow:
    gpu_type: str
    free: int
    preemptible: int
    short: int


@dataclass(frozen=True)
class PartitionAvailability:
    partition: Partition
    rows: tuple[GpuTypeRow, ...]


def build_availability(
    servers: Mapping[str, ServerState],
    partitions: Mapping[str, Partition],
    shown: Sequence[Partition],
) -> list[PartitionAvailability]:
    results = []
    for partition in shown:
        nodes = partition.nodes & servers.keys()
        if any(
            other.tier > partition.tier and other.nodes & servers.keys() == nodes
            for other in shown
        ):
            # Same GPUs as a higher-priority partition, so nothing to add.
            continue
        totals: dict[str, list[int]] = {}
        for node in nodes:
            server = servers[node]
            now, preempting = obtainable(server, partition.tier, partitions)
            free = now.usable_gpus
            reachable = preempting.usable_gpus
            counts = totals.setdefault(_display_gpu_type(server), [0, 0, 0])
            counts[0] += free
            counts[1] += max(reachable - free, 0)
            counts[2] += now.gpus - free
        rows = sorted(
            (
                GpuTypeRow(
                    gpu_type=gpu_type, free=free, preemptible=preemptible, short=short
                )
                for gpu_type, (free, preemptible, short) in totals.items()
                if free or preemptible or short
            ),
            # Strongest GPU types first.
            key=lambda row: tuple(
                -rank if isinstance(rank, int) else rank
                for rank in _gpu_capability_rank(row.gpu_type)
            ),
        )
        results.append(
            PartitionAvailability(
                partition=partition,
                rows=tuple(rows),
            )
        )
    return results


def render_availability(
    results: Sequence[PartitionAvailability],
    *,
    tier: str,
) -> Group:
    table = Table(box=None, pad_edge=False, padding=(0, 1), header_style="bold white")
    table.add_column("", no_wrap=True)
    table.add_column("Free", justify="right", no_wrap=True)
    table.add_column("+Preempt", justify="right", no_wrap=True)
    table.add_column("Short", justify="right", no_wrap=True)
    for result in results:
        table.add_row(Text(result.partition.name, style="bold cyan"), "", "", "")
        if not result.rows:
            table.add_row(Text("  nothing free", style="dim"), "", "", "")
        for row in result.rows:
            table.add_row(
                f"  {row.gpu_type}",
                Text(str(row.free), style="bold green" if row.free else "dim"),
                Text(f"+{row.preemptible}", style="yellow")
                if row.preemptible
                else Text("-", style="dim"),
                Text(str(row.short), style="red") if row.short else Text("-", style="dim"),
            )
    if tier == "all":
        return Group(table)
    note = f"gpu-{tier} nodes only"
    return Group(table, Text(note, style="dim"))


def availability_json(results: Sequence[PartitionAvailability]) -> dict[str, Any]:
    return {
        "view": "available",
        "partitions": [
            {
                "partition": result.partition.name,
                "gpu_types": [
                    {
                        "type": row.gpu_type,
                        "free": row.free,
                        "preemptible": row.preemptible,
                        "short_cpu_or_ram": row.short,
                    }
                    for row in result.rows
                ],
            }
            for result in results
        ],
    }
