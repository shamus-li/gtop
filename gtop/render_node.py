from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .models import JobRecord, NodeAllocation, ServerState
from .partitions import partition_names
from .render import _display_gpu_type
from .runner import Command
from .scheduling import INTERACTIVE_SUFFIX, Partition


def node_state_command(nodes: Sequence[str]) -> Command:
    return ("sinfo", "-h", "-N", "-n", ",".join(nodes), "-o", "%n|%T|%E")


@dataclass(frozen=True)
class NodeStatus:
    state: str
    reason: str


def parse_node_states(output: str) -> dict[str, NodeStatus]:
    states: dict[str, NodeStatus] = {}
    for line in output.splitlines():
        name, _, rest = line.partition("|")
        state, _, reason = rest.partition("|")
        states[name] = NodeStatus(
            state=state, reason="" if reason == "none" else reason
        )
    return states


def _partitions_on(node: str, partitions: Mapping[str, Partition]) -> list[Partition]:
    return sorted(
        (
            partition
            for partition in partitions.values()
            if node in partition.nodes
            and not partition.name.endswith(INTERACTIVE_SUFFIX)
        ),
        key=lambda partition: (-partition.tier, partition.name),
    )


def _sorted_allocations(
    server: ServerState, partitions: Mapping[str, Partition]
) -> list[NodeAllocation]:
    def tier(allocation: NodeAllocation) -> int:
        partition = partitions.get(allocation.job.partition)
        return partition.tier if partition is not None else 0

    return sorted(
        server.allocations.values(),
        key=lambda allocation: (
            -tier(allocation),
            allocation.job.user,
            allocation.job.job_id,
        ),
    )


@dataclass(frozen=True)
class QueuedJobs:
    """Pending jobs that could start on a node.

    Jobs for group-restricted (lab) partitions are listed; shared partitions
    span many nodes, so only their queue length is shown.
    """

    lab_partitions: tuple[str, ...]
    listed: tuple[JobRecord, ...]
    shared_counts: dict[str, int]


def queued_for(
    node: str,
    jobs: Sequence[JobRecord],
    partitions: Mapping[str, Partition],
) -> QueuedJobs:
    on_node = _partitions_on(node, partitions)
    shared = [partition for partition in on_node if "all" in partition.allow_groups]
    top_shared_tier = max((partition.tier for partition in shared), default=0)
    # Lab partitions outrank the shared ones; admin-only queues such as debug don't.
    labs = [
        partition
        for partition in on_node
        if "all" not in partition.allow_groups and partition.tier > top_shared_tier
    ]
    lab_names = {partition.name for partition in labs}
    shared_names = {partition.name for partition in shared}
    listed: list[JobRecord] = []
    shared_counts: dict[str, int] = {}
    for job in jobs:
        if job.state == "RUNNING":
            continue
        requested = {
            name.removesuffix(INTERACTIVE_SUFFIX)
            for name in partition_names(job.partition)
        }
        if requested & lab_names:
            listed.append(job)
            continue
        for name in requested & shared_names:
            shared_counts[name] = shared_counts.get(name, 0) + 1
    listed.sort(key=lambda job: (job.partition, job.user, job.job_id))
    return QueuedJobs(
        lab_partitions=tuple(partition.name for partition in labs),
        listed=tuple(listed),
        shared_counts={
            partition.name: shared_counts[partition.name]
            for partition in shared
            if partition.name in shared_counts
        },
    )


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def render_node(
    server: ServerState,
    status: NodeStatus,
    partitions: Mapping[str, Partition],
    queued: QueuedJobs,
) -> Group:
    used_gpus = server.gpu.occupied()
    used_cpus = server.cpu.total - server.cpu.idle - server.cpu.other
    total_mem = round(server.mem.total / 1024)
    used_mem = round((server.mem.total - server.mem.idle) / 1024)
    gpu_type = _display_gpu_type(server) if server.gpu.num else "no GPUs"
    lines: list[Any] = [
        Text.assemble(
            (server.name, "bold cyan"),
            f"  {server.gpu.num}× {gpu_type}  " if server.gpu.num else f"  {gpu_type}  ",
            (status.state, "green" if server.accepts_jobs else "red"),
        ),
        Text(
            f"GPUs {used_gpus}/{server.gpu.num} used · "
            f"CPUs {used_cpus}/{server.cpu.total} · RAM {used_mem}/{total_mem}G"
        ),
        Text(
            "Partitions: "
            + ", ".join(
                f"{partition.name} ({partition.tier})"
                for partition in _partitions_on(server.name, partitions)
            ),
            style="dim",
        ),
    ]
    if status.reason:
        lines.append(Text(f"Reason: {status.reason}", style="red"))
    allocations = _sorted_allocations(server, partitions)
    if allocations:
        lines.append(_running_table(allocations))
    else:
        lines.append(Text("No running jobs", style="dim"))
    lines.extend(_queued_lines(queued))
    return Group(*lines)


def _queued_lines(queued: QueuedJobs) -> list[Any]:
    lines: list[Any] = [Text("")]
    if queued.lab_partitions:
        count = len(queued.listed)
        lines.append(
            Text(
                f"Queued in {', '.join(queued.lab_partitions)}: "
                f"{count} {'job' if count == 1 else 'jobs'}",
                style="bold",
            )
        )
    if queued.listed:
        lines.append(_queued_table(queued.listed))
    if queued.shared_counts:
        lines.append(
            Text.assemble(
                "Queued in ",
                ", ".join(f"{name}: {n}" for name, n in queued.shared_counts.items()),
                (" (shared, may run on other nodes)", "dim"),
            )
        )
    return lines


def _queued_table(listed: Sequence[JobRecord]) -> Table:
    # Array tasks and repeated submissions differ only by job ID; show them once.
    groups: dict[tuple[str, str, float, float, float, str], list[str]] = {}
    for job in listed:
        key = (
            job.user,
            job.partition,
            job.usage.gpu,
            job.usage.cpu,
            round(job.usage.mem),
            job.reason,
        )
        groups.setdefault(key, []).append(job.job_id)
    show_reason = any(job.reason for job in listed)

    table = Table(box=None, pad_edge=False, padding=(0, 1), header_style="bold white")
    table.add_column("User", no_wrap=True, style="cyan")
    table.add_column("Jobs", no_wrap=True, style="dim")
    table.add_column("Partition", no_wrap=True)
    for label in ("GPU", "CPU", "RAM"):
        table.add_column(label, justify="right", no_wrap=True)
    if show_reason:
        table.add_column("Reason", no_wrap=True, style="dim")
    for (user, partition, gpu, cpu, mem, reason), job_ids in groups.items():
        row = [
            user,
            job_ids[0] if len(job_ids) == 1 else f"{len(job_ids)} jobs",
            partition,
            _number(gpu) if gpu else "-",
            _number(cpu) if cpu else "-",
            f"{_number(mem)}G" if mem else "-",
        ]
        if show_reason:
            row.append(reason or "-")
        table.add_row(*row)
    return table


def _running_table(allocations: Sequence[NodeAllocation]) -> Table:
    table = Table(box=None, pad_edge=False, padding=(0, 1), header_style="bold white")
    table.add_column("User", no_wrap=True, style="cyan")
    table.add_column("Job", no_wrap=True, style="dim")
    table.add_column("Partition", no_wrap=True)
    for label in ("GPU", "CPU", "RAM"):
        table.add_column(label, justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    for allocation in allocations:
        job = allocation.job
        usage = allocation.usage
        table.add_row(
            job.user,
            job.job_id,
            job.partition,
            _number(usage.gpu) if usage.gpu else "-",
            _number(usage.cpu),
            f"{_number(round(usage.mem))}G",
            job.elapsed or "-",
        )
    return table


def node_json(
    server: ServerState,
    status: NodeStatus,
    partitions: Mapping[str, Partition],
    queued: QueuedJobs,
) -> dict[str, Any]:
    return {
        "name": server.name,
        "gpu_type": _display_gpu_type(server) if server.gpu.num else None,
        "state": status.state,
        "reason": status.reason,
        "gpus": {"total": server.gpu.num, "used": server.gpu.occupied()},
        "cpus": {
            "total": server.cpu.total,
            "used": server.cpu.total - server.cpu.idle - server.cpu.other,
        },
        "memory_gib": {
            "total": round(server.mem.total / 1024),
            "used": round((server.mem.total - server.mem.idle) / 1024),
        },
        "partitions": {
            partition.name: partition.tier
            for partition in _partitions_on(server.name, partitions)
        },
        "jobs": [
            {
                "job_id": allocation.job.job_id,
                "user": allocation.job.user,
                "partition": allocation.job.partition,
                "gpus": allocation.usage.gpu,
                "cpus": allocation.usage.cpu,
                "memory_gib": round(allocation.usage.mem, 1),
                "elapsed": allocation.job.elapsed,
            }
            for allocation in _sorted_allocations(server, partitions)
        ],
        "queued": [
            {
                "job_id": job.job_id,
                "user": job.user,
                "partition": job.partition,
                "gpus": job.usage.gpu,
                "cpus": job.usage.cpu,
                "memory_gib": round(job.usage.mem, 1),
                "reason": job.reason,
            }
            for job in queued.listed
        ],
        "shared_queue_counts": queued.shared_counts,
    }
