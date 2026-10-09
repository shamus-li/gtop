from __future__ import annotations

import grp
import math
import os
from dataclasses import dataclass
from typing import Mapping

from .models import ServerState
from .runner import Command
from .slurm import parse_nodelist

PARTITION_COMMAND: Command = ("scontrol", "show", "partition", "-a", "-o")
INTERACTIVE_SUFFIX = "-interactive"
# A node's GPUs count as available when one job could take them all plus this
# much CPU and RAM on the same node.
MIN_JOB_CPUS = 4
MIN_JOB_MEM_GB = 16


@dataclass(frozen=True)
class Partition:
    name: str
    tier: int
    nodes: frozenset[str]
    allow_groups: frozenset[str]


@dataclass(frozen=True)
class Resources:
    gpus: int = 0
    cpus: int = 0
    mem_gb: int = 0

    @property
    def usable_gpus(self) -> int:
        if self.cpus < MIN_JOB_CPUS or self.mem_gb < MIN_JOB_MEM_GB:
            return 0
        return self.gpus


def parse_partitions(output: str) -> dict[str, Partition]:
    partitions: dict[str, Partition] = {}
    for line in output.splitlines():
        fields = dict(
            field.split("=", 1) for field in line.split() if "=" in field
        )
        name = fields.get("PartitionName")
        if not name:
            continue
        nodes = fields.get("Nodes", "(null)")
        partitions[name] = Partition(
            name=name,
            tier=int(fields.get("PriorityTier", "0")),
            nodes=frozenset(parse_nodelist(nodes)) if nodes != "(null)" else frozenset(),
            allow_groups=frozenset(
                group.lower() for group in fields.get("AllowGroups", "ALL").split(",")
            ),
        )
    return partitions


def user_groups(user: str) -> frozenset[str]:
    return frozenset(
        grp.getgrgid(gid).gr_name.lower()
        for gid in os.getgrouplist(user, os.getgid())
    )


def submittable_partitions(
    partitions: Mapping[str, Partition],
    groups: frozenset[str],
) -> list[Partition]:
    """Batch partitions the user may submit to, highest priority tier first."""
    return sorted(
        (
            partition
            for partition in partitions.values()
            if not partition.name.endswith(INTERACTIVE_SUFFIX)
            and ("all" in partition.allow_groups or partition.allow_groups & groups)
        ),
        key=lambda partition: (-partition.tier, partition.name),
    )


def obtainable(
    server: ServerState,
    tier: int,
    partitions: Mapping[str, Partition],
) -> tuple[Resources, Resources]:
    """Resources a job at this priority tier can get now, and by preempting.

    With partition_prio preemption, a job requeues running jobs from
    lower-tier partitions on the same node. A planned node's idle resources
    are held for a pending job, so only preemption can free anything there.
    """
    if not server.accepts_jobs:
        return Resources(), Resources()
    free_gpus = 0 if server.planned else max(server.gpu.num - server.gpu.occupied(), 0)
    now = (
        Resources()
        if server.planned
        else Resources(
            gpus=free_gpus,
            cpus=server.cpu.idle,
            mem_gb=int(server.mem.idle / 1024),
        )
    )
    gpus = cpus = mem_gb = shards = 0.0
    for allocation in server.allocations.values():
        partition = partitions.get(allocation.job.partition)
        if partition is None or partition.tier >= tier:
            continue
        usage = allocation.usage
        gpus += usage.gpu
        cpus += usage.cpu
        mem_gb += usage.mem
        shards += max(usage.shard - usage.gpu * server.gpu.shards_per_gpu, 0.0)
    if shards and server.gpu.shards_per_gpu:
        gpus += math.ceil(shards / server.gpu.shards_per_gpu)
    preempting = Resources(
        gpus=min(free_gpus + round(gpus), server.gpu.num),
        cpus=min(now.cpus + round(cpus), server.cpu.total),
        mem_gb=int(min(now.mem_gb + mem_gb, server.mem.total / 1024)),
    )
    return now, preempting
