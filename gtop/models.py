from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from .constants import JOB_RESOURCE_NAMES


@dataclass
class GpuInfo:
    type: str = "null"
    num: int = 0
    shards: int = 0
    used: int = 0
    used_shards: int = 0
    shard_gpus_used: int = 0

    @property
    def shards_per_gpu(self) -> float:
        if self.shards <= 0 or self.num <= 0:
            return 0.0
        return self.shards / self.num

    def occupied(self) -> int:
        shards_per_gpu = self.shards_per_gpu
        shard_gpus = self.shard_gpus_used
        if shard_gpus <= 0 and self.used_shards > 0 and shards_per_gpu > 0:
            shard_gpus = math.ceil(self.used_shards / shards_per_gpu)
        return min(self.num, self.used + shard_gpus)


@dataclass
class CpuInfo:
    idle: int = 0
    total: int = 0
    other: int = 0


@dataclass
class MemoryInfo:
    idle: float = 0.0
    total: float = 0.0


@dataclass
class ResourceUsageSplit:
    partitions: Dict[str, float] = field(default_factory=dict)

    def add(self, partition_name: str, amount: float) -> None:
        self.partitions[partition_name] = (
            self.partitions.get(partition_name, 0.0) + amount
        )


@dataclass(frozen=True)
class JobUsage:
    cpu: float = 0.0
    gpu: float = 0.0
    mem: float = 0.0
    shard: float = 0.0


@dataclass(frozen=True)
class JobRecord:
    user: str
    job_id: str
    job_name: str
    state: str
    partition: str
    nodelist: str
    usage: JobUsage
    time_limit: str
    elapsed: str = ""
    reason: str = ""
    constraints: frozenset[str] = frozenset()


@dataclass(frozen=True)
class NodeAllocation:
    job: JobRecord
    usage: JobUsage


def _default_usage_splits() -> Dict[str, ResourceUsageSplit]:
    return {resource: ResourceUsageSplit() for resource in JOB_RESOURCE_NAMES}


@dataclass
class ServerState:
    name: str
    features: Set[str]
    gpu: GpuInfo
    cpu: CpuInfo
    mem: MemoryInfo
    # sinfo's long state, e.g. "mixed", "drained*", "mixed-" (planned).
    state: str
    reason: str
    usage: Dict[str, ResourceUsageSplit] = field(default_factory=_default_usage_splits)
    allocations: Dict[str, NodeAllocation] = field(default_factory=dict)

    @property
    def accepts_jobs(self) -> bool:
        """sinfo reports CPUs on down or draining nodes as "other", except the
        CPUs still allocated on a draining node, so the state is checked too."""
        return self.cpu.other == 0 and not any(
            word in self.state for word in ("drain", "down", "fail")
        )

    @property
    def planned(self) -> bool:
        """Idle resources held for a pending job; sinfo marks the state with "-"."""
        return self.state.endswith("-")

    def has_target_users(self, target_users: Set[str]) -> bool:
        return any(
            allocation.job.user in target_users
            for allocation in self.allocations.values()
        )


@dataclass
class UserUsage:
    user: str
    nodes: Set[str] = field(default_factory=set)
    usage_by_partition: Dict[str, int] = field(default_factory=dict)

    def total_usage(self) -> int:
        return sum(self.usage_by_partition.values())

    def add(self, partition: str, amount: int, nodes: Iterable[str]) -> None:
        self.nodes.update(nodes)
        self.usage_by_partition[partition] = (
            self.usage_by_partition.get(partition, 0) + amount
        )


@dataclass
class ClusterState:
    servers: Dict[str, ServerState] = field(default_factory=dict)
    jobs: List[JobRecord] = field(default_factory=list)
