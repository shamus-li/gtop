from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Set

from .constants import JOB_RESOURCE_NAMES


@dataclass
class GpuInfo:
    type: str = "null"
    num: int = 0
    shards: int = 0
    used: int = 0
    used_shards: int = 0


@dataclass
class CpuInfo:
    idle: int = 0


@dataclass
class MemoryInfo:
    idle: float = 0.0
    total: float = 0.0


@dataclass
class ResourceUsageSplit:
    priority: float = 0.0
    gpu: float = 0.0
    default: float = 0.0

    def total(self) -> float:
        return self.priority + self.gpu + self.default

    def add(self, partition_type: str, amount: float) -> None:
        setattr(self, partition_type, getattr(self, partition_type) + amount)


@dataclass
class JobUsage:
    cpu: float = 0.0
    gpu: float = 0.0
    mem: float = 0.0
    shard: float = 0.0


@dataclass
class JobAllocation:
    netid: str
    job_id: str
    job_name: str
    state: str
    partition: str
    nodelist: str
    usage_str: str
    time_limit: str
    cpu: float = 0.0
    gpu: float = 0.0
    mem: float = 0.0
    shard: float = 0.0


@dataclass
class JobRecord:
    user: str
    job_id: str
    job_name: str
    state: str
    partition: str
    nodelist: str
    usage_str: str
    time_limit: str
    cpu: float = 0.0
    gpu: float = 0.0
    mem: float = 0.0
    shard: float = 0.0


def _default_usage_splits() -> Dict[str, ResourceUsageSplit]:
    return {resource: ResourceUsageSplit() for resource in JOB_RESOURCE_NAMES}


@dataclass
class ServerState:
    name: str
    features: Set[str]
    gpu: GpuInfo
    cpu: CpuInfo
    mem: MemoryInfo
    usage: Dict[str, ResourceUsageSplit] = field(default_factory=_default_usage_splits)
    users: Dict[str, JobAllocation] = field(default_factory=dict)

    def has_target_users(self, target_users: Set[str]) -> bool:
        return any(job.netid in target_users for job in self.users.values())


@dataclass
class UserSummary:
    nodes: Set[str] = field(default_factory=set)
    usage_by_partition: Dict[str, int] = field(default_factory=dict)
    priority_usage: int = 0
    gpu_usage: int = 0
    default_usage: int = 0

    def total_usage(self) -> int:
        return self.priority_usage + self.gpu_usage + self.default_usage


@dataclass
class TopUserSummary:
    user: str
    nodes: Set[str] = field(default_factory=set)
    priority_usage: int = 0
    gpu_usage: int = 0
    default_usage: int = 0

    def total_usage(self) -> int:
        return self.priority_usage + self.gpu_usage + self.default_usage


@dataclass
class ClusterState:
    servers: Dict[str, ServerState] = field(default_factory=dict)
    jobs: List[JobRecord] = field(default_factory=list)


@dataclass
class ClusterSummary:
    gpu_total: int
    gpu_used: int
    gpu_utilization_pct: float
    resource_label: str
    target_users: Dict[str, UserSummary] = field(default_factory=dict)
    top_users: List[TopUserSummary] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
