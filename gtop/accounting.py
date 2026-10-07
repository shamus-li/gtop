from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Set

from rich.text import Text

from .constants import JOB_RESOURCE_NAMES
from .models import (
    JobRecord,
    JobUsage,
    NodeAllocation,
    ResourceUsageSplit,
    ServerState,
    UserUsage,
)
from .slurm import parse_nodelist


def _add_shard_gpu_equivalents(
    server: ServerState,
    gpu_usage: ResourceUsageSplit,
    shards_by_partition: Mapping[str, float],
) -> None:
    shards_per_gpu = server.gpu.shards_per_gpu
    if shards_per_gpu <= 0:
        return
    for partition, shards in shards_by_partition.items():
        if shards > 0:
            gpu_usage.add(
                partition,
                min(server.gpu.num, math.ceil(shards / shards_per_gpu)),
            )


def _explicit_shard_usage(server: ServerState, usage: JobUsage) -> float:
    gpu_shards = usage.gpu * server.gpu.shards_per_gpu
    return max(usage.shard - gpu_shards, 0.0)


def process_jobs(
    jobs: Sequence[JobRecord],
    servers: Dict[str, ServerState],
    *,
    stderr_console: Optional[Any] = None,
    store_allocations: bool = True,
) -> None:
    explicit_shards: Dict[str, Dict[str, float]] = {}

    for job in jobs:
        if job.state != "RUNNING":
            continue

        nodes = parse_nodelist(job.nodelist)
        if not nodes:
            if stderr_console is not None:
                stderr_console.print(
                    Text(
                        f"Warning: No nodes found for job {job.job_id}", style="yellow"
                    )
                )
            continue

        matched_nodes = [node for node in nodes if node in servers]

        if not matched_nodes:
            continue

        per_node = {
            resource: getattr(job.usage, resource) / len(nodes)
            for resource in JOB_RESOURCE_NAMES
        }
        for node in matched_nodes:
            shard_amount = (
                per_node["shard"] + per_node["gpu"] * servers[node].gpu.shards_per_gpu
            )
            node_usage = JobUsage(
                cpu=per_node["cpu"],
                gpu=per_node["gpu"],
                mem=per_node["mem"],
                shard=shard_amount,
            )
            if store_allocations:
                servers[node].allocations[job.job_id] = NodeAllocation(
                    job=job,
                    usage=node_usage,
                )
            for resource in JOB_RESOURCE_NAMES:
                servers[node].usage[resource].add(
                    job.partition,
                    getattr(node_usage, resource),
                )
            if per_node["shard"] > 0:
                node_shards = explicit_shards.setdefault(node, {})
                node_shards[job.partition] = (
                    node_shards.get(job.partition, 0.0) + per_node["shard"]
                )

    for node, shards_by_partition in explicit_shards.items():
        _add_shard_gpu_equivalents(
            servers[node], servers[node].usage["gpu"], shards_by_partition
        )


def summarize_users(
    servers: Mapping[str, ServerState],
) -> Dict[str, UserUsage]:
    usage_by_user_partition: Dict[
        tuple[str, str],
        tuple[float, Set[str]],
    ] = {}
    for server in servers.values():
        for allocation in server.allocations.values():
            job = allocation.job
            usage = allocation.usage
            resource_count = usage.gpu + (
                _explicit_shard_usage(server, usage) / server.gpu.shards_per_gpu
                if server.gpu.shards_per_gpu > 0
                else 0.0
            )
            key = (job.user, job.partition)
            current_count, nodes = usage_by_user_partition.setdefault(
                key,
                (0.0, set()),
            )
            nodes.add(server.name)
            usage_by_user_partition[key] = (
                current_count + resource_count,
                nodes,
            )

    summaries: Dict[str, UserUsage] = {}
    for (user, partition), (resource_count, nodes) in usage_by_user_partition.items():
        rounded_count = math.ceil(resource_count - 1e-9)
        if rounded_count <= 0:
            continue
        summary = summaries.setdefault(user, UserUsage(user=user))
        summary.add(partition, rounded_count, nodes)
    return summaries


def project_servers_for_users(
    servers: Sequence[ServerState],
    *,
    target_users: Set[str],
) -> list[ServerState]:
    projected_servers: list[ServerState] = []
    for server in servers:
        projected_allocations: dict[str, NodeAllocation] = {}
        projected_usage = {
            resource: ResourceUsageSplit() for resource in JOB_RESOURCE_NAMES
        }
        explicit_shard_usage_by_partition: dict[str, float] = {}
        for job_id, allocation in server.allocations.items():
            job = allocation.job
            usage = allocation.usage
            if job.user not in target_users:
                continue
            projected_allocations[job_id] = allocation
            projected_usage["cpu"].add(job.partition, usage.cpu)
            projected_usage["mem"].add(job.partition, usage.mem)
            projected_usage["shard"].add(job.partition, usage.shard)
            projected_usage["gpu"].add(job.partition, usage.gpu)
            explicit_shards = _explicit_shard_usage(server, usage)
            if explicit_shards > 0:
                explicit_shard_usage_by_partition[job.partition] = (
                    explicit_shard_usage_by_partition.get(job.partition, 0.0)
                    + explicit_shards
                )
        _add_shard_gpu_equivalents(
            server, projected_usage["gpu"], explicit_shard_usage_by_partition
        )
        projected_servers.append(
            replace(
                server,
                usage=projected_usage,
                allocations=projected_allocations,
            )
        )
    return projected_servers
