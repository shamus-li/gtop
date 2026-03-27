from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Set

from rich.text import Text

from .constants import JOB_RESOURCE_NAMES, PARTITIONS
from .models import (
    ClusterState,
    ClusterSummary,
    CpuInfo,
    GpuInfo,
    JobAllocation,
    JobRecord,
    MemoryInfo,
    ResourceUsageSplit,
    ServerState,
    TopUserSummary,
    UserSummary,
)
from .resources import parse_usage
from .slurm import parse_jobs, parse_nodelist


def partition_bucket(name: str) -> str:
    lower_name = name.lower()
    if "default" in lower_name:
        return "default"
    if "gpu" in lower_name:
        return "gpu"
    return "priority"


def is_priority(name: str) -> bool:
    return partition_bucket(name) == "priority"


def shard_equivalent(server: ServerState, gpu_count: float) -> float:
    if gpu_count <= 0 or server.gpu.shards <= 0 or server.gpu.num <= 0:
        return 0.0
    return gpu_count * (server.gpu.shards / server.gpu.num)


def gpu_occupancy_equivalent(server: ServerState, shard_count: float) -> float:
    if shard_count <= 0 or server.gpu.shards <= 0 or server.gpu.num <= 0:
        return 0.0
    shards_per_gpu = server.gpu.shards / server.gpu.num
    if shards_per_gpu <= 0:
        return 0.0
    return float(min(server.gpu.num, math.ceil(shard_count / shards_per_gpu)))


def process_jobs(
    job_data: str | Sequence[JobRecord],
    servers: Dict[str, ServerState],
    *,
    debug_enabled: bool = False,
    stderr_console: Optional[Any] = None,
    store_users: bool = True,
) -> list[JobRecord]:
    jobs = parse_jobs(job_data) if isinstance(job_data, str) else list(job_data)
    processed_jobs = 0

    for job in jobs:
        if job.state != "RUNNING":
            continue

        usage = parse_usage(job.usage_str)
        nodes = parse_nodelist(job.nodelist)
        if not nodes:
            if stderr_console is not None:
                stderr_console.print(
                    Text(f"Warning: No nodes found for job {job.job_id}", style="yellow")
                )
            continue

        partition_type = partition_bucket(job.partition)
        matched_nodes = [node for node in nodes if node in servers]

        if debug_enabled and stderr_console is not None:
            stderr_console.print(
                Text(
                    f"Processing job {job.job_id}: {job.user}@{job.partition}, "
                    f"nodes: {len(nodes)}, usage: {usage}",
                    style="dim",
                )
            )

        if not matched_nodes:
            if debug_enabled and stderr_console is not None:
                stderr_console.print(
                    Text(
                        f"Skipping job {job.job_id}: nodes {nodes} not in server list",
                        style="dim",
                    )
                )
            continue

        per_node = {
            resource: getattr(usage, resource) / len(matched_nodes)
            for resource in JOB_RESOURCE_NAMES
        }
        for node in matched_nodes:
            shard_amount = per_node["shard"]
            if shard_amount <= 0 and per_node["gpu"] > 0:
                shard_amount = shard_equivalent(servers[node], per_node["gpu"])
            if store_users:
                servers[node].users[job.job_id] = JobAllocation(
                    netid=job.user,
                    job_id=job.job_id,
                    job_name=job.job_name,
                    state=job.state,
                    partition=job.partition,
                    nodelist=job.nodelist,
                    usage_str=job.usage_str,
                    time_limit=job.time_limit,
                    cpu=per_node["cpu"],
                    gpu=per_node["gpu"],
                    mem=per_node["mem"],
                    shard=shard_amount,
                )
            for resource in JOB_RESOURCE_NAMES:
                amount = shard_amount if resource == "shard" else per_node[resource]
                servers[node].usage[resource].add(partition_type, amount)
        processed_jobs += 1

    if debug_enabled and stderr_console is not None:
        stderr_console.print(Text(f"Processed {processed_jobs} job allocations", style="dim"))

    if (
        not jobs
        and isinstance(job_data, str)
        and job_data.strip()
        and stderr_console is not None
    ):
        stderr_console.print(Text("Warning: No jobs were parsed from sacct output", style="yellow"))
    return jobs


def build_top_users_summary(
    jobs: Sequence[JobRecord],
    *,
    target_users: Optional[Set[str]] = None,
    show_shards: bool = False,
    top_users_limit: int = 25,
) -> ClusterSummary:
    all_user_summaries: Dict[str, UserSummary] = {}

    for job in jobs:
        if job.state != "RUNNING":
            continue
        if target_users and job.user not in target_users:
            continue

        usage = parse_usage(job.usage_str)
        resource_count = int(usage.shard if show_shards else usage.gpu)
        if resource_count <= 0:
            continue

        user_summary = all_user_summaries.setdefault(job.user, UserSummary())
        user_summary.nodes.update(parse_nodelist(job.nodelist))
        current = user_summary.usage_by_partition.get(job.partition, 0)
        user_summary.usage_by_partition[job.partition] = current + resource_count
        bucket = partition_bucket(job.partition)
        if bucket == "priority":
            user_summary.priority_usage += resource_count
        elif bucket == "gpu":
            user_summary.gpu_usage += resource_count
        else:
            user_summary.default_usage += resource_count

    target_user_summaries = {
        user: all_user_summaries.get(user, UserSummary())
        for user in (target_users or set())
    }
    users_for_ranking = (
        target_user_summaries.items() if target_users else all_user_summaries.items()
    )
    top_users = sorted(
        users_for_ranking,
        key=lambda item: (
            -item[1].total_usage(),
            -item[1].priority_usage,
            -item[1].gpu_usage,
            item[0],
        ),
    )[:top_users_limit]

    return ClusterSummary(
        gpu_total=0,
        gpu_used=0,
        gpu_utilization_pct=0.0,
        resource_label="shard" if show_shards else "GPU",
        target_users=target_user_summaries,
        top_users=[
            TopUserSummary(
                user=user,
                nodes=set(stats.nodes),
                priority_usage=stats.priority_usage,
                gpu_usage=stats.gpu_usage,
                default_usage=stats.default_usage,
            )
            for user, stats in top_users
        ],
    )


def build_cluster_summary(
    state: ClusterState,
    *,
    target_users: Optional[Set[str]] = None,
    show_shards: bool = False,
    top_users_limit: int = 5,
) -> ClusterSummary:
    if show_shards:
        total_gpus = sum(server.gpu.shards for server in state.servers.values())
        used_gpus = sum(
            int(round(min(server.usage["shard"].total(), float(server.gpu.shards))))
            for server in state.servers.values()
            if server.gpu.shards > 0
        )
    else:
        total_gpus = sum(server.gpu.num for server in state.servers.values())
        used_gpus = sum(
            min(server.gpu.used, server.gpu.num) for server in state.servers.values()
        )
    gpu_pct = (used_gpus / total_gpus) * 100 if total_gpus > 0 else 0.0
    all_user_summaries: Dict[str, UserSummary] = {}

    for server in state.servers.values():
        usage_by_user_partition: Dict[tuple[str, str], Dict[str, float]] = {}
        for job in server.users.values():
            key = (job.netid, job.partition)
            totals = usage_by_user_partition.setdefault(
                key,
                {"gpu": 0.0, "shard": 0.0, "explicit_shard": 0.0},
            )
            totals["gpu"] += job.gpu
            totals["shard"] += job.shard
            if job.gpu <= 0 and job.shard > 0:
                totals["explicit_shard"] += job.shard

        for (user, partition), totals in usage_by_user_partition.items():
            resource_count = int(
                totals["shard"]
                if show_shards and totals["shard"] > 0
                else totals["gpu"] + gpu_occupancy_equivalent(server, totals["explicit_shard"])
            )
            if resource_count <= 0:
                continue
            user_summary = all_user_summaries.setdefault(user, UserSummary())
            user_summary.nodes.add(server.name)
            current = user_summary.usage_by_partition.get(partition, 0)
            user_summary.usage_by_partition[partition] = current + resource_count
            bucket = partition_bucket(partition)
            if bucket == "priority":
                user_summary.priority_usage += resource_count
            elif bucket == "gpu":
                user_summary.gpu_usage += resource_count
            else:
                user_summary.default_usage += resource_count

    target_user_summaries = {
        user: all_user_summaries.get(user, UserSummary())
        for user in (target_users or set())
    }
    users_for_ranking = (
        target_user_summaries.items() if target_users else all_user_summaries.items()
    )
    top_users = sorted(
        users_for_ranking,
        key=lambda item: (
            -item[1].total_usage(),
            -item[1].priority_usage,
            -item[1].gpu_usage,
            item[0],
        ),
    )[:top_users_limit]

    return ClusterSummary(
        gpu_total=total_gpus,
        gpu_used=used_gpus,
        gpu_utilization_pct=gpu_pct,
        resource_label="shard" if show_shards else "GPU",
        target_users=target_user_summaries,
        top_users=[
            TopUserSummary(
                user=user,
                nodes=set(stats.nodes),
                priority_usage=stats.priority_usage,
                gpu_usage=stats.gpu_usage,
                default_usage=stats.default_usage,
            )
            for user, stats in top_users
        ],
    )


def project_servers_for_users(
    servers: Sequence[ServerState],
    *,
    target_users: Set[str],
) -> list[ServerState]:
    projected_servers: list[ServerState] = []
    for server in servers:
        projected_users: dict[str, JobAllocation] = {}
        projected_usage = {
            resource: ResourceUsageSplit() for resource in JOB_RESOURCE_NAMES
        }
        gpu_usage_by_partition = {partition: 0.0 for partition in PARTITIONS}
        explicit_shard_usage_by_partition = {partition: 0.0 for partition in PARTITIONS}
        shard_usage_by_partition = {partition: 0.0 for partition in PARTITIONS}
        for job_id, job in server.users.items():
            if job.netid not in target_users:
                continue
            projected_users[job_id] = job
            partition_type = partition_bucket(job.partition)
            projected_usage["cpu"].add(partition_type, job.cpu)
            projected_usage["mem"].add(partition_type, job.mem)
            projected_usage["shard"].add(partition_type, job.shard)
            gpu_usage_by_partition[partition_type] += job.gpu
            shard_usage_by_partition[partition_type] += job.shard
            if job.gpu <= 0 and job.shard > 0:
                explicit_shard_usage_by_partition[partition_type] += job.shard
        for partition_type in PARTITIONS:
            projected_usage["gpu"].add(
                partition_type,
                gpu_usage_by_partition[partition_type]
                + gpu_occupancy_equivalent(
                    server,
                    explicit_shard_usage_by_partition[partition_type],
                ),
            )
        projected_servers.append(
            ServerState(
                name=server.name,
                features=set(server.features),
                gpu=GpuInfo(
                    type=server.gpu.type,
                    num=server.gpu.num,
                    shards=server.gpu.shards,
                    used=server.gpu.used,
                    used_shards=server.gpu.used_shards,
                ),
                cpu=CpuInfo(idle=server.cpu.idle),
                mem=MemoryInfo(idle=server.mem.idle, total=server.mem.total),
                usage=projected_usage,
                users=projected_users,
            )
        )
    return projected_servers
