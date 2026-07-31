from __future__ import annotations

from typing import Any, Sequence

from .models import JobRecord, ServerState, UserUsage
from .render import _display_gpu_type


def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else round(value, 2)


def _capacity(
    *,
    total: float,
    used: float,
    unit: str,
) -> dict[str, int | float | str]:
    return {
        "total": _number(total),
        "used": _number(used),
        "free": _number(max(total - used, 0)),
        "unit": unit,
    }


def _partitions(values: dict[str, float]) -> dict[str, int | float]:
    return {
        partition: _number(amount)
        for partition, amount in sorted(values.items())
        if amount > 0
    }


def summary_json(
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
) -> dict[str, Any]:
    unit = "shard" if show_shards else "GPU"
    grouped: dict[str, list[ServerState]] = {}
    for server in servers:
        grouped.setdefault(_display_gpu_type(server), []).append(server)

    gpu_types = []
    for gpu_type, grouped_servers in sorted(grouped.items()):
        total = sum(server.gpu.capacity(show_shards) for server in grouped_servers)
        used = sum(server.gpu.occupied(show_shards) for server in grouped_servers)
        partition_totals: dict[str, float] = {}
        for server in grouped_servers:
            usage = (
                server.usage["shard"]
                if show_shards and server.gpu.shards > 0
                else server.usage["gpu"]
            )
            for partition, amount in usage.partitions.items():
                partition_totals[partition] = (
                    partition_totals.get(partition, 0.0) + amount
                )

        gpu_types.append(
            {
                "type": gpu_type,
                "capacity": _capacity(total=total, used=used, unit=unit),
                "usage": {"partitions": _partitions(partition_totals)},
            }
        )

    return {
        "view": "summary",
        "capacity": _capacity(
            total=sum(server.gpu.capacity(show_shards) for server in servers),
            used=sum(server.gpu.occupied(show_shards) for server in servers),
            unit=unit,
        ),
        "gpu_types": gpu_types,
    }


def nodes_json(
    servers: Sequence[ServerState],
    *,
    show_shards: bool,
) -> dict[str, Any]:
    nodes = []
    for server in servers:
        gpu_usage = (
            server.usage["shard"]
            if show_shards and server.gpu.shards > 0
            else server.usage["gpu"]
        )
        partition_names = set(gpu_usage.partitions)
        partition_names.update(server.usage["cpu"].partitions)
        partition_names.update(server.usage["mem"].partitions)
        partitions = {}
        for partition in sorted(partition_names):
            values = {
                "gpu": _number(gpu_usage.partitions.get(partition, 0.0)),
                "cpu": _number(server.usage["cpu"].partitions.get(partition, 0.0)),
                "memory_gib": _number(
                    server.usage["mem"].partitions.get(partition, 0.0)
                ),
            }
            if any(values.values()):
                partitions[partition] = values

        nodes.append(
            {
                "name": server.name,
                "gpu_type": _display_gpu_type(server),
                "gpu": _capacity(
                    total=server.gpu.capacity(show_shards),
                    used=server.gpu.occupied(show_shards),
                    unit="shard" if show_shards else "GPU",
                ),
                "cpu": _capacity(
                    total=server.cpu.total,
                    used=server.cpu.total - server.cpu.idle,
                    unit="CPU",
                ),
                "memory": _capacity(
                    total=server.mem.total / 1024,
                    used=(server.mem.total - server.mem.idle) / 1024,
                    unit="GiB",
                ),
                "usage": {"partitions": partitions},
            }
        )
    return {"view": "nodes", "nodes": nodes}


def top_users_json(
    users: Sequence[UserUsage],
    *,
    unit: str,
) -> dict[str, Any]:
    return {
        "view": "top-users",
        "unit": unit,
        "users": [
            {
                "rank": rank,
                "user": user.user,
                "total": user.total_usage(),
                "nodes": sorted(user.nodes),
                "partitions": dict(sorted(user.usage_by_partition.items())),
            }
            for rank, user in enumerate(users, start=1)
        ],
    }


def jobs_json(jobs: Sequence[JobRecord]) -> dict[str, Any]:
    return {
        "view": "jobs",
        "jobs": [
            {
                "job_id": job.job_id,
                "user": job.user,
                "job_name": job.job_name,
                "state": job.state,
                "partition": job.partition,
                "nodelist": job.nodelist,
                "resources": {
                    "gpu": _number(job.usage.gpu),
                    "shards": _number(job.usage.shard),
                    "cpu": _number(job.usage.cpu),
                    "memory_gib": _number(job.usage.mem),
                },
                "time_limit": job.time_limit,
                "constraints": sorted(job.constraints),
            }
            for job in jobs
        ],
    }
