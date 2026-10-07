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
    unavailable: float = 0.0,
) -> dict[str, int | float | str]:
    return {
        "total": _number(total),
        "used": _number(used),
        "free": _number(max(total - used - unavailable, 0)),
        "unavailable": _number(unavailable),
        "unit": unit,
    }


def _partitions(values: dict[str, float]) -> dict[str, int | float]:
    return {
        partition: _number(amount)
        for partition, amount in sorted(values.items())
        if amount > 0
    }


def summary_json(servers: Sequence[ServerState]) -> dict[str, Any]:
    unit = "GPU"
    grouped: dict[str, list[ServerState]] = {}
    for server in servers:
        grouped.setdefault(_display_gpu_type(server), []).append(server)

    gpu_types = []
    total_capacity = 0
    total_used = 0.0
    for gpu_type, grouped_servers in sorted(grouped.items()):
        total = sum(server.gpu.num for server in grouped_servers)
        partition_totals: dict[str, float] = {}
        for server in grouped_servers:
            for partition, amount in server.usage["gpu"].partitions.items():
                partition_totals[partition] = (
                    partition_totals.get(partition, 0.0) + amount
                )
        used = sum(partition_totals.values())
        if used <= 0:
            continue
        total_capacity += total
        total_used += used

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
            total=total_capacity,
            used=total_used,
            unit=unit,
        ),
        "gpu_types": gpu_types,
    }


def nodes_json(servers: Sequence[ServerState]) -> dict[str, Any]:
    nodes = []
    for server in servers:
        gpu_usage = server.usage["gpu"]
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
                "accepts_jobs": server.accepts_jobs,
                "gpu": _capacity(
                    total=server.gpu.num,
                    used=server.gpu.occupied(),
                    unit="GPU",
                    unavailable=0
                    if server.accepts_jobs
                    else server.gpu.num - server.gpu.occupied(),
                ),
                "cpu": _capacity(
                    total=server.cpu.total,
                    used=server.cpu.total - server.cpu.idle - server.cpu.other,
                    unit="CPU",
                    unavailable=server.cpu.other
                    + (0 if server.accepts_jobs else server.cpu.idle),
                ),
                "memory": _capacity(
                    total=server.mem.total / 1024,
                    used=(server.mem.total - server.mem.idle) / 1024,
                    unit="GiB",
                    unavailable=0 if server.accepts_jobs else server.mem.idle / 1024,
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
