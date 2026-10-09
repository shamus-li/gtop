from __future__ import annotations

from typing import Any, Sequence

from .models import JobRecord, JobUsage, ServerState, UserUsage
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


def node_capacity(server: ServerState) -> dict[str, Any]:
    return {
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
    }


def job_resources(usage: JobUsage) -> dict[str, int | float]:
    return {
        "gpu": _number(usage.gpu),
        "shards": _number(usage.shard),
        "cpu": _number(usage.cpu),
        "memory_gib": _number(usage.mem),
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
                **node_capacity(server),
                "usage": {"partitions": partitions},
            }
        )
    return {"view": "nodes", "nodes": nodes}


def users_json(
    users: Sequence[UserUsage],
    *,
    unit: str,
) -> dict[str, Any]:
    return {
        "view": "users",
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
                "resources": job_resources(job.usage),
                "time_limit": job.time_limit,
                "constraints": sorted(job.constraints),
            }
            for job in jobs
        ],
    }
