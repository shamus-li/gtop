from __future__ import annotations

from typing import Dict, List, Optional

from .constants import SINFO_FIELD_WIDTHS
from .models import JobRecord, ServerState
from .resources import parse_cpu, parse_gpu, parse_mem, parse_usage


def expand_range(value: str) -> List[str]:
    if "-" not in value:
        return [value]

    try:
        start_str, end_str = value.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        width = max(len(start_str), len(end_str))
        return [str(number).zfill(width) for number in range(start, end + 1)]
    except ValueError:
        return [value]


def parse_nodelist(nodelist: str) -> List[str]:
    nodes: List[str] = []
    bracket_depth = 0
    start = 0

    for index, char in enumerate(nodelist):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and bracket_depth == 0:
            nodes.append(nodelist[start:index])
            start = index + 1
    nodes.append(nodelist[start:])

    expanded: List[str] = []
    for node in nodes:
        if "[" in node and "]" in node:
            base = node[: node.find("[")]
            ranges = node[node.find("[") + 1 : node.find("]")]
            for item in ranges.split(","):
                for suffix in expand_range(item):
                    expanded.append(base + suffix)
        else:
            expanded.append(node)
    return expanded


def parse_features_field(features: str) -> set[str]:
    if not features:
        return set()

    parsed = []
    for raw in features.replace("|", ",").split(","):
        cleaned = raw.strip().lower()
        if not cleaned or cleaned == "(null)":
            continue
        base_feature = cleaned.split("*", 1)[0].strip()
        if base_feature:
            parsed.append(base_feature)
    return set(parsed)


def _split_job_line(line: str) -> Optional[List[str]]:
    if not line:
        return None
    if "|" in line:
        parts = [segment.strip() for segment in line.rstrip("|").split("|")]
    else:
        parts = line.split()
    return parts if len(parts) >= 6 else None


def parse_jobs(output: str) -> List[JobRecord]:
    jobs: List[JobRecord] = []
    if not output.strip():
        return jobs

    for line in output.strip().splitlines():
        parts = _split_job_line(line.strip())
        if not parts:
            continue
        if len(parts) >= 8:
            user, job_id, job_name, state, partition, nodelist, usage_str, time_limit = parts[:8]
        else:
            user, partition, nodelist, state, usage_str, job_id = parts[:6]
            job_name = ""
            time_limit = ""
        usage = parse_usage(usage_str)
        jobs.append(
            JobRecord(
                user=user,
                job_id=job_id,
                job_name=job_name,
                state=state,
                partition=partition,
                nodelist=nodelist,
                usage_str=usage_str,
                time_limit=time_limit,
                cpu=usage.cpu,
                gpu=usage.gpu,
                mem=usage.mem,
                shard=usage.shard,
            )
        )
    return jobs


def _split_sinfo_line(line: str) -> Optional[List[str]]:
    if not line:
        return None
    if "|" in line:
        parts = [segment.strip() for segment in line.split("|")]
        return parts if len(parts) >= 7 else None

    fields: List[str] = []
    start = 0
    for width in SINFO_FIELD_WIDTHS:
        end = start + width
        fields.append(line[start:end].strip())
        start = end
    return fields if len(fields) >= 7 else None


def parse_sinfo(output: str, gpu_only: bool) -> Dict[str, ServerState]:
    servers: Dict[str, ServerState] = {}
    for raw_line in output.strip().splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        parts = _split_sinfo_line(line)
        if not parts:
            continue

        node_name, features_raw, gres, gres_used, cpu_state, alloc_mem, total_mem = (
            parts[:7]
        )
        if not node_name or not cpu_state or not total_mem:
            continue

        gpu = parse_gpu(gres, gres_used)
        if gpu_only and gpu.type == "null":
            continue

        for server_name in parse_nodelist(node_name):
            servers[server_name] = ServerState(
                name=server_name,
                features=parse_features_field(features_raw),
                gpu=gpu,
                cpu=parse_cpu(cpu_state),
                mem=parse_mem(alloc_mem, total_mem),
            )
    return servers
