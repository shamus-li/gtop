from __future__ import annotations

from dataclasses import replace
from typing import Collection, Dict, List, Optional, Set

from .constants import (
    SACCT_DELIMITER,
    SACCT_FIELD_COUNT,
    SINFO_FIELD_WIDTHS,
    STALE_RUNNING_GRACE_SECONDS,
)
from .constraints import requested_constraint_features
from .models import JobRecord, ServerState
from .resources import parse_cpu, parse_gpu, parse_mem, parse_usage


def expand_range(value: str) -> List[str]:
    if "-" not in value:
        if not value.isdigit():
            raise ValueError(f"Invalid host range '{value}'")
        return [value]

    try:
        start_str, end_str = value.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError
        width = max(len(start_str), len(end_str))
        return [str(number).zfill(width) for number in range(start, end + 1)]
    except ValueError:
        raise ValueError(f"Invalid host range '{value}'") from None


def _expand_host(value: str) -> List[str]:
    open_bracket = value.find("[")
    if open_bracket < 0:
        if "]" in value:
            raise ValueError(f"Invalid hostlist '{value}'")
        return [value]

    close_bracket = value.find("]", open_bracket + 1)
    if close_bracket < 0 or "[" in value[open_bracket + 1 : close_bracket]:
        raise ValueError(f"Invalid hostlist '{value}'")

    prefix = value[:open_bracket]
    ranges = value[open_bracket + 1 : close_bracket]
    suffix = value[close_bracket + 1 :]
    if not ranges:
        raise ValueError(f"Invalid hostlist '{value}'")

    expanded_suffixes = _expand_host(suffix)
    return [
        f"{prefix}{number}{expanded_suffix}"
        for item in ranges.split(",")
        for number in expand_range(item)
        for expanded_suffix in expanded_suffixes
    ]


def parse_nodelist(nodelist: str) -> List[str]:
    nodelist = nodelist.strip()
    if nodelist.upper() in {"", "N/A", "(NULL)", "NONE", "NONE ASSIGNED"}:
        return []

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

    if bracket_depth != 0 or any(not node for node in nodes):
        raise ValueError(f"Invalid hostlist '{nodelist}'")

    return [host for node in nodes for host in _expand_host(node)]


def array_task_count(job_id: str) -> int:
    """Tasks in a pending array record such as 123_[1-5,9:2%4]; 1 for any other job."""
    _, bracket, tasks = job_id.partition("_[")
    if not bracket:
        return 1
    count = 0
    for item in tasks.rstrip("]").split("%", 1)[0].split(","):
        span, _, step = item.partition(":")
        start, _, end = span.partition("-")
        count += (int(end or start) - int(start)) // int(step or 1) + 1
    return count


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


def _canonical_job_state(state: str) -> str:
    words = state.split(maxsplit=1)
    return words[0].removesuffix("+") if words else ""


def _valid_job_fields(
    user: str,
    job_id: str,
    state: str,
    usage: str,
) -> bool:
    state_code = state.replace("_", "")
    return bool(
        user
        and job_id
        and job_id[0].isdigit()
        and state_code.isalpha()
        and state_code.isupper()
        and (not usage or "=" in usage)
    )


def _duration_seconds(value: str) -> Optional[int]:
    days, _, clock = value.rpartition("-")
    fields = clock.split(":")
    if days and not days.isdigit():
        return None
    if len(fields) > 3 or not all(field.isdigit() for field in fields):
        return None
    hours, minutes, seconds = [0] * (3 - len(fields)) + [int(field) for field in fields]
    return int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds


def _outlived_time_limit(elapsed: str, time_limit: str) -> bool:
    elapsed_seconds = _duration_seconds(elapsed)
    limit_seconds = _duration_seconds(time_limit)
    if elapsed_seconds is None or limit_seconds is None:
        return False
    return elapsed_seconds > limit_seconds + STALE_RUNNING_GRACE_SECONDS


def parse_jobs(
    output: str,
    *,
    states: Optional[Collection[str]] = None,
) -> List[JobRecord]:
    jobs: List[JobRecord] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(SACCT_DELIMITER)]
        if len(fields) != SACCT_FIELD_COUNT:
            raise ValueError(f"Malformed sacct record on line {line_number}")
        (
            user,
            job_id,
            job_name,
            state,
            partition,
            nodelist,
            constraints_str,
            usage_str,
            time_limit,
            elapsed,
            requested_str,
            reason,
        ) = fields
        if constraints_str == "(null)":  # squeue's empty Feature
            constraints_str = ""
        state = _canonical_job_state(state)
        usage_str = usage_str or requested_str
        if not _valid_job_fields(user, job_id, state, usage_str):
            raise ValueError(f"Malformed sacct record on line {line_number}")
        if states is not None and state not in states:
            continue
        if state == "RUNNING" and _outlived_time_limit(elapsed, time_limit):
            continue
        try:
            usage = parse_usage(usage_str)
        except ValueError as error:
            raise ValueError(
                f"Malformed sacct record on line {line_number}: {error}"
            ) from error
        jobs.append(
            JobRecord(
                user=user,
                job_id=job_id,
                job_name=job_name,
                state=state,
                partition=partition,
                nodelist=nodelist,
                usage=usage,
                time_limit=time_limit,
                elapsed=elapsed,
                reason="" if reason == "None" else reason,
                constraints=requested_constraint_features(constraints_str),
            )
        )
    return jobs


def _split_sinfo_line(line: str) -> Optional[List[str]]:
    if not line:
        return None
    # sinfo pads every field to its width; shorter "|"-separated lines come from
    # tests. Length decides, since a drain reason may itself contain "|".
    if len(line) < sum(SINFO_FIELD_WIDTHS[:-1]):
        parts = [segment.strip() for segment in line.split("|")]
        return parts if len(parts) == len(SINFO_FIELD_WIDTHS) else None

    fields: List[str] = []
    start = 0
    for width in SINFO_FIELD_WIDTHS:
        end = start + width
        fields.append(line[start:end].strip())
        start = end
    return fields if not line[start:].strip() else None


def parse_sinfo(output: str) -> Dict[str, ServerState]:
    servers: Dict[str, ServerState] = {}
    # sinfo -N repeats each node once per partition with identical fields.
    seen_lines: Set[str] = set()
    for line_number, raw_line in enumerate(output.strip().splitlines(), start=1):
        line = raw_line.rstrip("\n")
        if not line.strip() or line in seen_lines:
            continue
        seen_lines.add(line)

        parts = _split_sinfo_line(line)
        if not parts:
            raise ValueError(f"Malformed sinfo record on line {line_number}")

        (
            node_name,
            features_raw,
            gres,
            gres_used,
            cpu_state,
            alloc_mem,
            total_mem,
            state,
            reason,
        ) = parts
        if not node_name or not cpu_state or not total_mem:
            raise ValueError(f"Malformed sinfo record on line {line_number}")

        try:
            gpu = parse_gpu(gres, gres_used)
            cpu = parse_cpu(cpu_state)
            mem = parse_mem(alloc_mem, total_mem)
            server_names = parse_nodelist(node_name)
        except ValueError as error:
            raise ValueError(
                f"Malformed sinfo record on line {line_number}: {error}"
            ) from error
        if not server_names:
            raise ValueError(f"Malformed sinfo record on line {line_number}")
        for server_name in server_names:
            servers[server_name] = ServerState(
                name=server_name,
                features=parse_features_field(features_raw),
                gpu=replace(gpu),
                cpu=replace(cpu),
                mem=replace(mem),
                state=state,
                reason="" if reason == "none" else reason,
            )
    return servers
