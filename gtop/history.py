from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .runner import Command, CommandRunner, SubprocessRunner

# Window: (days, seconds a cached result is shown before the next run refreshes
# it). The year query loads slurmdbd for over a minute, so it only refreshes on
# request.
WINDOWS = {"week": (7, 15 * 60), "month": (30, 15 * 60), "year": (365, None)}
SREPORT_TIMEOUT = 900


@dataclass(frozen=True)
class UsageHistory:
    generated_at: float
    gpu_hours: dict[str, float]
    accounts: dict[str, list[str]]

    def age_seconds(self) -> float:
        return max(time.time() - self.generated_at, 0.0)

    def ranked(self) -> list[tuple[str, float]]:
        """Users with at least half a GPU-hour, most first."""
        return [
            (user, hours)
            for user, hours in sorted(
                self.gpu_hours.items(), key=lambda item: (-item[1], item[0])
            )
            if round(hours) > 0
        ]


class HistoryError(RuntimeError):
    pass


def history_command(window: str) -> Command:
    # sreport reads slurmdbd's hourly/daily/monthly rollups, so long windows stay
    # cheap; summing raw sacct records for 30 days exhausts memory on Unicorn.
    days, _ = WINDOWS[window]
    return (
        "sreport",
        "-n",
        "-P",
        "-t",
        "hours",
        "user",
        "top",
        f"start=now-{days}days",
        "end=now",
        "TopCount=100000",
        "--tres=gres/gpu",
    )


def parse_history(output: str, *, now: float) -> UsageHistory:
    gpu_hours: dict[str, float] = {}
    accounts: dict[str, set[str]] = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) != 6:
            continue
        _, user, _, account, _, used = fields
        try:
            hours = float(used)
        except ValueError:
            continue
        gpu_hours[user] = gpu_hours.get(user, 0.0) + hours
        accounts.setdefault(user, set()).add(account)
    return UsageHistory(
        generated_at=now,
        gpu_hours=gpu_hours,
        accounts={user: sorted(names) for user, names in accounts.items()},
    )


def _cache_path(window: str) -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "gtop" / f"usage-{window}.json"


def _read_cache(window: str) -> Optional[UsageHistory]:
    try:
        data = json.loads(_cache_path(window).read_text())
        return UsageHistory(**data)
    except (OSError, ValueError, TypeError):
        return None


def _write_cache(window: str, history: UsageHistory) -> None:
    path = _cache_path(window)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    # Other users' GPU-hours: keep the cache private even under a shared cache root.
    with os.fdopen(
        os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
    ) as file:
        file.write(json.dumps(asdict(history)))
    temporary.replace(path)


def fetch_history(
    window: str,
    *,
    runner: Optional[CommandRunner] = None,
) -> UsageHistory:
    result = (runner or SubprocessRunner()).run(
        history_command(window), SREPORT_TIMEOUT
    )
    if result.returncode != 0:
        raise HistoryError(result.stderr.strip() or "sreport history query failed")
    history = parse_history(result.stdout, now=time.time())
    _write_cache(window, history)
    return history


def cached_history(window: str) -> Optional[UsageHistory]:
    """The cached history, or None when there is none or it is too old to show."""
    history = _read_cache(window)
    max_age = WINDOWS[window][1]
    if history is None or (max_age is not None and history.age_seconds() > max_age):
        return None
    return history
