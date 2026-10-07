from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .runner import Command, CommandRunner, SubprocessRunner

# Window: (days, seconds before a cached result is refreshed in the background).
# The year query loads slurmdbd for over a minute, so it only refreshes on request.
WINDOWS = {"week": (7, 3600), "month": (30, 6 * 3600), "year": (365, None)}


@dataclass(frozen=True)
class UsageHistory:
    generated_at: float
    gpu_hours: dict[str, float]
    accounts: dict[str, list[str]]

    def age_seconds(self) -> float:
        return max(time.time() - self.generated_at, 0.0)


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
    temporary.write_text(json.dumps(asdict(history)))
    temporary.replace(path)


def fetch_history(
    window: str,
    *,
    runner: Optional[CommandRunner] = None,
    timeout: int,
) -> UsageHistory:
    result = (runner or SubprocessRunner()).run(history_command(window), timeout)
    if result.returncode != 0:
        raise HistoryError(result.stderr.strip() or "sreport history query failed")
    history = parse_history(result.stdout, now=time.time())
    _write_cache(window, history)
    return history


def _claim_refresh(window: str) -> bool:
    lock = _cache_path(window).with_suffix(".lock")
    try:
        if time.time() - lock.stat().st_mtime < 600:
            return False
        lock.unlink()
    except FileNotFoundError:
        pass
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return False
    return True


def _refresh_in_background(window: str) -> None:
    if not _claim_refresh(window):
        return
    subprocess.Popen(
        (sys.executable, "-m", "gtop.history", window),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def cached_history(window: str) -> Optional[UsageHistory]:
    """Return cached history, starting a background refresh when it is stale."""
    history = _read_cache(window)
    max_age = WINDOWS[window][1]
    if history is not None and max_age is not None and history.age_seconds() > max_age:
        _refresh_in_background(window)
    return history


if __name__ == "__main__":
    try:
        fetch_history(sys.argv[1], timeout=900)
    finally:
        _cache_path(sys.argv[1]).with_suffix(".lock").unlink(missing_ok=True)
