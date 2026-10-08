"""Read and write PID metadata for one or more managed llama-server processes.

The legacy ``.llm-serve.pid`` file is the tracked server (``stop`` with no
argument). Concurrent launches also write ``<log_dir>/instances/<pid>.pid``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PidInfo:
    pid: int
    model: str
    port: int
    ts: str
    quant: str | None = None
    preset_slot: int | None = None
    remote: bool = False

    @property
    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OverflowError):
            return False
        except OSError:
            return False


def instances_dir(log_dir: Path) -> Path:
    return log_dir / "instances"


def instance_file(log_dir: Path, pid: int) -> Path:
    return instances_dir(log_dir) / f"{pid}.pid"


def instance_log_file(log_dir: Path, port: int) -> Path:
    return instances_dir(log_dir) / f"{port}.log"


def read_pid_file(path: Path) -> PidInfo | None:
    try:
        parts = path.read_text().split()
        return PidInfo(
            pid=int(parts[0]),
            model=parts[1],
            port=int(parts[2]),
            ts=parts[3] if len(parts) > 3 else "",
            quant=parts[4] if len(parts) > 4 else None,
            preset_slot=int(parts[5]) if len(parts) > 5 else None,
            remote=len(parts) > 6 and parts[6] == "1",
        )
    except (OSError, ValueError, IndexError):
        return None


def write_pid_file(
    path: Path,
    *,
    pid: int,
    model: str,
    port: int,
    quant: str,
    preset_slot: int,
    remote: bool,
    started_at: int | None = None,
) -> None:
    ts = int(time.time() if started_at is None else started_at)
    remote_flag = "1" if remote else "0"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{pid} {model} {port} {ts} {quant} {preset_slot} {remote_flag}\n"
    )


def clear_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def record_instance(
    *,
    pid_file: Path,
    log_dir: Path,
    pid: int,
    model: str,
    port: int,
    quant: str,
    preset_slot: int,
    remote: bool,
    started_at: int | None = None,
) -> None:
    """Persist a launch: instance file always, tracked pid file if vacant."""
    write_pid_file(
        instance_file(log_dir, pid),
        pid=pid,
        model=model,
        port=port,
        quant=quant,
        preset_slot=preset_slot,
        remote=remote,
        started_at=started_at,
    )
    tracked = read_pid_file(pid_file)
    if tracked is None or not tracked.alive:
        write_pid_file(
            pid_file,
            pid=pid,
            model=model,
            port=port,
            quant=quant,
            preset_slot=preset_slot,
            remote=remote,
            started_at=started_at,
        )


def forget_instance(*, pid_file: Path, log_dir: Path, pid: int) -> None:
    clear_pid_file(instance_file(log_dir, pid))
    tracked = read_pid_file(pid_file)
    if tracked is not None and tracked.pid == pid:
        clear_pid_file(pid_file)


def list_instances(*, pid_file: Path, log_dir: Path) -> list[PidInfo]:
    """Alive servers: instance dir plus the legacy tracked pid file."""
    seen: dict[int, PidInfo] = {}
    directory = instances_dir(log_dir)
    if directory.is_dir():
        for path in sorted(directory.glob("*.pid")):
            info = read_pid_file(path)
            if info is None:
                continue
            if not info.alive:
                clear_pid_file(path)
                continue
            seen[info.pid] = info
    tracked = read_pid_file(pid_file)
    if tracked is not None and tracked.alive and tracked.pid not in seen:
        seen[tracked.pid] = tracked
    tracked_pid = tracked.pid if tracked is not None and tracked.alive else None
    return sorted(
        seen.values(),
        key=lambda info: (info.pid != tracked_pid, info.ts, info.pid),
    )


def remap_pid_preset_slots(
    path: Path, remaps: dict[tuple[str, str], dict[int, int]]
) -> bool:
    """Update a running server's recorded preset slot after compacting. Returns True if rewritten."""
    info = read_pid_file(path)
    if (
        info is None
        or not info.alive
        or info.quant is None
        or info.preset_slot is None
    ):
        return False
    mapping = remaps.get((info.model, info.quant))
    if not mapping:
        return False
    new_slot = mapping.get(info.preset_slot)
    if new_slot is None or new_slot == info.preset_slot:
        return False
    write_pid_file(
        path,
        pid=info.pid,
        model=info.model,
        port=info.port,
        quant=info.quant,
        preset_slot=new_slot,
        remote=info.remote,
        started_at=int(info.ts) if info.ts else None,
    )
    return True


def remap_all_preset_slots(
    pid_file: Path,
    log_dir: Path,
    remaps: dict[tuple[str, str], dict[int, int]],
) -> bool:
    """Remap preset slots on the tracked pid file and every instance file."""
    changed = remap_pid_preset_slots(pid_file, remaps)
    directory = instances_dir(log_dir)
    if directory.is_dir():
        for path in directory.glob("*.pid"):
            if remap_pid_preset_slots(path, remaps):
                changed = True
    return changed
