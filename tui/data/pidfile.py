"""Read and write the PID file for the managed llama-server process."""

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
