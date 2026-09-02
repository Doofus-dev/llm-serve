"""Read the PID file written by the llm-serve bash launcher."""

from __future__ import annotations

import os
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
