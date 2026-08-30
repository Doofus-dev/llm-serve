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
        return PidInfo(pid=int(parts[0]), model=parts[1], port=int(parts[2]),
                       ts=parts[3] if len(parts) > 3 else "")
    except (OSError, ValueError, IndexError):
        return None
