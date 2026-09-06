"""Classify llama.cpp log lines into collapse families.

The live TUI feed uses ``tui.data.server_log.LogTailer``. This module only
owns family classification shared with that parser.
"""

from __future__ import annotations

import re

FALLBACK_SESSION_LINES = 400

_LAUNCH = re.compile(
    r"^──\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+launch:\s+\S+\s+\(PID\s+\d+\)\s+──\s*$"
)
_TIMESTAMPED = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+([IWED])\s+(.*)$"
)
_NUMBERS = re.compile(r"\d+(?:\.\d+)?")


def is_launch_marker(line: str) -> bool:
    return bool(_LAUNCH.match(line.rstrip("\n\r")))


def slice_to_session(text: str, fallback_lines: int = FALLBACK_SESSION_LINES) -> str:
    """Keep from the last llm-serve launch marker, else the last N lines."""
    lines = text.splitlines(keepends=True)
    idx = None
    for i, line in enumerate(lines):
        if is_launch_marker(line):
            idx = i
    if idx is not None:
        return "".join(lines[idx:])
    return "".join(lines[-fallback_lines:])


def _body_after_timestamp(line: str) -> str | None:
    match = _TIMESTAMPED.match(line)
    if not match:
        return None
    return match.group(6)


def _normalize(text: str) -> str:
    return " ".join(_NUMBERS.sub("N", text).split())


def classify_family(line: str) -> str | None:
    """Return a collapse family, or None for timestamp-less continuation lines."""
    stripped = line.rstrip("\n\r")
    if not stripped.strip():
        return None
    if is_launch_marker(stripped):
        return "launch"

    body = _body_after_timestamp(stripped)
    if body is None:
        return None

    lowered = body.lower()
    if "all slots are idle" in lowered:
        return "idle"
    if "cached n_tokens =" in lowered:
        return "cached_tokens"
    if "prompt processing" in lowered:
        return "prompt_progress"
    if "n_gen =" in lowered:
        return "gen_ticks"
    if "created context checkpoint" in lowered:
        return "checkpoint"
    if "restored context checkpoint" in lowered:
        return "checkpoint_restore"
    return "norm:" + _normalize(body)
