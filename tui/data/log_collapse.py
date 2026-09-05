"""Collapse consecutive llama.cpp log runs into live-updating rows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_ROWS = 200
FALLBACK_SESSION_LINES = 400

_LAUNCH = re.compile(
    r"^──\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+launch:\s+\S+\s+\(PID\s+\d+\)\s+──\s*$"
)
_TIMESTAMPED = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+([IWED])\s+(.*)$"
)
_NUMBERS = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class DisplayRow:
    family: str
    text: str
    count: int = 1
    raw_lines: list[str] = field(default_factory=list)

    def render(self) -> str:
        escaped = self.text.replace("[", "\\[")
        if self.count > 1:
            return f"{escaped}  [dim]×{self.count}[/]"
        return escaped


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


class LogCollapser:
    """Fold consecutive same-family lines into one row with a live count."""

    def __init__(self, max_rows: int = MAX_ROWS) -> None:
        self.max_rows = max_rows
        self.rows: list[DisplayRow] = []

    def reset(self) -> None:
        self.rows.clear()

    def feed_text(self, text: str) -> None:
        self.feed_lines(text.splitlines())

    def feed_lines(self, lines: list[str]) -> None:
        for raw in lines:
            self.feed_line(raw)

    def feed_line(self, line: str) -> bool:
        """Ingest one line. Return True if the display rows changed."""
        stripped = line.rstrip("\n\r")
        if not stripped.strip():
            return False

        family = classify_family(stripped)
        if family is None:
            family = "raw:" + stripped

        if self.rows and self.rows[-1].family == family:
            last = self.rows[-1]
            last.text = stripped
            last.count += 1
            last.raw_lines.append(stripped)
            return True

        self.rows.append(DisplayRow(family=family, text=stripped, raw_lines=[stripped]))
        self._trim()
        return True

    def _trim(self) -> None:
        extra = len(self.rows) - self.max_rows
        if extra > 0:
            del self.rows[:extra]


class LogTailer:
    """Incrementally read a log file and collapse consecutive runs."""

    def __init__(self) -> None:
        self.collapser = LogCollapser()
        self._offset = 0
        self._partial = ""
        self._started = False

    def reset(self) -> None:
        self.collapser.reset()
        self._offset = 0
        self._partial = ""
        self._started = False

    @property
    def rows(self) -> list[DisplayRow]:
        return self.collapser.rows

    def poll(self, path: Path) -> bool:
        """Feed new bytes. Return True if display rows changed."""
        if not path.exists():
            had_rows = bool(self.collapser.rows) or self._started
            self.reset()
            return had_rows

        size = path.stat().st_size
        if not self._started or size < self._offset:
            self.collapser.reset()
            text = path.read_bytes().decode("utf-8", errors="replace")
            session = slice_to_session(text)
            self._offset = size
            self._partial = ""
            self._started = True
            self.collapser.feed_text(session)
            return True

        if size == self._offset and not self._partial:
            return False

        with path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        self._offset += len(chunk)
        text = self._partial + chunk.decode("utf-8", errors="replace")
        if text.endswith("\n"):
            lines = text.splitlines()
            self._partial = ""
        else:
            parts = text.splitlines()
            if not parts:
                self._partial = text
                return False
            lines = parts[:-1]
            self._partial = parts[-1]
        before = len(self.collapser.rows)
        last = (
            (self.collapser.rows[-1].count, self.collapser.rows[-1].text)
            if self.collapser.rows
            else None
        )
        self.collapser.feed_lines(lines)
        if not self.collapser.rows:
            return before != 0
        now = (self.collapser.rows[-1].count, self.collapser.rows[-1].text)
        return before != len(self.collapser.rows) or last != now
