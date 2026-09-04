"""TUI user settings (theme, etc.). Stored in tui-settings.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_HF_AUTHORS = ["bartowski", "unsloth"]
DEFAULT_LOG_VERBOSITY = 4
LOG_VERBOSITY_CYCLE = (3, 4, 5)
LOG_VERBOSITY_LABELS = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "TRACE", 5: "DEBUG"}


@dataclass
class TUISettings:
    theme: str | None = None
    hf_authors: list[str] = field(default_factory=lambda: list(DEFAULT_HF_AUTHORS))
    log_verbosity: int = DEFAULT_LOG_VERBOSITY


def clamp_log_verbosity(value: object, default: int = DEFAULT_LOG_VERBOSITY) -> int:
    try:
        level = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if 0 <= level <= 5:
        return level
    return default


def cycle_log_verbosity(current: int) -> int:
    try:
        index = LOG_VERBOSITY_CYCLE.index(current)
    except ValueError:
        return DEFAULT_LOG_VERBOSITY
    return LOG_VERBOSITY_CYCLE[(index + 1) % len(LOG_VERBOSITY_CYCLE)]


def log_verbosity_label(level: int) -> str:
    return LOG_VERBOSITY_LABELS.get(level, str(level))


def load_settings(path: Path) -> TUISettings:
    if not path.exists():
        return TUISettings()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return TUISettings()
    theme = data.get("theme")
    hf_authors_raw = data.get("hf_authors", DEFAULT_HF_AUTHORS)
    hf_authors: list[str] = []
    if isinstance(hf_authors_raw, list):
        hf_authors = sorted({str(a).strip() for a in hf_authors_raw if str(a).strip()})
    return TUISettings(
        theme=theme if isinstance(theme, str) and theme else None,
        hf_authors=hf_authors,
        log_verbosity=clamp_log_verbosity(data.get("log_verbosity", DEFAULT_LOG_VERBOSITY)),
    )


def save_settings(path: Path, settings: TUISettings) -> None:
    data: dict[str, object] = {}
    if settings.theme:
        data["theme"] = settings.theme
    if settings.hf_authors:
        data["hf_authors"] = list(settings.hf_authors)
    data["log_verbosity"] = clamp_log_verbosity(settings.log_verbosity)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def remember_hf_author(settings: TUISettings, author: str) -> bool:
    """Append author to remembered list if new. Returns True if changed."""
    author = author.strip()
    if not author or author in settings.hf_authors:
        return False
    settings.hf_authors = sorted([*settings.hf_authors, author])
    return True
