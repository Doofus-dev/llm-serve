"""TUI user settings (theme, etc.). Stored in tui-settings.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_HF_AUTHORS = ["bartowski", "unsloth"]


@dataclass
class TUISettings:
    theme: str | None = None
    hf_authors: list[str] = field(default_factory=lambda: list(DEFAULT_HF_AUTHORS))


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
    )


def save_settings(path: Path, settings: TUISettings) -> None:
    data: dict[str, object] = {}
    if settings.theme:
        data["theme"] = settings.theme
    if settings.hf_authors:
        data["hf_authors"] = list(settings.hf_authors)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def remember_hf_author(settings: TUISettings, author: str) -> bool:
    """Append author to remembered list if new. Returns True if changed."""
    author = author.strip()
    if not author or author in settings.hf_authors:
        return False
    settings.hf_authors = sorted([*settings.hf_authors, author])
    return True
