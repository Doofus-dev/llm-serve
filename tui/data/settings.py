"""TUI user settings (theme, etc.). Stored in tui-settings.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TUISettings:
    theme: str | None = None


def load_settings(path: Path) -> TUISettings:
    if not path.exists():
        return TUISettings()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return TUISettings()
    theme = data.get("theme")
    return TUISettings(theme=theme if isinstance(theme, str) and theme else None)


def save_settings(path: Path, settings: TUISettings) -> None:
    data: dict[str, str] = {}
    if settings.theme:
        data["theme"] = settings.theme
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
