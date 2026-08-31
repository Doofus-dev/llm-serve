"""Parameter help text parsed from models.conf.example."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_CONF_EXAMPLE = REPO_ROOT / "models.conf.example"

_PARAM_HEADER = re.compile(r"^#\s+([\w-]+(?:\s*/\s*[\w-]+)*)\s*(.*)$")
_CONTINUATION = re.compile(r"^#\s{4,}(.+)$")

# Params used in the TUI but not documented in models.conf.example
_FALLBACK: dict[str, str] = {
    "metrics": "Enable Prometheus /metrics endpoint on the server (on | off). Used by the TUI for live throughput stats.",
}


def parse_param_help(text: str) -> dict[str, str]:
    """Parse parameter documentation from models.conf.example comment blocks."""
    help_map: dict[str, str] = {}
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        m = _PARAM_HEADER.match(line)
        if not m:
            i += 1
            continue

        params_part = m.group(1)
        if params_part.startswith("═") or params_part.startswith("─"):
            i += 1
            continue

        inline = m.group(2).strip()
        desc_parts: list[str] = [inline] if inline else []
        i += 1

        while i < len(lines):
            cont = _CONTINUATION.match(lines[i])
            if cont:
                desc_parts.append(cont.group(1).strip())
                i += 1
                continue
            if lines[i].strip() == "":
                i += 1
                continue
            break

        desc = " ".join(p for p in desc_parts if p).strip()
        if not desc:
            continue

        for param in re.split(r"\s*/\s*", params_part):
            param = param.strip()
            if param:
                help_map[param] = desc

    return help_map


@lru_cache(maxsize=1)
def load_param_help(path: Path | None = None) -> dict[str, str]:
    """Load param help, falling back to built-ins for gaps."""
    src = path or MODELS_CONF_EXAMPLE
    if not src.exists():
        return dict(_FALLBACK)

    merged = parse_param_help(src.read_text())
    merged.update(_FALLBACK)
    return merged


def get_param_help(param: str, path: Path | None = None) -> str | None:
    return load_param_help(path).get(param)
