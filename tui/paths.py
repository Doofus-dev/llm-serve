"""Repo paths for the TUI, CLI, and launcher. Overridable via env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent

METRICS_POLL_INTERVAL = 0.5
METRICS_HISTORY_SAMPLES = 120
MAX_LOG_EVENTS = 200
LOCKED_PARAMS = frozenset({"total_layers", "context_length"})
PROFILE_KEYS = ("display", "port", "host", "notes")


@dataclass(frozen=True)
class AppPaths:
    root: Path
    models_json: Path
    presets_json: Path
    models_dir: Path
    llama_dir: Path
    llama_server: Path
    log_dir: Path
    log_file: Path
    pid_file: Path
    settings_json: Path
    baselines_json: Path


def default_paths(*, env: Mapping[str, str] | None = None, root: Path | None = None) -> AppPaths:
    env = os.environ if env is None else env
    base = Path(root) if root is not None else REPO_ROOT
    llama_dir = Path(env.get("LLAMA_DIR") or (base / "llama.cpp"))
    models_dir = Path(env.get("MODEL_DIR") or (base / "models"))
    log_dir = Path(env.get("LOG_DIR") or (base / "logs"))
    log_file = Path(env.get("LOG_FILE") or (log_dir / "llm-serve.log"))
    pid_file = Path(env.get("PID_FILE") or (log_dir / ".llm-serve.pid"))
    return AppPaths(
        root=base,
        models_json=base / "models.json",
        presets_json=base / "presets.json",
        models_dir=models_dir,
        llama_dir=llama_dir,
        llama_server=llama_dir / "build" / "bin" / "llama-server",
        log_dir=log_dir,
        log_file=log_file,
        pid_file=pid_file,
        settings_json=base / "tui-settings.json",
        baselines_json=base / "tui-baselines.json",
    )
