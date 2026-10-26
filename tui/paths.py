"""Checkout vs machine-wide data paths. Overridable via env vars."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# LLM_SERVE_HOME overrides the *code* root for non-editable installs
# (pip install into site-packages where __file__ is wrong).
_ENV_HOME = os.environ.get("LLM_SERVE_HOME")
REPO_ROOT = Path(_ENV_HOME).resolve() if _ENV_HOME else Path(__file__).resolve().parent.parent

METRICS_POLL_INTERVAL = 0.5
METRICS_HISTORY_SAMPLES = 120
MAX_LOG_EVENTS = 200
LOCKED_PARAMS = frozenset({"total_layers", "context_length"})
PROFILE_KEYS = ("display", "port", "host", "notes")

# Shared across every checkout on this machine. Not llama.cpp (that stays
# with the code you launched).
STATE_FILES = (
    "models.json",
    "presets.json",
    "tui-settings.json",
    "tui-baselines.json",
    "tui-benchmarks.json",
)
EMPTY_REGISTRY = '{"models": {}, "aliases": {}}\n'
_MOVED_README = """# models/ moved

GGUF weights and the TUI registry live in the machine-wide data dir
(not this git checkout), so every branch and worktree sees the same files:

    ~/.local/share/llm-serve/

Override with LLM_SERVE_DATA. MODEL_DIR still overrides only the weights tree.
"""


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data: Path
    models_json: Path
    presets_json: Path
    models_dir: Path
    llama_dir: Path
    llama_server: Path
    llama_bench: Path
    log_dir: Path
    log_file: Path
    pid_file: Path
    settings_json: Path
    baselines_json: Path
    benchmarks_json: Path


def shared_data_home(*, env: Mapping[str, str] | None = None) -> Path:
    """Machine-wide data dir. Independent of which llm-serve checkout is running."""
    env = os.environ if env is None else env
    explicit = env.get("LLM_SERVE_DATA")
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = env.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "llm-serve"
    home = env.get("HOME") or str(Path.home())
    return Path(home).expanduser().resolve() / ".local" / "share" / "llm-serve"


def _dir_has_gguf(path: Path) -> bool:
    if not path.is_dir():
        return False
    return next(path.rglob("*.gguf"), None) is not None


def _dir_is_empty(path: Path) -> bool:
    if not path.exists():
        return True
    if not path.is_dir():
        return False
    return next(path.iterdir(), None) is None


def migrate_shared_data(
    repo: Path,
    data: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """Copy repo-local JSON and move GGUFs into the shared data dir once."""
    env = os.environ if env is None else env
    data.mkdir(parents=True, exist_ok=True)
    (data / "logs").mkdir(exist_ok=True)
    for name in STATE_FILES:
        src = repo / name
        dst = data / name
        if not dst.exists() and src.is_file():
            shutil.copy2(src, dst)
    models_json = data / "models.json"
    if not models_json.exists():
        models_json.write_text(EMPTY_REGISTRY)
    if "MODEL_DIR" in env:
        (data / "models").mkdir(exist_ok=True)
        return
    dest_models = data / "models"
    src_models = repo / "models"
    if _dir_has_gguf(dest_models):
        return
    if _dir_has_gguf(src_models):
        if dest_models.exists() and _dir_is_empty(dest_models):
            dest_models.rmdir()
        if dest_models.exists():
            return
        src_models.replace(dest_models)
        src_models.mkdir(parents=True, exist_ok=True)
        (src_models / "README.md").write_text(_MOVED_README)
        return
    dest_models.mkdir(exist_ok=True)


def default_paths(*, env: Mapping[str, str] | None = None, root: Path | None = None) -> AppPaths:
    env = os.environ if env is None else env
    base = Path(root) if root is not None else REPO_ROOT
    llama_dir = Path(env.get("LLAMA_DIR") or (base / "llama.cpp"))
    # Isolated sandbox (tests pass root=): keep everything under that tree.
    isolated = root is not None
    if isolated:
        data = base
    else:
        data = shared_data_home(env=env)
        migrate_shared_data(base, data, env=env)
    models_dir = Path(env.get("MODEL_DIR") or (data / "models"))
    log_dir = Path(env.get("LOG_DIR") or (data / "logs"))
    log_file = Path(env.get("LOG_FILE") or (log_dir / "llm-serve.log"))
    pid_file = Path(env.get("PID_FILE") or (log_dir / ".llm-serve.pid"))
    return AppPaths(
        root=base,
        data=data,
        models_json=data / "models.json",
        presets_json=data / "presets.json",
        models_dir=models_dir,
        llama_dir=llama_dir,
        llama_server=llama_dir / "build" / "bin" / "llama-server",
        llama_bench=llama_dir / "build" / "bin" / "llama-bench",
        log_dir=log_dir,
        log_file=log_file,
        pid_file=pid_file,
        settings_json=data / "tui-settings.json",
        baselines_json=data / "tui-baselines.json",
        benchmarks_json=data / "tui-benchmarks.json",
    )
