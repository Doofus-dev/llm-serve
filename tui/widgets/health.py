"""Health labels for VRAM, GPU temperature, and generation speed."""

from __future__ import annotations

from datetime import timedelta

from tui.data.context_length import fmt_ctx_compact, fmt_ctx_range, resolve_context_length
from pathlib import Path


def fmt_uptime(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def fmt_ctx(n) -> str:
    """Compact context size: 32768 → 32k."""
    return fmt_ctx_compact(n)


def model_author(params: dict, file: str) -> str | None:
    source = params.get("source")
    if isinstance(source, dict):
        author = source.get("author")
        if author:
            return str(author)
    if "/" in file:
        return file.split("/", 1)[0]
    return None


def model_file_exists(file: str, models_dir: Path) -> bool:
    if not file:
        return False
    return (models_dir / file).is_file()


def fmt_model_runtime_line(params: dict, models_dir: Path) -> str:
    """Line 2: preset context vs model max, and GPU layers."""
    max_ctx = resolve_context_length(params, models_dir)
    ctx = fmt_ctx_range(params.get("ctx", "?"), max_ctx)
    ngl = params.get("gpu_layers", "?")
    return f"ctx: {ctx}  ngl: {ngl}"

def vram_health(percent: float) -> tuple[str, str]:
    """Return a concise VRAM pressure label and Rich style."""
    if percent >= 90:
        return "CRITICAL", "bold red"
    if percent >= 75:
        return "HIGH", "bold yellow"
    return "OK", "bold green"


def temperature_health(temp_c: float) -> tuple[str, str]:
    """Return a conservative GPU temperature label and Rich style."""
    if temp_c >= 85:
        return "HOT", "bold red"
    if temp_c >= 70:
        return "WARM", "bold yellow"
    return "COOL", "bold green"


def generation_health(tokens_per_second: float) -> tuple[str, str]:
    """Return a broad generation-speed indicator."""
    if tokens_per_second >= 20:
        return "FAST", "bold green"
    if tokens_per_second >= 5:
        return "MODERATE", "bold yellow"
    if tokens_per_second > 0:
        return "SLOW", "bold red"
    return "IDLE", "dim"
