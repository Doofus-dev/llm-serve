"""Health labels for VRAM, GPU temperature, and generation speed."""

from __future__ import annotations

from datetime import timedelta

from rich.text import Text
from tui.data.context_length import fmt_ctx_compact, fmt_ctx_range, resolve_context_length


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

# Health ramp: one green/amber/red scale shared by VRAM, temperature,
# and generation speed so "is it healthy at a glance" works across all three.
from tui.theme import ERR_STYLE, OK_STYLE, WARN_STYLE

HEALTH_OK = OK_STYLE
HEALTH_WARN = WARN_STYLE
HEALTH_ERR = ERR_STYLE


def vram_health(percent: float) -> tuple[str, str]:
    """Return a concise VRAM pressure label and Rich style."""
    if percent >= 90:
        return "CRITICAL", ERR_STYLE
    if percent >= 75:
        return "HIGH", WARN_STYLE
    return "OK", OK_STYLE


def temperature_health(temp_c: float) -> tuple[str, str]:
    """Return a conservative GPU temperature label and Rich style."""
    if temp_c >= 85:
        return "HOT", ERR_STYLE
    if temp_c >= 70:
        return "WARM", WARN_STYLE
    return "COOL", OK_STYLE


def generation_health(tokens_per_second: float) -> tuple[str, str]:
    """Return a broad generation-speed indicator."""
    if tokens_per_second >= 20:
        return "FAST", OK_STYLE
    if tokens_per_second >= 5:
        return "MODERATE", WARN_STYLE
    if tokens_per_second > 0:
        return "SLOW", ERR_STYLE
    return "IDLE", "dim"


def health_dot(style: str) -> Text:
    """Colored dot matching a health style: green/amber/red ramp, dim when idle."""
    if OK_STYLE in style:
        return Text("●", style="green")
    if WARN_STYLE in style:
        return Text("●", style="yellow")
    if ERR_STYLE in style:
        return Text("●", style="red")
    return Text("●", style="dim")

