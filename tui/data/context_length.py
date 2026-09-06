"""Model max context length — HF/GGUF storage, display, and preset caps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tui.data.gguf import read_gguf_architecture

CONTEXT_LENGTH_STOPS = (32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576)
DEFAULT_CONTEXT_FALLBACK = 65_536


def parse_context_length(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return int(value)


def resolve_context_length(identity: dict[str, Any], models_dir: Path | None = None) -> int | None:
    """Return stored model max context, falling back to the active GGUF header."""
    stored = parse_context_length(identity.get("context_length"))
    if stored is not None:
        return stored
    file_rel = str(identity.get("file", ""))
    if not file_rel or models_dir is None:
        return None
    return read_gguf_architecture(models_dir / file_rel).context_length


def apply_context_length(
    params: dict[str, Any],
    *,
    hf_context: int | None = None,
    gguf_path: Path | None = None,
    replace_cloned: bool = False,
) -> int | None:
    """Persist context_length from HF catalog and/or local GGUF metadata."""
    resolved: int | None = parse_context_length(hf_context)
    if gguf_path is not None:
        gguf_ctx = read_gguf_architecture(gguf_path).context_length
        if gguf_ctx is not None:
            resolved = gguf_ctx
    if resolved is not None:
        params["context_length"] = resolved
    elif replace_cloned:
        params.pop("context_length", None)
    return resolved


def context_length_options(model_max: int | None) -> list[int]:
    """Preset VRAM-estimate stops capped at the model's native context."""
    maximum = model_max or DEFAULT_CONTEXT_FALLBACK
    options = [value for value in CONTEXT_LENGTH_STOPS if value <= maximum]
    if maximum >= 32_768 and maximum not in options:
        options.append(maximum)
    if not options:
        options = [maximum]
    return options


def nearest_context_stop(ctx: int, stops: tuple[int, ...] | None = None) -> int:
    """Snap a runtime ctx (e.g. 65000) onto the nearest estimate stop."""
    if ctx <= 0:
        return 0
    options = stops if stops is not None else CONTEXT_LENGTH_STOPS
    return min(options, key=lambda stop: (abs(stop - ctx), stop))


def default_estimate_context(options: list[int]) -> int:
    """Hub / quant-picker default: 64k, or the model's max if smaller."""
    if not options:
        return DEFAULT_CONTEXT_FALLBACK
    return min(DEFAULT_CONTEXT_FALLBACK, options[-1])


def include_context_option(options: list[int], ctx: int) -> list[int]:
    """Add a runtime/preset ctx to the estimate dropdown when missing."""
    if ctx <= 0 or ctx in options:
        return options
    return sorted(set(options + [ctx]))


def pick_estimate_context(options: list[int], preferred: int | None = None) -> int:
    """Prefer the model's current/measured ctx, else the 64k default."""
    if preferred and preferred > 0:
        if preferred in options:
            return preferred
        if options:
            return min(options, key=lambda stop: (abs(stop - preferred), stop))
        return preferred
    return default_estimate_context(options)


def hub_min_context_options() -> list[tuple[str, int]]:
    """Hub browse choices: native context at least this large."""
    return [(f"{fmt_ctx_compact(value)}+", value) for value in CONTEXT_LENGTH_STOPS]


def cap_context_value(ctx: int, max_ctx: int | None) -> int:
    if max_ctx is None or max_ctx <= 0:
        return ctx
    return min(ctx, max_ctx)


def fmt_ctx_compact(value: object) -> str:
    """Compact context size: 32768 → 32k."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def clamp_preset_params(params: dict[str, Any], max_ctx: int | None) -> dict[str, Any]:
    """Return a copy with ctx capped to the model max when present."""
    clamped = dict(params)
    if max_ctx is None:
        return clamped
    ctx = clamped.get("ctx")
    if ctx is None or ctx == "":
        return clamped
    try:
        clamped["ctx"] = cap_context_value(int(ctx), max_ctx)
    except (TypeError, ValueError):
        pass
    return clamped


def fmt_ctx_range(preset_ctx: object, max_ctx: int | None) -> str:
    ctx_text = fmt_ctx_compact(preset_ctx)
    if max_ctx is None:
        return ctx_text
    return f"{ctx_text} / max {fmt_ctx_compact(max_ctx)}"
