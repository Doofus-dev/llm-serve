"""Approximate VRAM requirements for local GGUF inference."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

from tui.data.gpu import GPUStats


@dataclass(frozen=True)
class VRAMEstimate:
    total_mb: float
    percent_available: float | None
    status: str


def estimate_vram_mb(file_size: int, context_tokens: int) -> float:
    """Estimate full-offload VRAM use.

    GGUF file size is the best pre-download signal available. Runtime
    overhead and KV cache vary by architecture, so this deliberately uses a
    conservative heuristic rather than presenting false precision.
    """
    weights_mb = file_size / 1_000_000
    runtime_overhead_mb = max(512.0, weights_mb * 0.08)
    # A rough Q4 KV-cache estimate: at 64K, cache is ~30% of weight size.
    kv_cache_mb = weights_mb * 0.30 * (context_tokens / 65_536)
    return (weights_mb + runtime_overhead_mb + kv_cache_mb) * 1.10


def classify_vram(estimated_mb: float, gpu: GPUStats) -> VRAMEstimate:
    """Classify an estimate against currently available memory."""
    available_mb = gpu.vram_total_mb - gpu.vram_used_mb
    if available_mb <= 0:
        return VRAMEstimate(estimated_mb, None, "unknown")

    percent = estimated_mb / available_mb * 100
    if percent <= 70:
        status = "comfortable"
    elif percent <= 90:
        status = "fits, tight"
    elif percent <= 100:
        status = "marginal"
    else:
        status = "too large"
    return VRAMEstimate(estimated_mb, percent, status)


def fmt_memory_mb(mb: float) -> str:
    if mb <= 0:
        return "?"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def status_symbol(status: str) -> Text:
    """Return a compact, terminal-safe colored fit indicator."""
    symbols = {
        "comfortable": ("●", "green"),
        "fits, tight": ("●", "yellow"),
        "marginal": ("⚠", "yellow"),
        "too large": ("●", "red"),
    }
    symbol, color = symbols.get(status, ("?", "dim"))
    return Text(symbol, style=color)
