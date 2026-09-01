"""Approximate VRAM and decode-speed estimates for local GGUF inference."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

from tui.data.gpu import GPUStats

# Effective system RAM bandwidth for CPU-offloaded layers. Dual-channel
# DDR5 is theoretically ~70-80 GB/s; llama.cpp typically sees less.
RAM_BANDWIDTH_GB_S = 55.0

# llama.cpp decode usually lands well below theoretical HBM/GDDR peaks.
GPU_BANDWIDTH_EFFICIENCY = 0.55

# Longest needles first so "7900 xtx" wins over "7900 xt".
_GPU_BANDWIDTH_GB_S: tuple[tuple[str, float], ...] = tuple(
    sorted(
        (
            ("b200", 8000.0),
            ("h200", 4800.0),
            ("h100", 3000.0),
            ("a100 80", 2039.0),
            ("a100", 1555.0),
            ("a6000", 768.0),
            ("l40s", 864.0),
            ("rtx 6000 ada", 960.0),
            ("rtx 5090", 1792.0),
            ("rtx 5080", 960.0),
            ("rtx 5070 ti", 672.0),
            ("rtx 5070", 672.0),
            ("rtx 4090", 1008.0),
            ("rtx 4080 super", 736.0),
            ("rtx 4080", 717.0),
            ("rtx 4070 ti super", 672.0),
            ("rtx 4070 ti", 504.0),
            ("rtx 4070 super", 504.0),
            ("rtx 4070", 504.0),
            ("rtx 4060 ti", 288.0),
            ("rtx 4060", 272.0),
            ("rtx 3090 ti", 1008.0),
            ("rtx 3090", 936.0),
            ("rtx 3080 ti", 912.0),
            ("rtx 3080", 760.0),
            ("rtx 3070", 448.0),
            ("rtx 3060", 360.0),
            ("7900 xtx", 960.0),
            ("7900 xt", 800.0),
            ("7900 gre", 576.0),
            ("7800 xt", 624.0),
            ("7700 xt", 432.0),
            ("7600", 288.0),
            ("9070 xt", 640.0),
            ("9070", 640.0),
            ("6950 xt", 576.0),
            ("6900 xt", 512.0),
            ("6800 xt", 512.0),
            ("mi300x", 5300.0),
            ("mi250", 3277.0),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


@dataclass(frozen=True)
class VRAMEstimate:
    total_mb: float
    percent_available: float | None
    status: str


def estimate_vram_mb(
    file_size: int,
    context_tokens: int,
    offload_ratio: float = 1.0,
) -> float:
    """Estimate VRAM use for a given GPU-offload fraction.

    GGUF file size is the best pre-download signal available. Runtime
    overhead and KV cache vary by architecture, so this deliberately uses a
    conservative heuristic rather than presenting false precision.

    ``offload_ratio`` is 1.0 for every layer on GPU and 0.0 for CPU-only.
    Weights and KV for CPU layers are treated as staying in system RAM.
    """
    ratio = max(0.0, min(1.0, offload_ratio))
    if ratio <= 0:
        return 0.0
    weights_mb = file_size / 1_000_000
    gpu_weights_mb = weights_mb * ratio
    runtime_overhead_mb = max(512.0, gpu_weights_mb * 0.08)
    # A rough Q4 KV-cache estimate: at 64K, cache is ~30% of weight size.
    kv_cache_mb = weights_mb * 0.30 * (context_tokens / 65_536) * ratio
    return (gpu_weights_mb + runtime_overhead_mb + kv_cache_mb) * 1.10


def gpu_bandwidth_gb_s(gpu: GPUStats) -> float:
    """Best-guess device memory bandwidth in GB/s."""
    name = gpu.name.lower()
    if "cpu-only" in name or "system ram" in name:
        return RAM_BANDWIDTH_GB_S
    for needle, bandwidth in _GPU_BANDWIDTH_GB_S:
        if needle in name:
            return bandwidth
    vram_gb = gpu.vram_total_mb / 1024
    if vram_gb >= 40:
        return 1200.0
    if vram_gb >= 20:
        return 800.0
    if vram_gb >= 12:
        return 500.0
    if vram_gb >= 8:
        return 360.0
    if vram_gb > 0:
        return 300.0
    return RAM_BANDWIDTH_GB_S


def estimate_gen_tps(
    file_size: int,
    context_tokens: int,
    gpu: GPUStats,
    offload_ratio: float = 1.0,
) -> float | None:
    """Estimate decode (generation) tokens/sec.

    Decode is mostly memory-bandwidth bound: each token re-reads the
    resident weights, then the growing KV cache. GPU and CPU layers are
    sequential, so even a small CPU fraction dominates once RAM bandwidth
    is the bottleneck.

    This is the same class of guess as the VRAM column — useful for
    comparing quants and offload levels, not a promise.
    """
    weights_gb = file_size / 1_000_000_000
    if weights_gb <= 0:
        return None

    ratio = max(0.0, min(1.0, offload_ratio))
    name = gpu.name.lower()
    if "cpu-only" in name or "system ram" in name:
        ratio = 0.0

    gpu_bw = max(gpu_bandwidth_gb_s(gpu) * GPU_BANDWIDTH_EFFICIENCY, 1.0)
    kv_gb = weights_gb * 0.30 * (context_tokens / 65_536)
    gpu_gb = weights_gb * ratio
    cpu_gb = weights_gb * (1.0 - ratio)
    seconds = 0.0
    if gpu_gb > 0:
        seconds += (gpu_gb + kv_gb * ratio) / gpu_bw
    if cpu_gb > 0:
        seconds += (cpu_gb + kv_gb * (1.0 - ratio)) / RAM_BANDWIDTH_GB_S
    if seconds <= 0:
        return None
    return 1.0 / seconds


def classify_vram(estimated_mb: float, gpu: GPUStats) -> VRAMEstimate:
    """Classify an estimate against this GPU's memory pool.

    Uses total GPU-accessible memory, not whatever is free this second.
    On a discrete card that is leftover VRAM after other apps; on an APU
    it is the GTT/unified pool. Comparing to leftover while a model is
    already loaded makes a running 10 GB model look like 200% of the
    remaining 5 GB.
    """
    budget_mb = gpu.vram_total_mb
    if budget_mb <= 0:
        return VRAMEstimate(estimated_mb, None, "unknown")

    percent = estimated_mb / budget_mb * 100
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
    if mb < 0:
        return "?"
    if mb == 0:
        return "0 MB"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def fmt_tps(tps: float | None, *, estimated: bool = True) -> str:
    if tps is None or tps <= 0:
        return "?" if estimated else "—"
    prefix = "~" if estimated else ""
    if tps >= 10:
        return f"{prefix}{tps:.0f} t/s"
    if tps >= 1:
        return f"{prefix}{tps:.1f} t/s"
    return f"{prefix}{tps:.2f} t/s"


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
