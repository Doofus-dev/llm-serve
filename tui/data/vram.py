"""Approximate VRAM and decode-speed estimates for local GGUF inference."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rich.text import Text

from tui.data.baselines import RunBaseline, plausible_vram
from tui.data.gpu import GPUStats, same_gpu
from tui.data.quant import family_key, is_sidecar_gguf

# Effective system RAM bandwidth for CPU-offloaded layers. Dual-channel
# DDR5 is theoretically ~70-80 GB/s; llama.cpp typically sees less.
RAM_BANDWIDTH_GB_S = 55.0

# llama.cpp decode usually lands well below theoretical HBM/GDDR peaks.
GPU_BANDWIDTH_EFFICIENCY = 0.55

# CUDA graphs / compute buffers when we only have one measured context.
DEFAULT_RUNTIME_OVERHEAD_MB = 640.0
# Skip runs whose leftover after weights is just "file landed on the GPU".
MIN_RESIDUAL_MIB = 384.0
MIN_RESIDUAL_FILE_RATIO = 0.08

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


def _file_mib(file_size: int) -> float:
    """Bytes → MiB, matching nvidia-smi memory.used / memory.total."""
    return file_size / (1024 * 1024)


def _offload(run: RunBaseline) -> float:
    return max(0.0, min(1.0, run.offload_ratio))


def _usable_runs(
    runs: list[RunBaseline],
    gpu_name: str,
    offload_ratio: float,
) -> list[RunBaseline]:
    usable: list[RunBaseline] = []
    for run in runs:
        if gpu_name and run.gpu_name and not same_gpu(run.gpu_name, gpu_name):
            continue
        if abs(_offload(run) - offload_ratio) > 0.2:
            continue
        if is_sidecar_gguf(run.file):
            continue
        if run.file_size <= 0 or run.ctx <= 0:
            continue
        if not plausible_vram(run.vram_used_mb, run.file_size):
            continue
        usable.append(run)
    return usable


def _sibling_runs(runs: list[RunBaseline], filename: str) -> list[RunBaseline]:
    if is_sidecar_gguf(filename):
        return []
    want = family_key(filename)
    if not want:
        return []
    return [run for run in runs if family_key(run.file) == want]


@dataclass(frozen=True)
class MemoryFit:
    """VRAM ≈ file*offload + (overhead + kv_per_token*ctx) * offload."""

    overhead_mb: float
    kv_per_token: float


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _residual_mib(run: RunBaseline) -> float | None:
    """KV + runtime leftover, or None if the run looks like a partial load."""
    ratio = _offload(run)
    if ratio <= 0:
        return None
    weights = _file_mib(run.file_size) * ratio
    residual = run.vram_used_mb - weights
    if residual < MIN_RESIDUAL_MIB:
        return None
    if residual < weights * MIN_RESIDUAL_FILE_RATIO:
        return None
    return residual / ratio


def fit_memory(runs: list[RunBaseline]) -> MemoryFit | None:
    """Learn KV + overhead from residuals after subtracting file weights."""
    by_ctx: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        residual = _residual_mib(run)
        if residual is None:
            continue
        by_ctx[run.ctx].append(residual)
    points = [(float(ctx), _median(values)) for ctx, values in sorted(by_ctx.items())]
    if not points:
        return None

    if len(points) == 1:
        ctx, residual = points[0]
        overhead = min(DEFAULT_RUNTIME_OVERHEAD_MB, residual * 0.5)
        kv = max(0.0, (residual - overhead) / max(ctx, 1.0))
        return MemoryFit(overhead, kv)

    n = float(len(points))
    sum_x = sum(ctx for ctx, _ in points)
    sum_y = sum(residual for _, residual in points)
    sum_xx = sum(ctx * ctx for ctx, _ in points)
    sum_xy = sum(ctx * residual for ctx, residual in points)
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1.0:
        ctx = points[0][0]
        residual = sum_y / n
        overhead = min(DEFAULT_RUNTIME_OVERHEAD_MB, residual * 0.5)
        return MemoryFit(overhead, max(0.0, (residual - overhead) / max(ctx, 1.0)))
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    if slope < 0:
        intercept = sum_y / n
        slope = 0.0
    if intercept < 0:
        intercept = 0.0
    return MemoryFit(intercept, slope)


def apply_memory_fit(
    fit: MemoryFit,
    file_size: int,
    context_tokens: int,
    offload_ratio: float,
) -> float:
    ratio = max(0.0, min(1.0, offload_ratio))
    weights = _file_mib(file_size) * ratio
    return weights + (fit.overhead_mb + fit.kv_per_token * context_tokens) * ratio


def _naive_scales(runs: list[RunBaseline]) -> list[float]:
    scales: list[float] = []
    for run in runs:
        estimated = estimate_vram_mb(run.file_size, run.ctx, _offload(run))
        if estimated > 0:
            scales.append(run.vram_used_mb / estimated)
    return scales


def estimate_vram_calibrated(
    file_size: int,
    context_tokens: int,
    offload_ratio: float,
    *,
    runs: list[RunBaseline] | None = None,
    gpu_name: str = "",
    filename: str = "",
) -> float:
    """Estimate VRAM, scaled from on-machine runs when we have them.

    Sibling quants of the same model share KV cache and runtime overhead;
    only the weight blob changes. Other models on this GPU fall back to a
    median scale of the generic heuristic.
    """
    naive = estimate_vram_mb(file_size, context_tokens, offload_ratio)
    if not runs:
        return naive
    usable = _usable_runs(runs, gpu_name, offload_ratio)
    siblings = _sibling_runs(usable, filename)
    fit = fit_memory(siblings)
    if fit is not None:
        return apply_memory_fit(fit, file_size, context_tokens, offload_ratio)
    scales = _naive_scales(usable)
    if scales:
        return naive * _median(scales)
    return naive


def estimate_gen_tps_calibrated(
    file_size: int,
    context_tokens: int,
    gpu: GPUStats,
    offload_ratio: float = 1.0,
    *,
    runs: list[RunBaseline] | None = None,
    filename: str = "",
) -> float | None:
    """Estimate decode speed, scaled from measured tok/s on this GPU."""
    naive = estimate_gen_tps(file_size, context_tokens, gpu, offload_ratio)
    if naive is None or not runs:
        return naive
    usable = [
        run
        for run in _usable_runs(runs, gpu.name, offload_ratio)
        if run.gen_tps
    ]
    siblings = _sibling_runs(usable, filename) or usable
    scales: list[float] = []
    for run in siblings:
        estimated = estimate_gen_tps(run.file_size, run.ctx, gpu, _offload(run))
        if estimated and run.gen_tps:
            scales.append(run.gen_tps / estimated)
    if not scales:
        return naive
    return naive * _median(scales)


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
