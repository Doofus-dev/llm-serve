"""Shared quant/file table rows for Hub and the quant picker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from tui.data.baselines import load_baselines, lookup_baseline
from tui.data.gpu import GPUStats
from tui.data.hf import HubFile, fmt_size
from tui.data.quant import quant_from_filename
from tui.data.vram import (
    classify_vram,
    estimate_gen_tps,
    estimate_vram_mb,
    fmt_memory_mb,
    fmt_tps,
    status_symbol,
)


@dataclass(frozen=True)
class QuantFileRow:
    path: str
    quant_id: str
    size: int
    downloaded: bool
    vram_cell: Text
    act_vram: str
    est_tps: str
    act_tps: str


def local_file_size(models_dir: Path, author: str, filename: str) -> int:
    path = models_dir / author / filename
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def build_quant_file_rows(
    files: list[HubFile],
    *,
    gpu: GPUStats,
    context_tokens: int,
    offload_ratio: float,
    baselines_path: Path | None,
    models_dir: Path | None = None,
    author: str = "",
) -> list[QuantFileRow]:
    """Build quant rows sorted by file size, with VRAM/speed estimates."""
    runs = load_baselines(baselines_path) if baselines_path else []
    rows: list[QuantFileRow] = []

    for item in files:
        downloaded = False
        size = item.size
        if models_dir and author:
            local_size = local_file_size(models_dir, author, item.path)
            downloaded = local_size > 0
            if downloaded and size <= 0:
                size = local_size

        estimate = classify_vram(
            estimate_vram_mb(size, context_tokens, offload_ratio),
            gpu,
        )
        tps = estimate_gen_tps(size, context_tokens, gpu, offload_ratio)
        actual = lookup_baseline(
            runs,
            filename=item.path,
            file_size=size,
            ctx=context_tokens,
            offload_ratio=offload_ratio,
            gpu_name=gpu.name,
        )
        if estimate.percent_available is None:
            fit = "?"
        else:
            fit = f"{estimate.percent_available:.0f}%"
        vram_cell = Text(f"{fmt_memory_mb(estimate.total_mb)} · {fit} ")
        vram_cell.append_text(status_symbol(estimate.status))
        act_vram = (
            fmt_memory_mb(actual.vram_used_mb)
            if actual and actual.vram_used_mb > 0
            else "—"
        )
        act_tps = fmt_tps(actual.gen_tps if actual else None, estimated=False)
        rows.append(
            QuantFileRow(
                path=item.path,
                quant_id=quant_from_filename(item.path),
                size=size,
                downloaded=downloaded,
                vram_cell=vram_cell,
                act_vram=act_vram,
                est_tps=fmt_tps(tps),
                act_tps=act_tps,
            )
        )

    rows.sort(key=lambda row: row.size, reverse=True)
    return rows


def fmt_downloaded(downloaded: bool) -> str:
    return "●" if downloaded else "—"


__all__ = [
    "QuantFileRow",
    "build_quant_file_rows",
    "fmt_downloaded",
    "fmt_size",
    "local_file_size",
]
