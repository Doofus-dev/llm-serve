"""Shared quant/file table rows for Hub and the quant picker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text

from tui.data.baselines import load_baselines, lookup_baseline
from tui.data.gpu import GPUStats
from tui.data.hf import HubFile, fmt_size
from tui.data.capabilities import ModelCapabilities, addon_row_label
from tui.data.quant import (
    ROLE_MODEL,
    gguf_role,
    is_hidden_sidecar,
    is_runtime_addon,
    quant_from_filename,
)
from tui.data.vram import (
    classify_vram,
    estimate_gen_tps_calibrated,
    estimate_vram_calibrated,
    fmt_memory_mb,
    fmt_tps,
    status_symbol,
)
from tui.theme import OK, WARN


@dataclass(frozen=True)
class QuantFileRow:
    path: str
    quant_id: str
    size: int
    downloaded: bool
    vram_cell: Text
    act_vram_off: str
    act_vram_on: str
    est_tps: str
    act_tps: str
    is_draft: bool = False
    addon_kind: str | None = None
    addon_label: str = ""


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
    caps: ModelCapabilities | None = None,
) -> list[QuantFileRow]:
    """Build quant rows sorted by file size, with VRAM/speed estimates.

    Runtime add-ons (mmproj, MTP) are labeled rows, not mystery quants.
    Imatrix calibration files stay hidden.
    """
    runs = load_baselines(baselines_path) if baselines_path else []
    rows: list[QuantFileRow] = []

    for item in files:
        if is_hidden_sidecar(item.path):
            continue
        downloaded = False
        size = item.size
        if models_dir and author:
            local_size = local_file_size(models_dir, author, item.path)
            downloaded = local_size > 0
            if downloaded and size <= 0:
                size = local_size

        role = gguf_role(item.path)
        addon_kind = role if role != ROLE_MODEL else None
        addon_label = addon_row_label(role, caps) if addon_kind else ""
        if is_runtime_addon(item.path):
            dash = Text("—")
            rows.append(
                QuantFileRow(
                    path=item.path,
                    quant_id=quant_from_filename(item.path),
                    size=size,
                    downloaded=downloaded,
                    vram_cell=dash,
                    act_vram_off="—",
                    act_vram_on="—",
                    est_tps="—",
                    act_tps="—",
                    is_draft=role == "mtp",
                    addon_kind=addon_kind,
                    addon_label=addon_label,
                )
            )
            continue

        estimate = classify_vram(
            estimate_vram_calibrated(
                size,
                context_tokens,
                offload_ratio,
                runs=runs,
                gpu_name=gpu.name,
                filename=item.path,
            ),
            gpu,
        )
        tps = estimate_gen_tps_calibrated(
            size,
            context_tokens,
            gpu,
            offload_ratio,
            runs=runs,
            filename=item.path,
        )
        lookup_kw = dict(
            filename=item.path,
            file_size=size,
            ctx=context_tokens,
            offload_ratio=offload_ratio,
            gpu_name=gpu.name,
        )
        actual = lookup_baseline(runs, **lookup_kw)
        actual_off = lookup_baseline(runs, vision_enabled=False, **lookup_kw)
        actual_on = lookup_baseline(runs, vision_enabled=True, **lookup_kw)
        if estimate.percent_available is None:
            fit = "?"
        else:
            fit = f"{estimate.percent_available:.0f}%"
        vram_cell = Text(f"{fmt_memory_mb(estimate.total_mb)} · {fit} ")
        vram_cell.append_text(status_symbol(estimate.status))
        act_vram_off = (
            fmt_memory_mb(actual_off.vram_used_mb)
            if actual_off and actual_off.vram_used_mb > 0
            else "—"
        )
        act_vram_on = (
            fmt_memory_mb(actual_on.vram_used_mb)
            if actual_on and actual_on.vram_used_mb > 0
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
                act_vram_off=act_vram_off,
                act_vram_on=act_vram_on,
                est_tps=fmt_tps(tps),
                act_tps=act_tps,
                is_draft=False,
                addon_kind=None,
                addon_label="",
            )
        )

    rows.sort(key=lambda row: row.size, reverse=True)
    return rows


def fmt_downloaded(downloaded: bool) -> str:
    return "●" if downloaded else "—"


def fmt_downloaded_cell(downloaded: bool) -> str:
    mark = fmt_downloaded(downloaded)
    return f"[{OK}]{mark}[/]" if downloaded else mark


def fmt_disk_cell(*, downloaded: bool, transferring: bool = False) -> str:
    """Disk column: on-disk, in-flight, or missing."""
    if downloaded:
        return fmt_downloaded_cell(True)
    if transferring:
        return f"[{WARN}]⬇[/]"
    return fmt_downloaded_cell(False)


def quant_file_row_cells(row: QuantFileRow) -> tuple:
    """Cell values for the shared Disk + estimate file table."""
    if row.addon_kind:
        tag = row.addon_label or row.addon_kind
        label = f"[bold]{tag}[/] · {row.path}"
    else:
        label = row.path
    return (
        fmt_downloaded_cell(row.downloaded),
        label,
        fmt_size(row.size),
        row.vram_cell,
        row.act_vram_off,
        row.act_vram_on,
        row.est_tps,
        row.act_tps,
    )


__all__ = [
    "QuantFileRow",
    "build_quant_file_rows",
    "fmt_downloaded",
    "fmt_downloaded_cell",
    "fmt_disk_cell",
    "fmt_size",
    "local_file_size",
    "quant_file_row_cells",
]
