"""Measured-metric lookup and comparison for the Compare Presets page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tui.data.baselines import RunBaseline, lookup_baseline
from tui.data.models_json import coerce_rel_paths, quant_entry_rel_paths
from tui.data.context_length import fmt_ctx_compact
from tui.data.hf import fmt_size
from tui.data.presets import Preset, list_presets_for_quant
from tui.data.preset_template import DEFAULT_PRESET_NAME
from tui.data.throughput_history import baseline_speed
from tui.theme import ACCENT, ERR, OK

_NO_DATA = "[dim]no data[/]"

_METRICS: tuple[tuple[str, str, bool, str], ...] = (
    ("VRAM used (MB)", "vram_used_mb", False, "mb"),
    ("gen t/s", "gen_tps", True, "tps"),
    ("prompt t/s", "prompt_tps", True, "tps"),
    ("file size", "file_size", False, "bytes"),
    ("context", "ctx", True, "ctx"),
)


def resolve_default_preset(store, model: str, quant: str) -> Preset | None:
    """Return the seeded default preset for a model/quant, if one exists."""
    slots = list_presets_for_quant(store, model, quant)
    for preset in slots.values():
        if preset.name == DEFAULT_PRESET_NAME:
            return preset
    return slots.get(1)


def _filename(path: str) -> str:
    return Path(path).name.lower()


def _file_rels_for_quant(identity: dict[str, Any], quant: str) -> list[str]:
    quants = identity.get("quants")
    if isinstance(quants, dict):
        paths = quant_entry_rel_paths(quants.get(quant) or {})
        if paths:
            return paths
    return coerce_rel_paths(identity.get("files")) or coerce_rel_paths(identity.get("file"))


def _file_rel_for_quant(identity: dict[str, Any], quant: str) -> str:
    rels = _file_rels_for_quant(identity, quant)
    return rels[0] if rels else ""


def _disk_size(models_dir: Path | None, file_rel: str | list[str]) -> int:
    rels = [file_rel] if isinstance(file_rel, str) else list(file_rel)
    if not models_dir or not rels:
        return 0
    total = 0
    for rel in rels:
        if not rel:
            continue
        path = models_dir / rel
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def resolve_baseline(
    model_key: str,
    quant: str,
    preset_slot: int,
    models_dir: Path | None,
    baselines: list[RunBaseline],
    *,
    identity: dict[str, Any] | None = None,
    preset: Preset | None = None,
    gpu_name: str = "",
    live: Any | None = None,
    history: list[float] | None = None,
) -> RunBaseline | None:
    """Look up the measured baseline for a preset, with live-speed fallback.

    ``preset_slot`` is part of the public identity of the lookup; stored
    baselines do not record the slot, so matching uses model / file / ctx
    (and gpu layers when present). Live ``baseline_speed`` fills gen/prompt
    t/s when no stored speed exists.
    """
    del preset_slot  # slot distinguishes callers; runs are keyed by config
    identity = identity or {}
    file_rels = _file_rels_for_quant(identity, quant)
    file_rel = file_rels[0] if file_rels else ""
    filename = Path(file_rel).name if file_rel else ""
    file_size = _disk_size(models_dir, file_rels)

    ctx = 0
    gpu_layers = 0
    cache_k = ""
    cache_v = ""
    if preset is not None:
        ctx = _as_int(preset.params.get("ctx"), 0)
        gpu_layers = _as_int(preset.params.get("gpu_layers"), 0)
        cache_k = str(preset.params.get("cache_k") or "")
        cache_v = str(preset.params.get("cache_v") or "")
    total_layers = _as_int(identity.get("total_layers"), 0)

    offload_ratio = 1.0
    if total_layers > 0 and 0 < gpu_layers < 99:
        offload_ratio = max(0.0, min(1.0, gpu_layers / total_layers))
    elif gpu_layers == 0:
        offload_ratio = 0.0

    stored: RunBaseline | None = None
    want_name = _filename(filename) if filename else ""
    for run in baselines:
        if model_key and run.model and run.model != model_key:
            continue
        if want_name and _filename(run.file) != want_name:
            continue
        if ctx and run.ctx and run.ctx != ctx:
            continue
        if gpu_layers and run.gpu_layers and gpu_layers not in {99, run.gpu_layers} and run.gpu_layers not in {99, gpu_layers}:
            continue
        stored = run
        break

    if stored is None and (filename or file_size):
        stored = lookup_baseline(
            [run for run in baselines if not model_key or not run.model or run.model == model_key],
            filename=filename,
            file_size=file_size or 1,
            ctx=ctx or 1,
            offload_ratio=offload_ratio,
            gpu_name=gpu_name,
        )
        # Fuzzy lookup can return a sibling preset's measurement (different
        # ctx). Only keep it when ctx is unknown or matches.
        if stored is not None and ctx and stored.ctx and stored.ctx != ctx:
            stored = None

    gen_tps = stored.gen_tps if stored else None
    prompt_tps = stored.prompt_tps if stored else None
    if gen_tps is None or prompt_tps is None:
        live_gen, live_prompt, _tokens = baseline_speed(live, history)
        if gen_tps is None:
            gen_tps = live_gen
        if prompt_tps is None:
            prompt_tps = live_prompt

    vram = stored.vram_used_mb if stored and stored.vram_used_mb else 0.0
    size_out = (stored.file_size if stored and stored.file_size else 0) or file_size
    ctx_out = (stored.ctx if stored and stored.ctx else 0) or ctx

    if (
        stored is None
        and not size_out
        and not ctx_out
        and gen_tps is None
        and prompt_tps is None
        and not vram
    ):
        return None

    return RunBaseline(
        model=model_key,
        file=file_rel or (stored.file if stored else ""),
        file_size=size_out,
        gpu_name=gpu_name or (stored.gpu_name if stored else ""),
        ctx=ctx_out,
        gpu_layers=gpu_layers or (stored.gpu_layers if stored else 0),
        total_layers=total_layers or (stored.total_layers if stored else 0),
        cache_k=cache_k or (stored.cache_k if stored else ""),
        cache_v=cache_v or (stored.cache_v if stored else ""),
        vram_used_mb=vram,
        gen_tps=gen_tps,
        prompt_tps=prompt_tps,
        tokens_predicted=stored.tokens_predicted if stored else 0.0,
        updated_at=stored.updated_at if stored else "",
    )


def _metric_value(run: RunBaseline | None, key: str) -> float | int | None:
    if run is None:
        return None
    value = getattr(run, key, None)
    if value is None:
        return None
    if key in {"vram_used_mb", "file_size", "ctx"} and not value:
        return None
    if key in {"gen_tps", "prompt_tps"} and float(value) <= 0:
        return None
    return value


def _winner(
    left: float | int | None,
    right: float | int | None,
    *,
    higher_is_better: bool,
) -> str | None:
    if left is None or right is None:
        return None
    if left == right:
        return "equal"
    left_better = left > right if higher_is_better else left < right
    return "left" if left_better else "right"


def compare_presets_metrics(
    left: RunBaseline | None,
    right: RunBaseline | None,
) -> list[dict]:
    """Return metric rows with winner ``left`` / ``right`` / ``equal`` / None."""
    rows: list[dict] = []
    for name, key, higher_is_better, fmt in _METRICS:
        left_value = _metric_value(left, key)
        right_value = _metric_value(right, key)
        rows.append(
            {
                "name": name,
                "left_value": left_value,
                "right_value": right_value,
                "winner": _winner(
                    left_value, right_value, higher_is_better=higher_is_better
                ),
                "format": fmt,
            }
        )
    return rows


def format_metric_value(value: float | int | None, fmt: str) -> str:
    if value is None:
        return _NO_DATA
    if fmt == "mb":
        return f"{float(value):.0f} MB"
    if fmt == "tps":
        number = float(value)
        if number >= 10:
            return f"{number:.0f} t/s"
        if number >= 1:
            return f"{number:.1f} t/s"
        return f"{number:.2f} t/s"
    if fmt == "bytes":
        return fmt_size(int(value))
    if fmt == "ctx":
        return str(fmt_ctx_compact(int(value)))
    return str(value)


def _plain_metric_text(value: float | int | None, fmt: str) -> str:
    if value is None:
        return "no data"
    return format_metric_value(value, fmt)


def _escape_markup(text: str) -> str:
    """Escape brackets so labels like ``[2] max-ctx`` are not Rich tags."""
    return text.replace("[", r"\[").replace("]", r"\]")


def format_metric_cell(
    value: float | int | None,
    *,
    winner: str | None,
    side: str,
    fmt: str,
    width: int | None = None,
) -> str:
    text = _plain_metric_text(value, fmt)
    if value is None:
        cell = f"[dim]{text}[/]"
    elif winner in (None, "equal"):
        cell = text
    else:
        color = OK if winner == side else ERR
        cell = f"[{color}]{text}[/]"
    if width is not None and width > len(text):
        cell = f"{cell}{' ' * (width - len(text))}"
    return cell


def format_metrics_table(
    rows: list[dict],
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> str:
    """Rich markup for the measured-metrics comparison table.

    Pad plain text to column width, then wrap markup so color tags cannot
    shift the columns.
    """
    name_w = max((len(str(row["name"])) for row in rows), default=8)
    name_w = max(name_w, len("Metric"))
    left_w = max(
        [len(left_label)]
        + [
            len(_plain_metric_text(row["left_value"], row["format"]))
            for row in rows
        ],
        default=1,
    )
    right_w = max(
        [len(right_label)]
        + [
            len(_plain_metric_text(row["right_value"], row["format"]))
            for row in rows
        ],
        default=1,
    )
    left_header = f"{left_label:<{left_w}}"
    name_header = f"{'Metric':<{name_w}}"
    header = (
        f"[bold {ACCENT}]{_escape_markup(name_header)}[/]  "
        f"[bold {ACCENT}]{_escape_markup(left_header)}[/]  "
        f"[bold {ACCENT}]{_escape_markup(right_label)}[/]"
    )
    lines = [header]
    for row in rows:
        name = f"{row['name']:<{name_w}}"
        left_cell = format_metric_cell(
            row["left_value"],
            winner=row["winner"],
            side="left",
            fmt=row["format"],
            width=left_w,
        )
        right_cell = format_metric_cell(
            row["right_value"],
            winner=row["winner"],
            side="right",
            fmt=row["format"],
            width=right_w,
        )
        lines.append(
            f"[{ACCENT}]{_escape_markup(name)}[/]  {left_cell}  {right_cell}"
        )
    return "\n".join(lines)
