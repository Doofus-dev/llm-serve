"""Active preset summary for the selected model."""

from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static
from textual.app import ComposeResult

from tui.data.context_length import resolve_context_length
from tui.data.models_json import Registry, coerce_rel_paths, quant_entry_rel_paths, quant_rel_paths
from tui.data.presets import (
    Preset,
    PresetStore,
    get_active_slot,
    get_preset,
    merge_identity_and_preset,
)
from tui.paths import PROFILE_KEYS
from tui.theme import ACCENT, ERR, OK, WARN, WARN_STYLE
from tui.widgets.health import fmt_ctx, model_author, model_file_exists


def _grid_label(label: str) -> Text:
    return Text(label, style=ACCENT)


def _grid_value(value: object) -> Text:
    if value == "":
        return Text("—", style="dim")
    return Text(str(value), style=WARN_STYLE)


def _markup_label(label: str) -> str:
    return f"[{ACCENT}]{escape(label)}[/]"


def _markup_value(value: object) -> str:
    if value == "":
        return "[dim]—[/]"
    return f"[{WARN_STYLE}]{escape(str(value))}[/]"


def _identity_fields(
    identity: dict, preset: Preset, models_dir: Path | None
) -> dict[str, object]:
    file_rels = coerce_rel_paths(identity.get("file"))
    quants = identity.get("quants")
    active = str(identity.get("active_quant") or "")
    if isinstance(quants, dict) and active:
        from_quant = quant_entry_rel_paths(quants.get(active))
        if from_quant:
            file_rels = from_quant
    file_rel = file_rels[0] if file_rels else ""
    author = model_author(identity, file_rel)
    max_ctx = resolve_context_length(identity, models_dir)
    file_status = (
        "present"
        if models_dir and model_file_exists(file_rels, models_dir)
        else "missing"
    )
    return {
        "Model": str(identity.get("display") or "—"),
        "Quant": str(identity.get("active_quant") or "—"),
        "Preset": f"[{preset.slot}] {preset.name}",
        "Author": author or "local",
        "Max context": (
            f"{fmt_ctx(max_ctx)} tokens" if max_ctx is not None else "unknown"
        ),
        "File": file_status,
    }


def _runtime_settings(identity: dict, preset: Preset) -> list[tuple[str, object]]:
    params = merge_identity_and_preset(identity, preset)
    settings: list[tuple[str, object]] = []
    for key, value in params.items():
        if key in {"source", "quants", "file", "files"}:
            continue
        if key in PROFILE_KEYS:
            continue
        settings.append((key, value))
    return settings


def _paired_markup(rows: list[tuple[str, object]]) -> list[str]:
    lines: list[str] = []
    for index in range(0, len(rows), 2):
        left_label, left_value = rows[index]
        left = f"{_markup_label(left_label)}  {_markup_value(left_value)}"
        if index + 1 < len(rows):
            right_label, right_value = rows[index + 1]
            right = f"{_markup_label(right_label)}  {_markup_value(right_value)}"
            lines.append(f"{left}    {right}")
        else:
            lines.append(left)
    return lines


def _plain_cell(value: object) -> str:
    return "" if value == "" else str(value)


def _cell_width(value: object) -> int:
    text = _plain_cell(value)
    return len(text) if text else 1


def _markup_pair(
    label: str,
    left_value: object,
    right_value: object,
    label_w: int,
    left_w: int,
    right_w: int | None = None,
    *,
    color_diff: bool = False,
) -> str:
    """One aligned row: label, left value, right value with computed column widths."""
    left_str = _plain_cell(left_value)
    right_str = _plain_cell(right_value)
    label_padded = f"{label:<{label_w}}"
    left_padded = f"{left_str:<{left_w}}" if left_str else f"{'—':<{left_w}}"
    if right_w is None:
        right_padded = right_str if right_str else "—"
    else:
        right_padded = f"{right_str:<{right_w}}" if right_str else f"{'—':<{right_w}}"
    label_part = f"[{ACCENT}]{escape(label_padded)}[/]"
    left_style = WARN_STYLE
    right_style = WARN_STYLE
    if color_diff and left_str and right_str and left_str != right_str:
        from tui.screens.compare import _direction_style

        right_style = _direction_style(left_str, right_str)
        left_style = (
            WARN
            if right_style == WARN
            else (ERR if right_style == OK else OK)
        )
    left_part = f"[{left_style}]{escape(left_padded)}[/]" if left_str else f"[dim]{'—':<{left_w}}[/]"
    if right_str:
        right_part = f"[{right_style}]{escape(right_padded)}[/]"
    else:
        pad = right_w if right_w is not None else 1
        right_part = f"[dim]{'—':<{pad}}[/]"
    return f"{label_part}  {left_part}  {right_part}"


def preset_compare_rows(
    left: tuple[dict, Preset],
    right: tuple[dict, Preset],
    models_dir: Path | None,
) -> tuple[list[tuple[str, object, object]], list[tuple[str, object, object]]]:
    """Identity rows then runtime rows as (label, left, right)."""
    left_fields = _identity_fields(left[0], left[1], models_dir)
    right_fields = _identity_fields(right[0], right[1], models_dir)
    field_names = ("Model", "Quant", "Preset", "Author", "Max context", "File")
    identity_rows = [(name, left_fields[name], right_fields[name]) for name in field_names]
    left_settings = _runtime_settings(left[0], left[1])
    right_settings = _runtime_settings(right[0], right[1])
    rows: dict[str, tuple[object, object]] = {key: (value, "") for key, value in left_settings}
    for key, value in right_settings:
        if key in rows:
            rows[key] = (rows[key][0], value)
        else:
            rows[key] = ("", value)
    runtime_rows = [(key, lv, rv) for key, (lv, rv) in rows.items()]
    return identity_rows, runtime_rows


def compare_column_widths(
    *row_groups: list[tuple[str, object, object]],
    extra_labels: tuple[str, ...] = (),
    extra_left: tuple[str, ...] = (),
    extra_right: tuple[str, ...] = (),
) -> tuple[int, int, int]:
    """One label/left/right width for stacked compare sections."""
    labels = list(extra_labels)
    lefts: list[object] = list(extra_left)
    rights: list[object] = list(extra_right)
    for group in row_groups:
        for name, lv, rv in group:
            labels.append(name)
            lefts.append(lv)
            rights.append(rv)
    label_w = max((len(str(name)) for name in labels), default=1)
    left_w = max((_cell_width(value) for value in lefts), default=1)
    right_w = max((_cell_width(value) for value in rights), default=1)
    left_w = max(left_w, 1)
    right_w = max(right_w, 1)
    return label_w, left_w, right_w


def build_preset_column(
    left: tuple[dict, Preset],
    right: tuple[dict, Preset],
    models_dir: Path | None,
    *,
    widths: tuple[int, int, int] | None = None,
) -> str:
    """Side-by-side rich markup: each row shows label, left value, right value.

    Identity (model/quant/preset) and runtime use the same column widths so
    the top block lines up with the columns below.
    """
    identity_rows, runtime_rows = preset_compare_rows(left, right, models_dir)
    if widths is None:
        label_w, left_w, right_w = compare_column_widths(identity_rows, runtime_rows)
    else:
        label_w, left_w, right_w = widths

    lines: list[str] = []
    for name, lv, rv in identity_rows:
        lines.append(_markup_pair(name, lv, rv, label_w, left_w, right_w))
    lines.append("")
    lines.append("[bold dim]RUNTIME CONFIG[/]")
    for key, lv, rv in runtime_rows:
        lines.append(
            _markup_pair(key, lv, rv, label_w, left_w, right_w, color_diff=True)
        )
    return "\n".join(lines)


def build_preset_column_wide(
    left: tuple[dict, Preset],
    right: tuple[dict, Preset],
    models_dir: Path | None,
    width: int,
) -> str:
    """Two-column layout: identity fields on left, runtime config on right."""
    left_fields = _identity_fields(left[0], left[1], models_dir)
    right_fields = _identity_fields(right[0], right[1], models_dir)
    field_names = ("Model", "Quant", "Preset", "Author", "Max context", "File")
    left_settings = _runtime_settings(left[0], left[1])
    right_settings = _runtime_settings(right[0], right[1])
    rows: dict[str, tuple[object, object]] = {key: (value, "") for key, value in left_settings}
    for key, value in right_settings:
        if key in rows:
            rows[key] = (rows[key][0], value)
        else:
            rows[key] = ("", value)

    # Identity rows (left column)
    identity_rows = [(name, left_fields[name], right_fields[name]) for name in field_names]
    # Runtime rows (right column)
    runtime_rows = [(key, lv, rv) for key, (lv, rv) in rows.items()]

    # Compute widths for each column independently
    id_label_w = max(len(name) for name, _, _ in identity_rows)
    id_left_w = max(len(str(lv)) if lv != "" else 1 for _, lv, _ in identity_rows)
    id_left_w = max(id_left_w, 1)

    rt_label_w = max(len(name) for name, _, _ in runtime_rows)
    rt_left_w = max(len(str(lv)) if lv != "" else 1 for _, lv, _ in runtime_rows)
    rt_left_w = max(rt_left_w, 1)

    # Estimate column widths (label + 2 + left + 2 + right)
    id_col_w = id_label_w + 2 + id_left_w + 2 + max(
        len(str(rv)) if rv != "" else 1 for _, _, rv in identity_rows
    )
    rt_col_w = rt_label_w + 2 + rt_left_w + 2 + max(
        len(str(rv)) if rv != "" else 1 for _, _, rv in runtime_rows
    )

    # If total width fits, render two columns side by side
    gap = 4
    if id_col_w + gap + rt_col_w <= width:
        id_lines = [_markup_pair(name, lv, rv, id_label_w, id_left_w) for name, lv, rv in identity_rows]
        rt_lines = [
            _markup_pair(key, lv, rv, rt_label_w, rt_left_w, color_diff=True)
            for key, lv, rv in runtime_rows
        ]
        # Pad shorter column to match height
        max_h = max(len(id_lines), len(rt_lines))
        while len(id_lines) < max_h:
            id_lines.append("")
        while len(rt_lines) < max_h:
            rt_lines.append("")
        lines = []
        for i in range(max_h):
            left_part = f"{id_lines[i]:<{id_col_w}}"
            right_part = rt_lines[i]
            lines.append(f"{left_part}{' ' * gap}{right_part}")
        return "\n".join(lines)
    else:
        # Fall back to single-column layout
        return build_preset_column(left, right, models_dir)


def build_preset_grid(
    identity: dict, preset: Preset, models_dir: Path | None
) -> str:
    """Rich markup for a preset identity + parameter grid."""
    fields = _identity_fields(identity, preset, models_dir)
    lines = _paired_markup(
        [
            ("Model", fields["Model"]),
            ("Quant", fields["Quant"]),
            ("Preset", fields["Preset"]),
            ("Author", fields["Author"]),
            ("Max context", fields["Max context"]),
            ("File", fields["File"]),
        ]
    )
    lines.append("")
    lines.append("[bold dim]RUNTIME CONFIG[/]")
    lines.extend(_paired_markup(_runtime_settings(identity, preset)))
    return "\n".join(lines)


class ConfigPanel(Static):
    """Active preset config for the selected model."""

    selected: reactive[str | None] = reactive(None)
    registry: Registry | None = None
    preset_store: PresetStore | None = None
    models_dir: Path | None = None

    def compose(self) -> ComposeResult:
        yield Static("ACTIVE PRESET", classes="card-title", id="config-title")

    @staticmethod
    def _label(label: str) -> Text:
        return _grid_label(label)

    @staticmethod
    def _value(value: object) -> Text:
        return _grid_value(value)

    def render(self) -> Group:
        title = Text("ACTIVE PRESET", style="bold")
        if self.selected and self.registry and self.selected in self.registry.models:
            model = self.registry.models[self.selected]
            identity = model.params
            active_q = model.active_quant
            preset = None
            if self.preset_store and active_q:
                slot = get_active_slot(self.preset_store, self.selected, active_q)
                if slot is not None:
                    preset = get_preset(self.preset_store, self.selected, active_q, slot)

            renderables: list = [title]
            source = identity.get("source")
            if isinstance(source, dict) and source.get("repo"):
                renderables.append(
                    Text(
                        f"{source['repo']}  •  {source.get('filename', Path(model.file).name)}",
                        style="dim",
                    )
                )
            if not preset:
                author = model_author(identity, model.file)
                max_ctx = resolve_context_length(identity, self.models_dir)
                file_status = (
                    "present"
                    if self.models_dir and model_file_exists(
                        quant_rel_paths(model, active_q) or [model.file],
                        self.models_dir,
                    )
                    else "missing"
                )
                summary = Table.grid(expand=True, padding=(0, 1))
                summary.add_column(width=12, no_wrap=True)
                summary.add_column(ratio=1)
                summary.add_column(width=12, no_wrap=True)
                summary.add_column(ratio=1)
                summary.add_row(
                    self._label("Model"),
                    self._value(model.display),
                    self._label("Quant"),
                    self._value(active_q or "—"),
                )
                summary.add_row(
                    self._label("Preset"),
                    self._value("none"),
                    self._label("Author"),
                    self._value(author or "local"),
                )
                summary.add_row(
                    self._label("Max context"),
                    self._value(
                        f"{fmt_ctx(max_ctx)} tokens" if max_ctx is not None else "unknown"
                    ),
                    self._label("File"),
                    self._value(file_status),
                )
                renderables.insert(1, summary)
                renderables.append(
                    Text("Select a preset, or press 1-5 to activate one", style="dim")
                )
                return Group(*renderables)

            renderables.insert(
                1,
                Text.from_markup(
                    build_preset_grid(identity, preset, self.models_dir)
                ),
            )
            return Group(*renderables)
        empty = Text(justify="center")
        empty.append("○ ", style="dim")
        empty.append("No model selected", style="dim")
        empty.append("\n")
        empty.append("Pick a model in the tree on the left", style="bold dim")
        return Group(title, empty)
