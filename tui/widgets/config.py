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
from tui.data.models_json import Registry
from tui.data.presets import (
    Preset,
    PresetStore,
    get_active_slot,
    get_preset,
    merge_identity_and_preset,
)
from tui.paths import PROFILE_KEYS
from tui.theme import ACCENT, WARN_STYLE
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
    file_rel = str(identity.get("file") or "")
    author = model_author(identity, file_rel)
    max_ctx = resolve_context_length(identity, models_dir)
    file_status = (
        "present"
        if models_dir and model_file_exists(file_rel, models_dir)
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
        if key in {"source", "quants", "file"}:
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
                    if self.models_dir and model_file_exists(model.file, self.models_dir)
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
