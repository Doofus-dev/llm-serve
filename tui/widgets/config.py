"""Active preset summary for the selected model."""

from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from tui.data.context_length import resolve_context_length
from tui.data.models_json import Registry
from tui.data.presets import PresetStore, get_active_slot, get_preset, merge_identity_and_preset
from tui.paths import PROFILE_KEYS
from tui.widgets.health import fmt_ctx, model_author, model_file_exists


class ConfigPanel(Static):
    """Active preset config for the selected model."""

    selected: reactive[str | None] = reactive(None)
    registry: Registry | None = None
    preset_store: PresetStore | None = None
    models_dir: Path | None = None

    @staticmethod
    def _label(label: str) -> Text:
        return Text(label, style="cyan")

    @staticmethod
    def _value(value: object) -> Text:
        if value == "":
            return Text("—", style="dim")
        return Text(str(value), style="bold yellow")

    def render(self) -> Group:
        title = Text("ACTIVE PRESET", style="bold")
        if self.selected and self.registry and self.selected in self.registry.models:
            model = self.registry.models[self.selected]
            identity = model.params
            active_q = model.active_quant
            slot = None
            preset = None
            if self.preset_store and active_q:
                slot = get_active_slot(self.preset_store, self.selected, active_q)
                if slot is not None:
                    preset = get_preset(self.preset_store, self.selected, active_q, slot)

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
                self._value(f"[{slot}] {preset.name}" if preset else "none"),
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

            renderables: list = [title, summary]
            source = identity.get("source")
            if isinstance(source, dict) and source.get("repo"):
                renderables.append(
                    Text(
                        f"{source['repo']}  •  {source.get('filename', Path(model.file).name)}",
                        style="dim",
                    )
                )
            if not preset:
                renderables.append(
                    Text("Select a preset and press A to activate", style="dim")
                )
                return Group(*renderables)

            params = merge_identity_and_preset(identity, preset)
            settings: list[tuple[str, object]] = []
            for k, v in params.items():
                if k in {"source", "quants", "file"}:
                    continue
                if k in PROFILE_KEYS:
                    continue
                settings.append((k, v))

            config = Table.grid(expand=True, padding=(0, 1))
            config.add_column(width=19, no_wrap=True)
            config.add_column(ratio=1)
            config.add_column(width=19, no_wrap=True)
            config.add_column(ratio=1)
            for index in range(0, len(settings), 2):
                left_label, left_value = settings[index]
                if index + 1 < len(settings):
                    right_label, right_value = settings[index + 1]
                    right_cells = (
                        self._label(right_label),
                        self._value(right_value),
                    )
                else:
                    right_cells = (Text(""), Text(""))
                config.add_row(
                    self._label(left_label),
                    self._value(left_value),
                    *right_cells,
                )
            renderables.extend((Text("RUNTIME CONFIG", style="bold dim"), config))
            return Group(*renderables)
        else:
            return Group(title, Text("Select a model in the tree", style="dim"))
