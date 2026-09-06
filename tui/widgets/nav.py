"""Sidebar model cards and alias list."""

from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from tui.data.models_json import Registry, model_author_size_line
from tui.data.presets import (
    PresetStore,
    get_active_slot,
    get_preset,
    list_presets_for_quant,
    merge_identity_and_preset,
)
from tui.data.quant import quant_from_filename
from tui.widgets.health import fmt_ctx, fmt_model_runtime_line


class ModelNav(OptionList):
    """Card-based model navigation with directly selectable presets."""

    def __init__(self, registry: Registry, preset_store: PresetStore, models_dir):
        super().__init__(id="model-nav")
        self.registry = registry
        self.preset_store = preset_store
        self.models_dir = models_dir
        self._option_data: dict[str, tuple] = {}

    def on_mount(self) -> None:
        self.refresh_cards()

    @property
    def selected_data(self) -> tuple | None:
        if self.highlighted is None:
            return None
        option = self.get_option_at_index(self.highlighted)
        return self._option_data.get(option.id or "")

    def _add(self, prompt: Text, data: tuple, option_id: str) -> None:
        self._option_data[option_id] = data
        self.add_option(Option(prompt, id=option_id))

    def refresh_cards(self) -> None:
        """Rebuild model cards while retaining the current selection."""
        selected = self.selected_data
        self.clear_options()
        self._option_data.clear()
        aliases = self.registry.aliases
        for model_index, (name, model) in enumerate(self.registry.models.items()):
            active_q = model.active_quant or quant_from_filename(model.file)
            runtime_params = dict(model.params)
            slot = get_active_slot(self.preset_store, name, active_q)
            preset = None
            if slot is not None:
                preset = get_preset(self.preset_store, name, active_q, slot)
                if preset:
                    runtime_params = merge_identity_and_preset(model.params, preset)

            card = Text()
            card.append(model.display, style="bold cyan")
            model_aliases = [
                alias for alias, target in aliases.items() if target.model == name
            ]
            if model_aliases:
                card.append(f"  {', '.join(model_aliases)}", style="dim")
            card.append("\n")
            card.append(model_author_size_line(model), style="dim")
            card.append("\n")
            card.append("Quant ", style="cyan")
            card.append(active_q, style="bold yellow")
            if preset is not None and slot is not None:
                card.append("  •  Preset ", style="dim")
                card.append(f"[{slot}] {preset.name}", style="bold yellow")
            card.append("\n")
            card.append(fmt_model_runtime_line(runtime_params, self.models_dir), style="dim")
            self._add(card, ("model", name), f"model-{model_index}")

            for slot, preset in sorted(list_presets_for_quant(self.preset_store, name, active_q).items()):
                active = get_active_slot(self.preset_store, name, active_q) == slot
                preset_line = Text("  ")
                preset_line.append("● " if active else "○ ", style="green" if active else "dim")
                preset_line.append(f"[{slot}] {preset.name}", style="bold" if active else "")
                preset_line.append(
                    f"  ctx {fmt_ctx(preset.params.get('ctx', '?'))}",
                    style="dim",
                )
                self._add(
                    preset_line,
                    ("preset", name, active_q, slot),
                    f"preset-{model_index}-{slot}",
                )
            self.add_option(None)

        if selected:
            for index in range(self.option_count):
                option = self.get_option_at_index(index)
                if self._option_data.get(option.id or "") == selected:
                    self.highlighted = index
                    break
        elif self.option_count:
            for index in range(self.option_count):
                option = self.get_option_at_index(index)
                data = self._option_data.get(option.id or "")
                if data and data[0] == "model":
                    self.highlighted = index
                    break


class AliasNav(OptionList):
    """Bottom sidebar section for aliases, reachable with Tab."""

    BINDINGS = [
        Binding("left", "cycle_alias(-1)", "Previous alias target", show=False),
        Binding("right", "cycle_alias(1)", "Next alias target", show=False),
    ]

    def __init__(self, registry: Registry):
        super().__init__(id="alias-nav")
        self.registry = registry
        self._option_data: dict[str, tuple] = {}

    def on_mount(self) -> None:
        self.refresh_aliases()

    def action_cycle_alias(self, direction: int) -> None:
        self.app.cycle_selected_alias(direction)

    @property
    def selected_data(self) -> tuple | None:
        if self.highlighted is None:
            return None
        option = self.get_option_at_index(self.highlighted)
        return self._option_data.get(option.id or "")

    def refresh_aliases(self) -> None:
        selected = self.selected_data
        self.clear_options()
        self._option_data.clear()
        for index, (alias, target) in enumerate(self.registry.aliases.items()):
            line = Text(f"{alias}", style="bold yellow")
            target_model = self.registry.models.get(target.model)
            line.append(
                f"  →  {target_model.display if target_model else target.model}",
                style="dim",
            )
            if target.quant is not None and target.preset_slot is not None:
                line.append(
                    f"  {target.preset_slot}",
                    style="bold green",
                )
            line.append("  ←/→", style="bold cyan")
            option_id = f"alias-{index}"
            self._option_data[option_id] = ("alias", alias)
            self.add_option(Option(line, id=option_id))

        if not self.registry.aliases:
            self.add_option(Option(Text("No aliases", style="dim"), disabled=True))

        if selected:
            for index in range(self.option_count):
                option = self.get_option_at_index(index)
                if self._option_data.get(option.id or "") == selected:
                    self.highlighted = index
                    break
        elif self.registry.aliases:
            for index in range(self.option_count):
                option = self.get_option_at_index(index)
                data = self._option_data.get(option.id or "")
                if data and data[0] == "alias":
                    self.highlighted = index
                    break

