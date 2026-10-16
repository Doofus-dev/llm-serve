"""Sidebar model cards and alias list."""

from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal
from textual.events import Click
from textual.message import Message
from textual.widgets import ListItem, ListView, OptionList, Static
from textual.widgets.option_list import Option

from tui.data.capabilities import is_vision_enabled, nav_capability_icons
from tui.data.models_json import Registry, model_author_size_line, save_registry
from tui.data.presets import (
    PresetStore,
    get_active_slot,
    get_preset,
    list_presets_for_quant,
    merge_identity_and_preset,
)
from tui.data.quant import quant_from_filename
from tui.theme import ACCENT, ACCENT_STYLE, OK, OK_STYLE, WARN_STYLE
from tui.widgets.health import fmt_ctx, fmt_model_runtime_line

EYEBALL = "👁"
HAMMER = "🔨"
VISION_TOOLTIPS = {
    "missing": "Vision projector not downloaded",
    "off": "Vision off",
    "on": "Vision on",
}


class ToggleVision(Message):
    """User clicked the eyeball on a model card."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name


class VisionEyeball(Static):
    """Clickable vision icon; not focusable so card keyboard selection stays on ModelNav."""

    can_focus = False
    DEFAULT_CSS = """
    VisionEyeball {
        width: auto;
        height: 1;
        min-width: 2;
        padding: 0 1 0 0;
        content-align: left middle;
    }
    VisionEyeball.vision-on {
        color: $accent;
        text-style: bold;
    }
    VisionEyeball.vision-off {
        color: $text-muted;
    }
    VisionEyeball.vision-missing {
        color: $warning;
        text-style: bold;
    }
    VisionEyeball:hover {
        text-style: bold underline;
    }
    """

    def __init__(self, model_name: str, state: str) -> None:
        label = f"{EYEBALL}!" if state == "missing" else EYEBALL
        super().__init__(label, classes=f"vision-{state}", markup=False)
        self.model_name = model_name
        self.state = state
        self.tooltip = VISION_TOOLTIPS[state]

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(ToggleVision(self.model_name))


class ToolsBadge(Static):
    """Display-only hammer; not clickable and not focusable."""

    can_focus = False
    DEFAULT_CSS = """
    ToolsBadge {
        width: auto;
        height: 1;
        min-width: 2;
        padding: 0;
        color: magenta;
        content-align: left middle;
    }
    """

    def __init__(self) -> None:
        super().__init__(HAMMER, markup=False)
        self.tooltip = "Tool calls"


class ModelNav(ListView):
    """Card-based model navigation with directly selectable presets."""

    DEFAULT_CSS = """
    ModelNav {
        background: $surface-darken-1;
        padding: 1;
        border: none;
        scrollbar-size-vertical: 1;
        & > ListItem {
            width: 1fr;
            height: auto;
            padding: 0 1;
            background: $surface;
            overflow: hidden hidden;
        }
        & > ListItem.nav-model {
            margin-top: 1;
        }
        & > ListItem.nav-model:first-child {
            margin-top: 0;
        }
        & > ListItem.-highlight {
            background: $surface-darken-1;
            border-left: solid $accent;
            color: $foreground;
            text-style: none;
        }
        &:focus {
            background-tint: $foreground 0%;
            & > ListItem.-highlight {
                background: $surface-darken-1;
                color: $foreground;
                text-style: none;
            }
        }
    }
    .nav-card-row {
        width: 1fr;
        height: auto;
    }
    .nav-card-body {
        width: 1fr;
        height: auto;
    }
    .nav-card-caps {
        width: auto;
        height: 1;
        dock: right;
        align: right top;
    }
    """

    def __init__(self, registry: Registry, preset_store: PresetStore, models_dir):
        super().__init__(id="model-nav", initial_index=None)
        self.registry = registry
        self.preset_store = preset_store
        self.models_dir = models_dir
        self._option_data: dict[str, tuple] = {}
        self._option_order: list[Option] = []
        self.running_models: set[str] = set()

    async def on_mount(self) -> None:
        await self._refresh_cards()

    @property
    def option_count(self) -> int:
        return len(self._option_order)

    @property
    def highlighted(self) -> int | None:
        return self.index

    @highlighted.setter
    def highlighted(self, value: int | None) -> None:
        self.index = value

    def get_option_at_index(self, index: int) -> Option:
        return self._option_order[index]

    def get_option(self, option_id: str) -> Option:
        for option in self._option_order:
            if option.id == option_id:
                return option
        raise KeyError(option_id)

    @property
    def selected_data(self) -> tuple | None:
        if self.index is None:
            return None
        if not (0 <= self.index < len(self._option_order)):
            return None
        option = self._option_order[self.index]
        return self._option_data.get(option.id or "")

    def refresh_cards(self) -> None:
        """Rebuild model cards while retaining the current selection."""
        if self.is_attached:
            self.run_worker(
                self._refresh_cards(),
                exclusive=True,
                group="model-nav-cards",
                name="model-nav-cards",
            )

    async def _refresh_cards(self) -> None:
        selected = self.selected_data
        items, option_data, option_order = self._build_entries()
        self._option_data = option_data
        self._option_order = option_order
        if not self.is_attached:
            return
        async with self.batch():
            await self.clear()
            if items:
                await self.extend(items)
        self._restore_selection(selected)

    def _restore_selection(self, selected: tuple | None) -> None:
        if selected:
            for index, option in enumerate(self._option_order):
                if self._option_data.get(option.id or "") == selected:
                    self.index = index
                    return
        for index, option in enumerate(self._option_order):
            data = self._option_data.get(option.id or "")
            if data and data[0] == "model":
                self.index = index
                return

    def _highlight_model(self, model_name: str) -> None:
        for index, option in enumerate(self._option_order):
            if self._option_data.get(option.id or "") == ("model", model_name):
                self.index = index
                return

    def _persist_vision_enabled(self, model_name: str, enabled: bool) -> None:
        cfg = self.registry.models.get(model_name)
        if cfg is None:
            return
        cfg.params["vision_enabled"] = bool(enabled)
        paths = getattr(self.app, "paths", None)
        if paths is not None:
            save_registry(paths.models_json, self.registry)

    async def on_toggle_vision(self, event: ToggleVision) -> None:
        event.stop()
        cfg = self.registry.models.get(event.model_name)
        if cfg is None:
            return
        self._highlight_model(event.model_name)
        self._persist_vision_enabled(event.model_name, not is_vision_enabled(cfg.params))
        await self._refresh_cards()

    def _build_entries(self) -> tuple[list[ListItem], dict[str, tuple], list[Option]]:
        items: list[ListItem] = []
        option_data: dict[str, tuple] = {}
        option_order: list[Option] = []
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
            is_running = name in self.running_models
            card = Text()
            card.append("● " if is_running else "○ ", style=OK if is_running else "dim")
            card.append(model.display, style=ACCENT_STYLE)
            model_aliases = [
                alias for alias, target in aliases.items() if target.model == name
            ]
            if model_aliases:
                card.append(f"  {', '.join(model_aliases)}", style="dim")
            card.append("  ", style="dim")
            card.append(f"[{active_q}]", style=WARN_STYLE)
            card.append("\n")
            card.append(model_author_size_line(model), style="dim")
            card.append("\n")
            card.append("\n")
            card.append("Quant ", style=ACCENT)
            card.append(active_q, style=WARN_STYLE)
            if preset is not None and slot is not None:
                card.append("  •  Preset ", style="dim")
                card.append(f"[{slot}] {preset.name}", style=WARN_STYLE)
            card.append("\n")
            card.append(fmt_model_runtime_line(runtime_params, self.models_dir), style="dim")
            option_id = f"model-{model_index}"
            data = ("model", name)
            option_data[option_id] = data
            option_order.append(Option(card, id=option_id))
            vision_state, show_tools = nav_capability_icons(model.params, self.models_dir)
            items.append(
                self._model_item(option_id, card, name, vision_state, show_tools)
            )

            for slot, preset in sorted(
                list_presets_for_quant(self.preset_store, name, active_q).items()
            ):
                active = get_active_slot(self.preset_store, name, active_q) == slot
                preset_line = Text("  ")
                preset_line.append("● " if active else "○ ", style=OK if active else "dim")
                preset_line.append(f"[{slot}] {preset.name}", style="bold" if active else "")
                preset_line.append(
                    f"  ctx {fmt_ctx(preset.params.get('ctx', '?'))}",
                    style="dim",
                )
                preset_id = f"preset-{model_index}-{slot}"
                option_data[preset_id] = ("preset", name, active_q, slot)
                option_order.append(Option(preset_line, id=preset_id))
                items.append(
                    ListItem(
                        Static(preset_line, markup=False, classes="nav-card-body"),
                        id=preset_id,
                        classes="nav-preset",
                    )
                )
        return items, option_data, option_order

    @staticmethod
    def _model_item(
        option_id: str,
        card: Text,
        model_name: str,
        vision_state: str | None,
        show_tools: bool,
    ) -> ListItem:
        caps: list[Static] = []
        if vision_state is not None:
            caps.append(VisionEyeball(model_name, vision_state))
        if show_tools:
            caps.append(ToolsBadge())
        body = Static(card, markup=False, classes="nav-card-body")
        if caps:
            row = Horizontal(
                body,
                Horizontal(*caps, classes="nav-card-caps"),
                classes="nav-card-row",
            )
        else:
            row = Horizontal(body, classes="nav-card-row")
        return ListItem(row, id=option_id, classes="nav-model")


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
            line = Text(f"{alias}", style=WARN_STYLE)
            target_model = self.registry.models.get(target.model)
            line.append(
                f"  →  {target_model.display if target_model else target.model}",
                style="dim",
            )
            if target.quant is not None and target.preset_slot is not None:
                line.append(
                    f"  {target.preset_slot}",
                    style=OK_STYLE,
                )
            line.append("  ←/→", style=ACCENT_STYLE)
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
