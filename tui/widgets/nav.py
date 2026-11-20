"""Sidebar model cards and alias list."""

from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal
from textual.events import Blur, Click, Focus, Key
from textual.message import Message
from textual.widgets import ListItem, ListView, OptionList, Static
from textual.widgets.option_list import Option

from tui.data.capabilities import is_mtp_enabled, is_vision_enabled, nav_capability_icons
from tui.data.hf import fmt_size
from tui.data.models_json import (
    Registry,
    coerce_rel_paths,
    quant_rel_paths,
    save_registry,
)
from tui.data.context_length import fmt_ctx_compact
from tui.data.presets import PresetStore, get_active_slot, list_presets_for_quant
from tui.data.quant import author_size_label, quant_from_filename
from tui.data.settings import save_settings
from tui.theme import ACCENT_STYLE, BOLD_STYLE, DIM_STYLE, OK, OK_STYLE, WARN_STYLE

EYEBALL = "👁"
HAMMER = "🔨"
MTP_MARK = "M"
VISION_TOOLTIPS = {
    "missing": "Vision projector not downloaded",
    "off": "Vision off",
    "on": "Vision on",
}
MTP_TOOLTIPS = {
    "missing": "MTP draft GGUF not downloaded",
    "off": "MTP off",
    "on": "MTP on",
}


class ToggleVision(Message):
    """User clicked the eyeball on a model card."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name


class ToggleMtp(Message):
    """User clicked the MTP mark on a model card."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name


class ToggleAuthor(Message):
    """User clicked an author header to expand/collapse its models."""

    def __init__(self, author: str) -> None:
        super().__init__()
        self.author = author


class VisionEyeball(Static):
    """Clickable vision icon; not focusable so card keyboard selection stays on ModelNav."""

    can_focus = False
    DEFAULT_CSS = """
    VisionEyeball {
        width: auto;
        height: 1;
        min-width: 3;
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
        # Trailing space matches the missing "!" so on/off keep the same box
        # and the hammer never sits flush against the eyeball.
        label = f"{EYEBALL}!" if state == "missing" else f"{EYEBALL} "
        super().__init__(label, classes=f"vision-{state}", markup=False)
        self.model_name = model_name
        self.state = state
        self.tooltip = VISION_TOOLTIPS[state]

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(ToggleVision(self.model_name))


class MtpMark(Static):
    """Clickable MTP icon; not focusable so card keyboard selection stays on ModelNav."""

    can_focus = False
    DEFAULT_CSS = """
    MtpMark {
        width: auto;
        height: 1;
        min-width: 3;
        padding: 0 1 0 0;
        content-align: left middle;
    }
    MtpMark.mtp-on {
        color: $accent;
        text-style: bold;
    }
    MtpMark.mtp-off {
        color: $text-muted;
    }
    MtpMark.mtp-missing {
        color: $warning;
        text-style: bold;
    }
    MtpMark:hover {
        text-style: bold underline;
    }
    """

    def __init__(self, model_name: str, state: str) -> None:
        label = f"{MTP_MARK}!" if state == "missing" else f"{MTP_MARK} "
        super().__init__(label, classes=f"mtp-{state}", markup=False)
        self.model_name = model_name
        self.state = state
        self.tooltip = MTP_TOOLTIPS[state]

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(ToggleMtp(self.model_name))


class ToolsBadge(Static):
    """Display-only hammer; not clickable and not focusable."""

    can_focus = False
    DEFAULT_CSS = """
    ToolsBadge {
        width: auto;
        height: 1;
        min-width: 2;
        padding: 0 0 0 1;
        color: magenta;
        content-align: left middle;
    }
    """

    def __init__(self) -> None:
        super().__init__(HAMMER, markup=False)
        self.tooltip = "Tool calls"


class AuthorHeader(Static):
    """Clickable author section title; not focusable so keyboard stays on ModelNav."""

    can_focus = False
    DEFAULT_CSS = """
    AuthorHeader {
        width: 1fr;
        height: 1;
        color: $text;
        text-style: bold;
    }
    AuthorHeader:hover {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self, author: str, label: Text) -> None:
        super().__init__(label, markup=False)
        self.author = author

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(ToggleAuthor(self.author))


def _on_disk_size_bytes(models_dir, rels: list[str]) -> int:
    total = 0
    for rel in rels:
        try:
            total += (models_dir / rel).stat().st_size
        except OSError:
            pass
    return total


def _first_rel(model) -> str:
    rels = coerce_rel_paths(model.params.get("files")) or coerce_rel_paths(
        model.params.get("file")
    )
    return rels[0] if rels else ""


class ModelNav(ListView):
    """Compact model rows grouped by author, with type-to-filter search.

    Author headers are expandable/collapsible sections; each model's presets
    are selectable sub-rows indented under it. ``1-5`` activate a preset
    directly and ``n`` opens the preset picker overlay.
    """

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
    .nav-group {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
        color: $text;
        text-style: bold;
        border-bottom: solid $border;
        background: $panel;
    }
    .nav-group:first-child {
        margin-top: 0;
    }
    .nav-group.-highlight {
        background: $surface-darken-1;
        color: $text;
    }
    .nav-model {
        padding-left: 2;
    }
    .nav-preset {
        padding-left: 4;
    }
    .nav-preset.-highlight {
        background: $surface-darken-1;
    }
    .nav-filter {
        height: 1;
        padding: 0 1;
        color: $accent;
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
        self.filter_text: str = ""
        self.collapsed_authors: set[str] = set()

    def _on_focus(self, event: Focus) -> None:
        super()._on_focus(event)
        self.app._update_footer()

    def _on_blur(self, event: Blur) -> None:
        super()._on_blur(event)
        self.app._update_footer()

    @property
    def _filter_marker(self) -> str:
        return "nav-filter-row"

    async def on_mount(self) -> None:
        settings = getattr(self.app, "settings", None)
        if settings is not None and hasattr(settings, "collapsed_authors"):
            self.collapsed_authors = set(settings.collapsed_authors)
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

    def _persist_mtp_enabled(self, model_name: str, enabled: bool) -> None:
        cfg = self.registry.models.get(model_name)
        if cfg is None:
            return
        cfg.params["mtp_enabled"] = bool(enabled)
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

    async def on_toggle_mtp(self, event: ToggleMtp) -> None:
        event.stop()
        cfg = self.registry.models.get(event.model_name)
        if cfg is None:
            return
        self._highlight_model(event.model_name)
        self._persist_mtp_enabled(event.model_name, not is_mtp_enabled(cfg.params))
        await self._refresh_cards()

    async def on_toggle_author(self, event: ToggleAuthor) -> None:
        event.stop()
        self.toggle_author(event.author)
        await self._refresh_cards()

    def toggle_author(self, author: str) -> None:
        if author in self.collapsed_authors:
            self.collapsed_authors.discard(author)
        else:
            self.collapsed_authors.add(author)
        self._persist_collapsed_authors()

    def _persist_collapsed_authors(self) -> None:
        settings = getattr(self.app, "settings", None)
        if settings is None:
            return
        settings.collapsed_authors = sorted(self.collapsed_authors)
        paths = getattr(self.app, "paths", None)
        if paths is not None:
            save_settings(paths.settings_json, settings)

    def _author_for(self, model) -> str:
        return author_size_label(model.params, _first_rel(model)).split(" ", 1)[0] or "local"

    def _row_matches_filter(self, name: str, model) -> bool:
        needle = self.filter_text.lower()
        if not needle:
            return True
        haystacks = [model.display, name]
        quants = model.params.get("quants")
        if isinstance(quants, dict):
            haystacks.extend(quants.keys())
        haystacks.append(self._author_for(model))
        return any(needle in haystack.lower() for haystack in haystacks)

    def _build_entries(self) -> tuple[list[ListItem], dict[str, tuple], list[Option]]:
        items: list[ListItem] = []
        option_data: dict[str, tuple] = {}
        option_order: list[Option] = []
        if self.filter_text:
            filter_line = Text()
            filter_line.append("filter: ", style=DIM_STYLE)
            filter_line.append(self.filter_text, style=ACCENT_STYLE)
            items.append(
                ListItem(
                    Static(filter_line, markup=False, classes="nav-filter"),
                    id=self._filter_marker,
                    classes="nav-filter",
                )
            )

        groups: dict[str, list[str]] = {}
        for name, model in self.registry.models.items():
            if not self._row_matches_filter(name, model):
                continue
            groups.setdefault(self._author_for(model), []).append(name)

        model_index = 0
        for author in sorted(groups):
            collapsed = author in self.collapsed_authors and not self.filter_text
            marker = "▸ " if collapsed else "▾ "
            group_line = Text(marker, style=ACCENT_STYLE)
            group_line.append(f"{author.title()}", style=BOLD_STYLE)
            group_line.append(f"  ({len(groups[author])})", style=ACCENT_STYLE)
            group_classes = "nav-group nav-group-collapsed" if collapsed else "nav-group"
            group_id = f"nav-group-{author}"
            option_data[group_id] = ("author", author)
            option_order.append(Option(group_line, id=group_id))
            items.append(
                ListItem(
                    AuthorHeader(author, group_line),
                    id=group_id,
                    classes=group_classes,
                )
            )
            if collapsed:
                continue
            for name in groups[author]:
                model = self.registry.models[name]
                active_q = model.active_quant or quant_from_filename(model.file)
                is_running = name in self.running_models
                presets = list_presets_for_quant(self.preset_store, name, active_q)
                active_slot = get_active_slot(self.preset_store, name, active_q)
                card = Text()
                card.append("● " if is_running else "○ ", style=OK if is_running else "dim")
                card.append(model.display, style=ACCENT_STYLE)
                card.append(f"  [{active_q}]", style=WARN_STYLE)
                size = _on_disk_size_bytes(
                    self.models_dir, quant_rel_paths(model, active_q)
                )
                if size:
                    card.append(f"  {fmt_size(size)}", style=DIM_STYLE)
                option_id = f"model-{model_index}"
                option_data[option_id] = ("model", name)
                option_order.append(Option(card, id=option_id))
                vision_state, show_tools, mtp_state = nav_capability_icons(
                    model.params, self.models_dir
                )
                items.append(
                    self._model_item(
                        option_id, card, name, vision_state, show_tools, mtp_state
                    )
                )
                model_index += 1
                # Preset sub-rows: indented under the model, selectable.
                for slot in sorted(presets):
                    preset = presets[slot]
                    row = Text()
                    row.append("    ", style="dim")
                    row.append(
                        "● " if slot == active_slot else "○ ",
                        style=OK_STYLE if slot == active_slot else "dim",
                    )
                    row.append(f"[{slot}] ", style=ACCENT_STYLE)
                    row.append(preset.name, style="dim")
                    row.append("  " + fmt_ctx_compact(preset.params.get("ctx", 0)), style="dim")
                    preset_id = f"preset-{model_index}-{slot}"
                    option_data[preset_id] = ("preset", name, active_q, slot)
                    option_order.append(Option(row, id=preset_id))
                    items.append(
                        ListItem(
                            Static(row, markup=False, classes="nav-preset"),
                            id=preset_id,
                            classes="nav-preset",
                        )
                    )
        return items, option_data, option_order

    def on_key(self, event: Key) -> None:
        """Type-to-filter: printable chars search, Esc clears.

        App-level binding keys (launch/stop/pick/edit/…) are not consumed so
        footer hotkeys keep working while the model list is focused.
        """
        if not self.has_focus:
            return
        key = event.key
        if key == "escape":
            if self.filter_text:
                self.filter_text = ""
                self.refresh_cards()
                event.stop()
            return
        if len(key) == 1 and key.isprintable():
            try:
                if self.app.check_action(key, ()):
                    return
            except Exception:
                pass
            self.filter_text += key
            self.refresh_cards()
            event.stop()

    @staticmethod
    def _model_item(
        option_id: str,
        card: Text,
        model_name: str,
        vision_state: str | None,
        show_tools: bool,
        mtp_state: str | None = None,
    ) -> ListItem:
        caps: list[Static] = []
        if vision_state is not None:
            caps.append(VisionEyeball(model_name, vision_state))
        if mtp_state is not None:
            caps.append(MtpMark(model_name, mtp_state))
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

    def _on_focus(self, event: Focus) -> None:
        super()._on_focus(event)
        self.app._update_footer()

    def _on_blur(self, event: Blur) -> None:
        super()._on_blur(event)
        self.app._update_footer()

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
