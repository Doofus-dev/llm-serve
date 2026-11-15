"""Inline preset picker overlay for a model's active quant."""

from __future__ import annotations

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from tui.data.presets import Preset, list_presets_for_quant
from tui.widgets.health import fmt_ctx


class PresetPickerScreen(ModalScreen[int | None]):
    """List a quant's presets; returns the chosen slot, or None on cancel.

    Replaces the removed per-model preset sub-rows in the sidebar: ``n``
    opens this overlay, 1-5 still activate directly, and Esc cancels.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    PresetPickerScreen {
        align: center middle;
    }

    #preset-picker {
        width: 46;
        height: auto;
        max-height: 16;
        border: round $primary;
        background: $surface;
        padding: 1 0;
    }

    #preset-picker-title {
        height: 1;
        padding: 0 1;
        color: $accent;
        text-style: bold;
        border-bottom: solid $border;
    }

    #preset-picker-list {
        height: auto;
        max-height: 12;
        border: none;
    }

    #preset-picker-list > .option-list--option {
        padding: 0 1;
    }

    #preset-picker-list > .option-list--option-highlighted {
        background: $primary-background;
        color: $foreground;
        text-style: none;
    }
    """

    def __init__(self, title: str, presets: dict[int, Preset], active_slot: int | None):
        super().__init__()
        self.title = title
        self.presets = dict(presets)
        self.active_slot = active_slot

    def compose(self):
        with Vertical(id="preset-picker"):
            yield Label(self.title, id="preset-picker-title")
            yield OptionList(id="preset-picker-list")

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        for slot in sorted(self.presets):
            preset = self.presets[slot]
            line = Text()
            active = slot == self.active_slot
            line.append("● " if active else "○ ", style="bold green" if active else "dim")
            line.append(f"[{slot}] {preset.name}", style="bold" if active else "")
            ctx = preset.params.get("ctx")
            if ctx not in ("", None):
                line.append(f"  ctx {fmt_ctx(ctx)}", style="dim")
            option_list.add_option(Option(line, id=str(slot)))
        if not self.presets:
            empty = Text("no presets", style="dim")
            option_list.add_option(Option(empty, disabled=True))
        option_list.focus()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        slot = int(event.option.id or 0)
        self.dismiss(slot)

    def action_cancel(self) -> None:
        self.dismiss(None)
