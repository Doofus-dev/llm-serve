"""Side-by-side preset compare and independent edit."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from tui.data.presets import (
    PresetStore,
    diff_preset_params,
    get_preset,
    list_presets_for_quant,
    save_presets,
    set_preset,
)
from tui.screens.editors import ParamHelpPanel, PresetEditor
from tui.theme import ERR, OK, WARN
from tui.widgets.action_bar import ActionBar

DIFF_CLASSES = ("diff-changed", "diff-added", "diff-removed")
SLOT_SELECT_IDS = {"cmp-left-slot", "cmp-right-slot"}


def _fmt_side(value: str) -> str:
    return value if value else "—"


def format_diff_summary(
    left_name: str,
    right_name: str,
    left_params: dict,
    right_params: dict,
) -> str:
    lines = ["[bold]Diff[/]"]
    if left_name != right_name:
        lines.append(
            f"  [{WARN}]name[/]  {_fmt_side(left_name)} → {_fmt_side(right_name)}"
        )
    diff = diff_preset_params(left_params, right_params)
    labels = {
        "changed": "changed",
        "added": "added on right",
        "removed": "only on left",
    }
    for kind, color in (("changed", WARN), ("added", OK), ("removed", ERR)):
        items = diff[kind]
        if not items:
            continue
        lines.append(f"  [{color}]{labels[kind]}[/]")
        for param, (left_val, right_val) in items.items():
            lines.append(
                f"    [{color}]{param}[/]  {_fmt_side(left_val)} → {_fmt_side(right_val)}"
            )
    if len(lines) == 1 and left_name == right_name:
        lines.append("  [dim]no differences[/]")
    elif len(lines) == 1:
        lines.append("  [dim]no param differences[/]")
    return "\n".join(lines)


class PresetCompareScreen(ModalScreen[None]):
    """Compare and independently edit two preset slots of one model/quant."""

    BINDINGS = [
        Binding("escape", "close_compare", "Close"),
        Binding("ctrl+s", "save_focused", "Save side"),
        Binding("f2", "toggle_param_help", "Param Help"),
    ]

    CSS = """\
    PresetCompareScreen {
        background: $surface;
        align: center middle;
    }
    #compare-root {
        height: 100%;
        width: 100%;
        padding: 0 1;
    }
    #compare-header {
        height: auto;
        color: $foreground;
        padding: 0 1;
        border-bottom: solid $border;
    }
    #compare-pickers {
        height: auto;
        padding: 0 1;
    }
    #compare-pickers Select {
        width: 1fr;
        margin: 0 1;
    }
    #compare-pickers Label {
        width: auto;
        color: $accent;
        height: 1;
        content-align: left middle;
    }
    #diff-summary-wrap {
        height: auto;
        max-height: 10;
    }
    #diff-summary {
        height: auto;
        padding: 0 1;
        color: $foreground;
        border: round $border;
        background: $panel;
    }
    #compare-editors {
        height: 1fr;
        width: 1fr;
    }
    .compare-editor {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        border: round $border;
        background: $panel;
        overflow-y: auto;
    }
    .param-field.diff-changed {
        border-left: solid $warning;
    }
    .param-field.diff-added {
        border-left: solid $success;
    }
    .param-field.diff-removed {
        border-left: solid $error;
    }
    #compare-param-help {
        height: 8;
        display: none;
        border-top: solid $border;
        padding: 0 1;
    }
    PresetCompareScreen.help-open #compare-param-help {
        display: block;
    }
    """

    def __init__(
        self,
        *,
        model_name: str,
        quant: str,
        identity: dict,
        store: PresetStore,
        presets_path: Path,
        models_dir: Path,
        left_slot: int,
        right_slot: int,
        on_saved=None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.quant = quant
        self.identity = dict(identity)
        self.store = store
        self.presets_path = presets_path
        self.models_dir = models_dir
        self.left_slot = left_slot
        self.right_slot = right_slot
        self.on_saved = on_saved
        self._help_visible = False
        self._syncing_selects = False
        self._replacing = False
        self._replace_lock = asyncio.Lock()

    def _slot_options(self, exclude: int) -> list[tuple[str, str]]:
        presets = list_presets_for_quant(self.store, self.model_name, self.quant)
        return [
            (f"{slot}: {preset.name}", str(slot))
            for slot, preset in sorted(presets.items())
            if slot != exclude
        ]

    def _refresh_slot_selects(self) -> None:
        if self._replacing:
            return
        try:
            left = self.query_one("#cmp-left-slot", Select)
            right = self.query_one("#cmp-right-slot", Select)
        except NoMatches:
            return
        self._syncing_selects = True
        try:
            self._apply_select_options(
                left, self._slot_options(exclude=self.right_slot), str(self.left_slot)
            )
            self._apply_select_options(
                right, self._slot_options(exclude=self.left_slot), str(self.right_slot)
            )
        except Exception:
            self._syncing_selects = False
            raise
        self.call_after_refresh(self._finish_syncing_selects)

    def _finish_syncing_selects(self) -> None:
        self._syncing_selects = False

    def _apply_select_options(
        self, select: Select, options: list[tuple[str, str]], value: str
    ) -> None:
        select.set_options(options)
        select.value = value
        prompt = next((label for label, opt in options if opt == value), None)
        if prompt is not None:
            select.query_one("#label", Static).update(prompt)

    def _set_compare_title(self, editor: PresetEditor, slot: int, name: str) -> None:
        title_id = f"#{editor.id_prefix}title"
        try:
            editor.query_one(title_id, Label).update(
                f"[bold][{slot}] {name}[/bold]  Ctrl+S saves this side"
            )
        except NoMatches:
            return
        editor.preset.name = name

    def _make_editor(self, side: str, slot: int) -> PresetEditor:
        preset = get_preset(self.store, self.model_name, self.quant, slot)

        def on_save(name: str, params: dict, *, _slot: int = slot, _side: str = side) -> None:
            set_preset(self.store, self.model_name, self.quant, _slot, name, params)
            save_presets(self.presets_path, self.store)
            if self.on_saved:
                self.on_saved()
            if hasattr(self.app, "preset_store"):
                self.store = self.app.preset_store
            self.app.notify(
                f"Saved {self.model_name}/{self.quant} [{_slot}] {name}"
            )
            try:
                self._set_compare_title(self._editor(_side), _slot, name)
            except NoMatches:
                pass
            self._refresh_slot_selects()
            self._refresh_diff()

        editor = PresetEditor(
            self.model_name,
            slot,
            preset,
            self.identity,
            models_dir=self.models_dir,
            on_save=on_save,
            on_cancel=self.action_close_compare,
            id_prefix=f"cmp-{side}-",
            compare_mode=True,
        )
        editor.id = f"cmp-{side}-editor"
        editor.add_class("compare-editor")
        return editor

    def compose(self) -> ComposeResult:
        with Vertical(id="compare-root"):
            yield Label(
                f"[bold]Compare presets[/]  {self.model_name} / {self.quant}",
                id="compare-header",
            )
            with Horizontal(id="compare-pickers"):
                yield Label("Left")
                yield Select(
                    self._slot_options(exclude=self.right_slot),
                    value=str(self.left_slot),
                    id="cmp-left-slot",
                    compact=True,
                    allow_blank=False,
                )
                yield Label("Right")
                yield Select(
                    self._slot_options(exclude=self.left_slot),
                    value=str(self.right_slot),
                    id="cmp-right-slot",
                    compact=True,
                    allow_blank=False,
                )
            with VerticalScroll(id="diff-summary-wrap"):
                yield Static("", id="diff-summary")
            with Horizontal(id="compare-editors"):
                yield self._make_editor("left", self.left_slot)
                yield self._make_editor("right", self.right_slot)
            with ActionBar():
                yield Button("Close", id="close-compare")
            yield ParamHelpPanel(id="compare-param-help")

    def on_mount(self) -> None:
        self.call_after_refresh(self._refresh_diff)

    def _editor(self, side: str) -> PresetEditor:
        return self.query_one(f"#cmp-{side}-editor", PresetEditor)

    def _current_params(self, editor: PresetEditor) -> dict:
        values = dict(editor.preset.params)
        values.update(editor._read_fields())
        return values

    def _set_field_diff(self, side: str, param: str, kind: str | None) -> None:
        field_id = f"#cmp-{side}-field_{param}"
        try:
            field = self.query_one(field_id)
        except NoMatches:
            return
        field.remove_class(*DIFF_CLASSES)
        if kind:
            field.add_class(f"diff-{kind}")

    def _refresh_diff(self) -> None:
        if self._replacing:
            return
        try:
            left = self._editor("left")
            right = self._editor("right")
        except NoMatches:
            return
        left_params = self._current_params(left)
        right_params = self._current_params(right)
        left_name = left.name_input.value.strip() if left.name_input else left.preset.name
        right_name = (
            right.name_input.value.strip() if right.name_input else right.preset.name
        )
        summary = self.query_one("#diff-summary", Static)
        summary.update(
            format_diff_summary(left_name, right_name, left_params, right_params)
        )
        diff = diff_preset_params(left_params, right_params)
        kinds: dict[str, str] = {}
        for param in diff["changed"]:
            kinds[param] = "changed"
        for param in diff["added"]:
            kinds[param] = "added"
        for param in diff["removed"]:
            kinds[param] = "removed"
        params = set(left.fields) | set(right.fields)
        for param in params:
            kind = kinds.get(param)
            left_kind = None if kind == "added" else kind
            right_kind = None if kind == "removed" else kind
            self._set_field_diff("left", param, left_kind)
            self._set_field_diff("right", param, right_kind)

    def _replace_side(self, side: str, slot: int) -> None:
        self.run_worker(
            self._replace_side_async(side, slot),
            group=f"replace-side-{side}",
        )

    async def _replace_side_async(self, side: str, slot: int) -> None:
        async with self._replace_lock:
            if side == "left":
                self.left_slot = slot
            else:
                self.right_slot = slot
            old = self._editor(side)
            if old.slot == slot:
                return
            self._replacing = True
            try:
                container = self.query_one("#compare-editors", Horizontal)
                other_side = "right" if side == "left" else "left"
                other = self._editor(other_side)
                new = self._make_editor(side, slot)
                await old.remove()
                if side == "left":
                    await container.mount(new, before=other)
                else:
                    await container.mount(new, after=other)
            finally:
                self._replacing = False
            self.call_after_refresh(self._after_side_replaced)

    def _after_side_replaced(self) -> None:
        self._refresh_slot_selects()
        self._refresh_diff()

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._syncing_selects:
            return
        if event.select.id not in SLOT_SELECT_IDS:
            self._refresh_diff()
            return
        if event.value is Select.NULL or event.value is None:
            return
        slot = int(str(event.value))
        other_slot = self.right_slot if event.select.id == "cmp-left-slot" else self.left_slot
        if slot == other_slot:
            self._refresh_slot_selects()
            return
        if event.select.id == "cmp-left-slot":
            if slot == self.left_slot or self._editor("left").slot == slot:
                self.left_slot = slot
                return
            self.left_slot = slot
            self._refresh_slot_selects()
            self._replace_side("left", slot)
            return
        if slot == self.right_slot or self._editor("right").slot == slot:
            self.right_slot = slot
            return
        self.right_slot = slot
        self._refresh_slot_selects()
        self._replace_side("right", slot)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_diff()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-compare":
            self.action_close_compare()

    def action_close_compare(self) -> None:
        self.dismiss(None)

    def action_save_focused(self) -> None:
        node = self.focused
        while node is not None:
            if isinstance(node, PresetEditor):
                node.save()
                return
            node = node.parent
        self.app.notify("Focus a preset field, then Ctrl+S to save that side", severity="warning")

    def action_toggle_param_help(self) -> None:
        self._help_visible = not self._help_visible
        if self._help_visible:
            self.add_class("help-open")
            panel = self.query_one("#compare-param-help", ParamHelpPanel)
            focused = self.focused
            param = getattr(focused, "param_name", None)
            panel.show_param(param)
        else:
            self.remove_class("help-open")
