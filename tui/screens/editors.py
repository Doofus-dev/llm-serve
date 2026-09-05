"""Model/preset editors and small modal dialogs."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Focus
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Label, Select, Static

from tui.data.context_length import fmt_ctx_compact as fmt_ctx, resolve_context_length
from tui.data.gguf import apply_architecture_from_gguf
from tui.data.models_json import Registry, validate_display_name
from tui.data.param_help import get_param_help, load_param_help
from tui.data.param_widgets import (
    HIDDEN_EDITOR_PARAMS,
    PARAM_ALLOW_BLANK,
    fmt_locked_value,
    is_select_param,
    read_field_value,
    select_initial_value,
    select_options,
)
from tui.data.preset_template import PRESET_PARAM_GROUPS, default_preset_params
from tui.data.presets import Preset
from tui.paths import LOCKED_PARAMS
from tui.widgets.action_bar import ActionBar


class ParamFocused(Message):
    """Posted when an editor input field receives focus."""

    def __init__(self, param: str) -> None:
        self.param = param
        super().__init__()


EDITOR_BINDINGS = [
    Binding("f2", "toggle_param_help", "Param Help"),
    Binding("ctrl+s", "save_editor", "Save"),
    Binding("escape", "cancel_editor", "Cancel"),
]

MODAL_CANCEL_BINDINGS = [
    Binding("escape", "cancel", "Cancel"),
]


class EditorInput(Input):
    """Input in model/preset editor — keeps editor hotkeys visible in the footer."""

    BINDINGS = EDITOR_BINDINGS

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("compact", True)
        super().__init__(*args, **kwargs)

    def _param_editor(self) -> ProfileEditor | PresetEditor | None:
        node = self.parent
        while node is not None:
            if isinstance(node, (ProfileEditor, PresetEditor)):
                return node
            node = node.parent
        return None

    def action_toggle_param_help(self) -> None:
        self.app.action_toggle_param_help()

    def action_save_editor(self) -> None:
        editor = self._param_editor()
        if editor:
            editor.save()

    def action_cancel_editor(self) -> None:
        editor = self._param_editor()
        if editor:
            editor.on_cancel_callback()


class ParamInput(EditorInput):
    """Input that notifies the app when focused (for F2 param help)."""

    def __init__(self, param_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_name = param_name

    def on_focus(self, event: Focus) -> None:
        self.post_message(ParamFocused(self.param_name))


class EditorSelect(Select):
    """Select in model/preset editor — keeps editor hotkeys visible in the footer."""

    BINDINGS = EDITOR_BINDINGS

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("compact", True)
        super().__init__(*args, **kwargs)

    def _param_editor(self) -> ProfileEditor | PresetEditor | None:
        node = self.parent
        while node is not None:
            if isinstance(node, (ProfileEditor, PresetEditor)):
                return node
            node = node.parent
        return None

    def action_toggle_param_help(self) -> None:
        self.app.action_toggle_param_help()

    def action_save_editor(self) -> None:
        editor = self._param_editor()
        if editor:
            editor.save()

    def action_cancel_editor(self) -> None:
        editor = self._param_editor()
        if editor:
            editor.on_cancel_callback()


class ParamSelect(EditorSelect):
    """Select that notifies the app when focused (for F2 param help)."""

    def __init__(self, param_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_name = param_name

    def on_focus(self, event: Focus) -> None:
        self.post_message(ParamFocused(self.param_name))


class ParamEditorMixin:
    """Shared compose/save helpers for model and preset parameter editors."""

    fields: dict[str, ParamInput | ParamSelect]

    def _yield_param_controls(
        self,
        param: str,
        value: object,
        *,
        label_class: str = "param-label",
        label: str | None = None,
    ):
        if param in HIDDEN_EDITOR_PARAMS:
            return
        if param in LOCKED_PARAMS:
            yield Label(f"{param}  [dim]from GGUF[/]", classes="param-label")
            yield Label(fmt_locked_value(value), classes="param-locked")
            return

        yield Label(label or param, classes=label_class)
        if is_select_param(param):
            initial = select_initial_value(param, value)
            select_kwargs: dict = {
                "options": select_options(param),
                "id": f"select_{param}",
                "prompt": param,
            }
            if param in PARAM_ALLOW_BLANK:
                select_kwargs["allow_blank"] = True
                select_kwargs["value"] = Select.NULL if initial is None else initial
            else:
                select_kwargs["value"] = initial
            widget = ParamSelect(param, **select_kwargs)
            self.fields[param] = widget
            yield widget
            return

        widget = ParamInput(param, value=str(value), placeholder=param, id=f"input_{param}")
        self.fields[param] = widget
        yield widget

    def _read_fields(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for param, widget in self.fields.items():
            values[param] = read_field_value(param, widget)
        return values

    def _first_focusable_field(self) -> ParamInput | ParamSelect | EditorInput | None:
        if getattr(self, "name_input", None):
            return self.name_input
        return next(iter(self.fields.values()), None)


class ConfirmDialog(ModalScreen[bool]):
    """Yes/No confirmation dialog."""

    BINDINGS = MODAL_CANCEL_BINDINGS

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.message)
            with ActionBar():
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class ParamHelpPanel(VerticalScroll):
    """Bottom-half help panel showing docs for the focused parameter."""

    DEFAULT_TEXT = (
        "[bold]Parameter Help[/]  [dim](F2 to close · Tab to a field)[/]\n\n"
        "Focus an input to see what it does and how changing it affects the server."
    )

    def compose(self) -> ComposeResult:
        yield Static(self.DEFAULT_TEXT, id="param-help-text")

    def on_mount(self) -> None:
        load_param_help()

    def show_param(self, param: str | None) -> None:
        text = self.query_one("#param-help-text", Static)
        if not param:
            text.update(self.DEFAULT_TEXT)
            return
        help_text = get_param_help(param)
        if help_text:
            text.update(f"[bold $accent]{param}[/]\n\n{help_text}")
        else:
            text.update(f"[bold $accent]{param}[/]\n\n[dim]No documentation found for this parameter.[/]")


class ProfileEditor(VerticalScroll):
    """Edit model profile identity (display name, port, host)."""

    BINDINGS = EDITOR_BINDINGS

    def __init__(self, model_name: str, params: dict, registry: Registry, on_save, on_cancel):
        super().__init__()
        self.model_name = model_name
        self.params = dict(params)
        self.registry = registry
        self.on_save_callback = on_save
        self.on_cancel_callback = on_cancel
        self.display_input: EditorInput | None = None
        self.port_input: EditorInput | None = None
        self.host_input: EditorInput | None = None
        self.notes_input: EditorInput | None = None

    def compose(self) -> ComposeResult:
        yield Label(
            f"[bold]Edit Profile: {self.params.get('display', self.model_name)}[/bold]  "
            "(Ctrl+S: save, Esc: cancel)"
        )
        yield Label("[dim]Display name is what you pass to llm-serve[/dim]")
        yield Label("Display name:", classes="field-label")
        self.display_input = EditorInput(
            value=str(self.params.get("display", "")),
            placeholder="Qwen 3.8",
            id="profile_display",
        )
        yield self.display_input
        yield Label("Port:", classes="field-label")
        self.port_input = EditorInput(value=str(self.params.get("port", 8081)), id="profile_port")
        yield self.port_input
        yield Label("Host:", classes="field-label")
        self.host_input = EditorInput(value=str(self.params.get("host", "127.0.0.1")), id="profile_host")
        yield self.host_input
        yield Label("Notes:", classes="field-label")
        self.notes_input = EditorInput(value=str(self.params.get("notes", "")), id="profile_notes")
        yield self.notes_input
        with ActionBar():
            yield Button("Save", variant="success", id="save")
            yield Button("Cancel", id="cancel")

    def save(self) -> None:
        display = self.display_input.value.strip() if self.display_input else ""
        error = validate_display_name(self.registry, display, exclude=self.model_name)
        if error:
            self.app.notify(error, severity="error")
            return
        updated = dict(self.params)
        updated["display"] = display
        updated["port"] = int(self.port_input.value.strip()) if self.port_input else updated.get("port", 8081)
        updated["host"] = self.host_input.value.strip() if self.host_input else updated.get("host", "127.0.0.1")
        updated["notes"] = self.notes_input.value.strip() if self.notes_input else ""
        self.on_save_callback(updated)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.on_cancel_callback()
        elif event.button.id == "save":
            self.save()

    def on_mount(self) -> None:
        if self.display_input:
            self.display_input.focus()

    def action_toggle_param_help(self) -> None:
        self.app.action_toggle_param_help()

    def action_save_editor(self) -> None:
        self.save()

    def action_cancel_editor(self) -> None:
        self.on_cancel_callback()

    def on_key(self, event) -> None:
        if event.key == "ctrl+s":
            event.prevent_default()
            self.save()
        elif event.key == "escape":
            event.prevent_default()
            self.on_cancel_callback()


class PresetEditor(ParamEditorMixin, VerticalScroll):
    """Editor panel for a preset (full llama-server params)."""

    BINDINGS = EDITOR_BINDINGS

    def __init__(
        self,
        model_name: str,
        slot: int,
        preset: Preset | None,
        identity: dict,
        *,
        is_new: bool = False,
        models_dir: Path,
        on_save,
        on_cancel,
    ):
        super().__init__()
        self.model_name = model_name
        self.slot = slot
        self.is_new = is_new
        if preset is not None:
            self.preset = preset
        else:
            self.preset = Preset(
                slot=slot,
                name=f"slot-{slot}",
                params=default_preset_params(),
            )
        self.identity = dict(identity)
        apply_architecture_from_gguf(
            self.identity, models_dir / str(self.identity.get("file", ""))
        )
        self.max_ctx = resolve_context_length(self.identity, models_dir)
        self.fields: dict[str, ParamInput | ParamSelect] = {}
        self.name_input: EditorInput | None = None
        self.on_save_callback = on_save
        self.on_cancel_callback = on_cancel

    def compose(self) -> ComposeResult:
        verb = "New" if self.is_new else "Edit"
        yield Label(
            f"[bold]{verb} Preset: {self.model_name} [{self.slot}][/bold]  "
            "(Ctrl+S: save, Esc: cancel, F2: help)"
        )
        yield Label("Preset name:", classes="field-label")
        self.name_input = EditorInput(value=self.preset.name, placeholder="preset-name", id="preset_name")
        yield self.name_input

        for group_name, param_names in PRESET_PARAM_GROUPS.items():
            with Collapsible(title=group_name, collapsed=False):
                for i in range(0, len(param_names), 3):
                    row_params = param_names[i : i + 3]
                    with Horizontal():
                        for param in row_params:
                            if param in LOCKED_PARAMS:
                                if param == "context_length":
                                    value = self.max_ctx if self.max_ctx is not None else self.identity.get(param, "")
                                else:
                                    value = self.identity.get(param, "")
                                with Vertical(classes="param-field"):
                                    yield Label(f"{param}  [dim]from model[/]", classes="param-label")
                                    yield Label(fmt_locked_value(value), classes="param-locked")
                                continue
                            value = self.preset.params.get(param, "")
                            if param == "ctx" and self.max_ctx is not None:
                                with Vertical(classes="param-field"):
                                    yield from self._yield_param_controls(
                                        param,
                                        value,
                                        label=f"ctx  [dim]max {fmt_ctx(self.max_ctx)}[/]",
                                    )
                                continue
                            with Vertical(classes="param-field"):
                                yield from self._yield_param_controls(param, value)

        with ActionBar():
            yield Button("Save", variant="success", id="save")
            yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.on_cancel_callback()
        elif event.button.id == "save":
            self.save()

    def save(self) -> None:
        name = self.name_input.value.strip() if self.name_input else self.preset.name
        if not name:
            name = f"slot-{self.slot}"

        params = {}
        for param, val in self._read_fields().items():
            if param in LOCKED_PARAMS:
                continue
            params[param] = val

        if self.max_ctx is not None and "ctx" in params:
            try:
                requested = int(params["ctx"])
            except (TypeError, ValueError):
                requested = None
            if requested is not None and requested > self.max_ctx:
                self.app.notify(
                    f"Context capped to model max ({fmt_ctx(self.max_ctx)} tokens)",
                    severity="warning",
                )
                params["ctx"] = self.max_ctx

        self.on_save_callback(name, params)

    def on_mount(self) -> None:
        first = self._first_focusable_field()
        if first:
            first.focus()

    def action_toggle_param_help(self) -> None:
        self.app.action_toggle_param_help()

    def action_save_editor(self) -> None:
        self.save()

    def action_cancel_editor(self) -> None:
        self.on_cancel_callback()

    def on_key(self, event) -> None:
        if event.key == "ctrl+s":
            event.prevent_default()
            self.save()
        elif event.key == "escape":
            event.prevent_default()
            self.on_cancel_callback()


class CreateAliasDialog(ModalScreen[tuple[str, str] | None]):
    """Dialog to create a new alias."""

    BINDINGS = MODAL_CANCEL_BINDINGS

    def __init__(self, registry: Registry):
        super().__init__()
        self.registry = registry

    def compose(self) -> ComposeResult:
        with Vertical(id="create-dialog"):
            yield Label("[bold]Create New Alias[/bold]")
            yield Label("")
            yield Label("Alias name:")
            yield Input(placeholder="fast", id="name")
            yield Label("")
            yield Label("Points to model:")
            options = [(name, name) for name in self.registry.models.keys()]
            yield Select(options, id="target", value=options[0][1] if options else None)
            yield Label("")
            with ActionBar():
                yield Button("Create", variant="success", id="create")
                yield Button("Cancel", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "create":
            name = self.query_one("#name", Input).value.strip()
            target = self.query_one("#target", Select).value
            if not name:
                self.app.notify("Name cannot be empty", severity="error")
                return
            if name in self.registry.aliases:
                self.app.notify(f"Alias '{name}' already exists", severity="error")
                return
            self.dismiss((name, target))


class EditAliasDialog(ModalScreen[str | None]):
    """Dialog to rename an existing alias."""

    BINDINGS = MODAL_CANCEL_BINDINGS

    def __init__(self, registry: Registry, alias_name: str):
        super().__init__()
        self.registry = registry
        self.alias_name = alias_name

    def compose(self) -> ComposeResult:
        with Vertical(id="create-dialog"):
            yield Label(f"[bold]Rename Alias: {self.alias_name}[/bold]")
            yield Label("")
            yield Label("Alias name:")
            yield Input(value=self.alias_name, placeholder="fast", id="name")
            yield Label("")
            with ActionBar():
                yield Button("Save", variant="success", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        name_input = self.query_one("#name", Input)
        name_input.focus()
        name_input.action_select_all()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "save":
            name = self.query_one("#name", Input).value.strip()
            if not name:
                self.app.notify("Name cannot be empty", severity="error")
                return
            if name == self.alias_name:
                self.dismiss(None)
                return
            if name in self.registry.aliases:
                self.app.notify(f"Alias '{name}' already exists", severity="error")
                return
            self.dismiss(name)
