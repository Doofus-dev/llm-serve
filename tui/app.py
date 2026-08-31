"""llm-serve TUI — Phase 1.5: config editor + JSON backend."""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Focus
from textual.reactive import reactive
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, RichLog, Static, Tree,
    Collapsible, Select
)
from textual.widgets.tree import TreeNode
from textual.screen import ModalScreen
from textual.message import Message

from tui.data.models_json import (
    Registry,
    ModelConfig,
    load_registry,
    save_registry,
    delete_model,
    create_model,
    update_model,
    merge_editor_params,
)
from tui.data.param_help import get_param_help, load_param_help
from tui.data.presets import PresetStore, Preset, load_presets, save_presets, get_preset, set_preset, delete_preset, apply_preset, overrides_to_env, get_active_slot, set_active_preset, clear_active_preset, MAX_PRESETS_PER_MODEL
from tui.data.settings import TUISettings, load_settings, save_settings
from tui.screens.hub import HubScreen
from tui.data.gpu import GPUStats, query_gpu
from tui.data.pidfile import PidInfo, read_pid_file
from tui.data.stats import Metrics, ServerClient

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = REPO_ROOT / "models.json"
MODELS_DIR = REPO_ROOT / "models"
PRESETS_JSON = REPO_ROOT / "presets.json"
TUI_SETTINGS_JSON = REPO_ROOT / "tui-settings.json"
MODELS_CONF_EXAMPLE = REPO_ROOT / "models.conf.example"
LOG_FILE = REPO_ROOT / "logs" / "llm-serve.log"
PID_FILE = REPO_ROOT / "logs" / ".llm-serve.pid"

# Parameter groups for the editor
PARAM_GROUPS = {
    "Model": ["file", "port", "host", "notes"],
    "Architecture": ["total_layers", "gpu_layers", "n_cpu_moe"],
    "Context": ["ctx", "per_slot_min", "cache_k", "cache_v"],
    "Batching": ["batch", "ubatch", "threads", "threads_batch", "parallel"],
    "Generation": ["n_predict", "defrag", "thinking", "reasoning_format", "reasoning_budget"],
    "Server": ["flash_attn", "checkpoint_every", "jinja", "timeout", "cache_prompt", "cache_reuse", "cont_batching", "ctx_checkpoints"],
    "Sampling": ["temp", "top_p", "top_k", "min_p", "seed", "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty"],
    "Speculative": ["mtp", "spec_draft_n_max", "spec_draft_n_min"],
    "Metrics": ["metrics"],
}

# All known params in order
ALL_PARAMS = []
for group_params in PARAM_GROUPS.values():
    ALL_PARAMS.extend(group_params)


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


class EditorInput(Input):
    """Input in model/preset editor — keeps editor hotkeys visible in the footer."""

    BINDINGS = EDITOR_BINDINGS

    def _param_editor(self) -> ModelEditor | PresetEditor | None:
        node = self.parent
        while node is not None:
            if isinstance(node, (ModelEditor, PresetEditor)):
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


def fmt_uptime(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def fmt_ctx(n) -> str:
    """Compact context size: 32768 → 32k."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def parse_gguf_filename(filename: str) -> tuple[str | None, str | None, str | None]:
    """Extract (params, variant, quant) from a GGUF filename.

    Examples:
      Qwen2.5-7B-Instruct-Q8_0.gguf        → 7B, Instruct, Q8_0
      Qwen3.6-27B-Q3_K_S.gguf              → 27B, None, Q3_K_S
      Qwen3.6-27B-UD-Q3_K_XL.gguf          → 27B, UD, Q3_K_XL
    """
    stem = Path(filename).stem

    params = None
    if m := re.search(r"(\d+(?:\.\d+)?[BMbm])", stem):
        params = m.group(1).upper()

    variant = None
    if re.search(r"[-_]UD[-_]", stem, re.I):
        variant = "UD"  # Unsloth Dynamic — mixed-layer quant
    elif re.search(r"Instruct", stem, re.I):
        variant = "Instruct"
    elif re.search(r"Chat", stem, re.I):
        variant = "Chat"

    quant = None
    if m := re.search(r"(Q\d+_K_[A-Z0-9]+|Q\d+_\d+|IQ\d+_[A-Z0-9]+|F16|BF16)", stem, re.I):
        quant = m.group(1).upper()

    return params, variant, quant


def model_author(params: dict, file: str) -> str | None:
    source = params.get("source")
    if isinstance(source, dict):
        author = source.get("author")
        if author:
            return str(author)
    if "/" in file:
        return file.split("/", 1)[0]
    return None


def model_file_exists(file: str) -> bool:
    if not file:
        return False
    return (MODELS_DIR / file).is_file()


def fmt_model_file_line(file: str, params: dict | None = None) -> str:
    """Line 1: author · params · variant · quant from filename/path."""
    parts: list[str] = []
    author = model_author(params or {}, file) if params is not None else None
    if author:
        parts.append(author)
    model_params, variant, quant = parse_gguf_filename(Path(file).name)
    if model_params:
        parts.append(model_params)
    if variant:
        parts.append(variant)
    if quant:
        parts.append(quant)
    return " · ".join(parts) if parts else "?"


def fmt_model_source_line(params: dict, file: str) -> str:
    """Line for Hub source or on-disk status."""
    source = params.get("source")
    if isinstance(source, dict) and source.get("repo"):
        repo = source.get("repo")
        return f"HF: {repo}"
    if model_file_exists(file):
        return "on disk"
    return "missing file"


def fmt_model_runtime_line(params: dict) -> str:
    """Line 2: context and GPU layers from config."""
    ctx = fmt_ctx(params.get("ctx", "?"))
    ngl = params.get("gpu_layers", "?")
    return f"ctx: {ctx}  ngl: {ngl}"


class ModelTree(Tree):
    """Left panel: models, presets, and aliases."""

    def __init__(self, registry: Registry, preset_store: PresetStore):
        super().__init__("Models")
        self.registry = registry
        self.preset_store = preset_store
        self.show_root = False

    def on_mount(self) -> None:
        self.refresh_tree()

    def refresh_tree(self) -> None:
        """Rebuild the tree from registry."""
        self.root.remove_children()
        aliases = self.registry.aliases
        for name, model in self.registry.models.items():
            node = self.root.add(name, data=("model", name))
            node.add_leaf(fmt_model_file_line(model.file, model.params))
            node.add_leaf(fmt_model_source_line(model.params, model.file))
            node.add_leaf(fmt_model_runtime_line(model.params))
            
            # Add presets
            if name in self.preset_store.presets:
                for slot in sorted(self.preset_store.presets[name].keys()):
                    preset = self.preset_store.presets[name][slot]
                    label = f"[{slot}] {preset.name}"
                    if self.preset_store.active.get(name) == slot:
                        label += " [ACTIVE]"
                    node.add_leaf(label, data=("preset", name, slot))
        
        if aliases:
            an = self.root.add("Aliases")
            for alias, target in aliases.items():
                an.add_leaf(f"{alias} → {target}", data=("alias", alias))
        self.root.expand_all()


class StatusPanel(Static):
    """Live status + throughput + GPU."""

    pid_info: reactive[PidInfo | None] = reactive(None)
    metrics: reactive[Metrics | None] = reactive(None)
    gpu: reactive[GPUStats | None] = reactive(None)
    props: reactive[dict | None] = reactive(None)
    uptime: reactive[float] = reactive(0.0)

    def render(self) -> str:
        lines: list[str] = []
        info = self.pid_info
        if info and info.alive:
            lines.append(f"[bold $success]● RUNNING[/]  [$accent]{info.model}[/]  (PID {info.pid}, port {info.port})")
            lines.append(f"Uptime: {fmt_uptime(self.uptime)}")
        else:
            lines.append("[bold $error]○ NOT RUNNING[/] — press [bold]L[/] to launch selected model")

        if self.props:
            alias = self.props.get("model_alias", "?")
            mp = self.props.get("model_path", "?")
            lines.append(f"Alias: {alias}  Path: {Path(str(mp)).name}")

        lines.append("")
        lines.append("[bold]── THROUGHPUT ──[/]")
        m = self.metrics
        if m:
            gen = m.predicted_tokens_seconds or m.gen_tps_derived
            prompt = m.prompt_tokens_seconds or m.prompt_tps_derived
            bar_len = min(int(gen / 5), 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"Generation: {gen:7.1f} tok/s  {bar}")
            lines.append(f"Prompt:     {prompt:7.1f} tok/s")
            lines.append(f"Total gen:  {int(m.tokens_predicted_total)} tokens   (avg {m.avg_gen_tps:.1f} tok/s)")
            lines.append(f"Total prompt: {int(m.prompt_tokens_total)} tokens (avg {m.avg_prompt_tps:.1f} tok/s)")
            if gen > 0:
                lines.append(f"ms/token:   {1000.0/gen:.0f}")
            lines.append(f"Requests:   {int(m.requests_processing)} processing, {int(m.requests_deferred)} deferred")
        else:
            lines.append("[dim]metrics unavailable (server not running, or no --metrics)[/]")

        lines.append("")
        lines.append("[bold]── GPU ──[/]")
        g = self.gpu
        if g and g.available:
            lines.append(g.name)
            lines.append(f"VRAM: {g.vram_used_mb/1024:.1f} / {g.vram_total_mb/1024:.1f} GB ({g.vram_pct:.0f}%)")
            lines.append(f"Util: {g.utilization_pct:.0f}%   Temp: {g.temp_c:.0f}°C")
        else:
            lines.append("[dim]GPU stats unavailable[/]")

        return "\n".join(lines)


class ConfigPanel(Static):
    """Active config of the selected model."""

    selected: reactive[str | None] = reactive(None)
    registry: Registry | None = None

    def render(self) -> str:
        lines = ["[bold]── ACTIVE CONFIG ──[/]"]
        if self.selected and self.registry and self.selected in self.registry.models:
            model = self.registry.models[self.selected]
            params = model.params
            author = model_author(params, model.file)
            if author:
                lines.append(f"[$accent]{'author':<18}[/] {author}")
            source = params.get("source")
            if isinstance(source, dict):
                for key in ("repo", "filename", "revision"):
                    if source.get(key):
                        lines.append(f"[$accent]{key:<18}[/] {source[key]}")
            file_status = "present" if model_file_exists(model.file) else "missing"
            lines.append(f"[$accent]{'file_status':<18}[/] {file_status}")
            lines.append("")
            for k, v in params.items():
                if k == "source":
                    continue
                val = str(v) if v != "" else "[dim]—[/]"
                lines.append(f"[$accent]{k:<18}[/] {val}")
        else:
            lines.append("[dim]select a model in the tree[/]")
        return "\n".join(lines)


class LogPanel(RichLog):
    """Tail of logs/llm-serve.log."""

    def tail_file(self, path: Path, n: int = 200) -> None:
        if not path.exists():
            self.write("[dim]no log file yet[/]")
            return
        try:
            out = subprocess.run(["tail", "-n", str(n), str(path)],
                                 capture_output=True, text=True, timeout=5).stdout
            self.clear()
            self.write(out.rstrip())
        except Exception as e:
            self.write(f"[$error]log read error: {e}[/]")


class ConfirmDialog(ModalScreen[bool]):
    """Yes/No confirmation dialog."""

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.message)
            with Horizontal():
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

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
        load_param_help(MODELS_CONF_EXAMPLE)

    def show_param(self, param: str | None) -> None:
        text = self.query_one("#param-help-text", Static)
        if not param:
            text.update(self.DEFAULT_TEXT)
            return
        help_text = get_param_help(param, MODELS_CONF_EXAMPLE)
        if help_text:
            text.update(f"[bold $accent]{param}[/]\n\n{help_text}")
        else:
            text.update(f"[bold $accent]{param}[/]\n\n[dim]No documentation found for this parameter.[/]")


class ModelEditor(VerticalScroll):
    """Editor panel for a model's parameters (takes over right side)."""

    BINDINGS = EDITOR_BINDINGS

    def __init__(self, name: str, params: dict, registry: Registry, on_save, on_cancel):
        super().__init__()
        self.model_name = name
        self.params = dict(params)
        self.registry = registry
        self.inputs: dict[str, ParamInput] = {}
        self.on_save_callback = on_save
        self.on_cancel_callback = on_cancel

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]Edit Model: {self.model_name}[/bold]  (Ctrl+S: save, Esc: cancel, F2: help)")
        yield Label("")
        
        for group_name, param_names in PARAM_GROUPS.items():
            with Collapsible(title=group_name, collapsed=False):
                # Group params into rows of 3
                for i in range(0, len(param_names), 3):
                    row_params = param_names[i:i+3]
                    with Horizontal():
                        for param in row_params:
                            value = self.params.get(param, "")
                            with Vertical(classes="param-field"):
                                yield Label(param, classes="param-label")
                                inp = ParamInput(param, value=str(value), placeholder=param, id=f"input_{param}")
                                self.inputs[param] = inp
                                yield inp
        
        yield Label("")
        with Horizontal():
            yield Button("Save", variant="success", id="save")
            yield Button("Cancel", variant="default", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.on_cancel_callback()
        elif event.button.id == "save":
            self.save()

    def save(self) -> None:
        """Collect values and call save callback."""
        new_params = {}
        for param, inp in self.inputs.items():
            val = inp.value.strip()
            # Try to convert to number
            if val and val.lstrip('-').replace('.', '').isdigit():
                try:
                    if '.' in val:
                        new_params[param] = float(val)
                    else:
                        new_params[param] = int(val)
                except ValueError:
                    new_params[param] = val
            else:
                new_params[param] = val
        new_params = merge_editor_params(self.params, new_params, set(ALL_PARAMS))
        
        self.on_save_callback(new_params)

    def on_mount(self) -> None:
        """Focus the first input when the editor mounts."""
        first_input = next(iter(self.inputs.values()), None)
        if first_input:
            first_input.focus()

    def action_toggle_param_help(self) -> None:
        self.app.action_toggle_param_help()

    def action_save_editor(self) -> None:
        self.save()

    def action_cancel_editor(self) -> None:
        self.on_cancel_callback()

    def on_key(self, event) -> None:
        """Handle Ctrl+S and Esc."""
        if event.key == "ctrl+s":
            event.prevent_default()
            self.save()
        elif event.key == "escape":
            event.prevent_default()
            self.on_cancel_callback()


class CreateModelDialog(ModalScreen[tuple[str, str] | None]):
    """Dialog to create a new model (name + clone from)."""

    def __init__(self, registry: Registry):
        super().__init__()
        self.registry = registry

    def compose(self) -> ComposeResult:
        with Vertical(id="create-dialog"):
            yield Label("[bold]Create New Model[/bold]")
            yield Label("")
            yield Label("Name:")
            yield Input(placeholder="my-model", id="name")
            yield Label("")
            yield Label("Clone from:")
            options = [(name, name) for name in self.registry.models.keys()]
            yield Select(options, id="clone_from", value=options[0][1] if options else None)
            yield Label("")
            with Horizontal():
                yield Button("Create", variant="success", id="create")
                yield Button("Cancel", variant="default", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "create":
            name = self.query_one("#name", Input).value.strip()
            clone_from = self.query_one("#clone_from", Select).value
            if not name:
                self.app.notify("Name cannot be empty", severity="error")
                return
            if name in self.registry.models:
                self.app.notify(f"Model '{name}' already exists", severity="error")
                return
            self.dismiss((name, clone_from))


class AliasEditor(VerticalScroll):
    """Editor panel for an alias (takes over right side)."""

    def __init__(self, alias_name: str, current_target: str, registry: Registry, on_save, on_cancel):
        super().__init__()
        self.alias_name = alias_name
        self.current_target = current_target
        self.registry = registry
        self.on_save_callback = on_save
        self.on_cancel_callback = on_cancel

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]Edit Alias: {self.alias_name}[/bold]  (Ctrl+S: save, Esc: cancel)")
        yield Label("")
        yield Label("Points to model:", classes="field-label")
        
        options = [(name, name) for name in self.registry.models.keys()]
        yield Select(options, id="target_model", value=self.current_target)
        
        yield Label("")
        with Horizontal():
            yield Button("Save", variant="success", id="save")
            yield Button("Cancel", variant="default", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.on_cancel_callback()
        elif event.button.id == "save":
            self.save()

    def save(self) -> None:
        target = self.query_one("#target_model", Select).value
        self.on_save_callback(target)

    def on_mount(self) -> None:
        """Focus the dropdown when the editor mounts."""
        self.query_one("#target_model", Select).focus()

    def on_key(self, event) -> None:
        if event.key == "ctrl+s":
            event.prevent_default()
            self.save()
        elif event.key == "escape":
            event.prevent_default()
            self.on_cancel_callback()


class PresetEditor(VerticalScroll):
    """Editor panel for a preset (takes over right side)."""

    BINDINGS = EDITOR_BINDINGS

    def __init__(self, model_name: str, slot: int, preset: Preset | None, base_params: dict, on_save, on_cancel):
        super().__init__()
        self.model_name = model_name
        self.slot = slot
        self.preset = preset or Preset(slot=slot, name=f"slot-{slot}", overrides={})
        self.base_params = base_params
        self.inputs: dict[str, ParamInput] = {}
        self.name_input: EditorInput | None = None
        self.on_save_callback = on_save
        self.on_cancel_callback = on_cancel

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]Edit Preset: {self.model_name} [{self.slot}][/bold]  (Ctrl+S: save, Esc: cancel, F2: help)")
        yield Label("")
        yield Label("Preset name:", classes="field-label")
        self.name_input = EditorInput(value=self.preset.name, placeholder="preset-name", id="preset_name")
        yield self.name_input
        yield Label("")
        yield Label("[dim]Params that differ from base model are highlighted[/dim]")
        yield Label("")
        
        for group_name, param_names in PARAM_GROUPS.items():
            with Collapsible(title=group_name, collapsed=False):
                # Group params into rows of 3
                for i in range(0, len(param_names), 3):
                    row_params = param_names[i:i+3]
                    with Horizontal():
                        for param in row_params:
                            base_value = self.base_params.get(param, "")
                            override_value = self.preset.overrides.get(param, base_value)
                            is_override = param in self.preset.overrides
                            
                            with Vertical(classes="param-field"):
                                if is_override:
                                    yield Label(f"{param}*", classes="param-label-override")
                                else:
                                    yield Label(param, classes="param-label")
                                inp = ParamInput(param, value=str(override_value), placeholder=param, id=f"input_{param}")
                                self.inputs[param] = inp
                                yield inp
        
        yield Label("")
        with Horizontal():
            yield Button("Save", variant="success", id="save")
            yield Button("Cancel", variant="default", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.on_cancel_callback()
        elif event.button.id == "save":
            self.save()

    def save(self) -> None:
        """Collect values and call save callback."""
        name = self.name_input.value.strip() if self.name_input else self.preset.name
        if not name:
            name = f"slot-{self.slot}"
        
        overrides = {}
        for param, inp in self.inputs.items():
            val = inp.value.strip()
            base_val = str(self.base_params.get(param, ""))
            
            # Only save if different from base
            if val != base_val:
                # Try to convert to number
                if val and val.lstrip('-').replace('.', '').isdigit():
                    try:
                        if '.' in val:
                            overrides[param] = float(val)
                        else:
                            overrides[param] = int(val)
                    except ValueError:
                        overrides[param] = val
                else:
                    overrides[param] = val
        
        self.on_save_callback(name, overrides)

    def on_mount(self) -> None:
        """Focus the name input when the editor mounts."""
        if self.name_input:
            self.name_input.focus()

    def action_toggle_param_help(self) -> None:
        self.app.action_toggle_param_help()

    def action_save_editor(self) -> None:
        self.save()

    def action_cancel_editor(self) -> None:
        self.on_cancel_callback()

    def on_key(self, event) -> None:
        """Handle Ctrl+S and Esc."""
        if event.key == "ctrl+s":
            event.prevent_default()
            self.save()
        elif event.key == "escape":
            event.prevent_default()
            self.on_cancel_callback()


class CreateAliasDialog(ModalScreen[tuple[str, str] | None]):
    """Dialog to create a new alias."""

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
            with Horizontal():
                yield Button("Create", variant="success", id="create")
                yield Button("Cancel", variant="default", id="cancel")

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


class LLMServeApp(App):
    TITLE = "llm-serve TUI"
    CSS = """
    Screen {
        background: $surface;
    }

    #main { height: 1fr; }
    #left { width: 32; border-right: solid $primary; }
    #right { layout: vertical; }
    #status { height: auto; max-height: 22; border-bottom: solid $secondary; padding: 0 1; }
    #config { height: 1fr; padding: 0 1; }
    #logs { height: 12; border-top: solid $secondary; }
    #editor-scroll { height: 1fr; }
    #param-help { height: 50%; border-top: solid $secondary; padding: 0 1; display: none; }
    #right.help-open #editor-scroll { height: 50%; }
    #right.help-open #param-help { display: block; }
    
    .param-field {
        width: 1fr;
        margin: 0 1;
        height: 4;
    }
    
    .param-field Label {
        height: 1;
        padding: 0;
        margin: 0;
    }
    
    .param-field Input {
        height: 3;
        margin: 0;
        padding: 0 1;
    }
    
    Collapsible {
        padding: 0;
        margin: 0;
        height: auto;
    }
    
    Collapsible > .collapsible--title {
        padding: 0;
        margin: 0;
        height: 1;
    }
    
    Collapsible > .collapsible--content {
        padding: 0;
        margin: 0;
    }
    
    Horizontal {
        height: auto;
    }
    
    #confirm-dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $error;
        padding: 1 2;
        align: center middle;
    }
    
    #create-dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        align: center middle;
    }
    
    Button { margin: 0 1; }
    Input { margin: 0 0 1 0; }

    .param-label {
        color: $accent;
    }

    .param-label-override {
        color: $warning;
    }

    .field-label {
        color: $accent;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "launch", "Launch"),
        Binding("s", "stop", "Stop"),
        Binding("e", "edit", "Edit"),
        Binding("n", "new", "New"),
        Binding("d", "delete", "Delete"),
        Binding("a", "apply", "Apply"),
        Binding("t", "change_theme", "Theme"),
        Binding("h", "open_hub", "Hub"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.registry = load_registry(MODELS_JSON)
        self.preset_store = load_presets(PRESETS_JSON)
        self.settings = load_settings(TUI_SETTINGS_JSON)
        self.client: ServerClient | None = None
        self._launch_time: float | None = None
        self._log_size: int = 0
        self._editor_mode: bool = False
        self._editor_widget: ModelEditor | PresetEditor | None = None
        self._help_panel: ParamHelpPanel | None = None
        self._help_visible: bool = False
        self._focused_param: str | None = None
        self._alias_editor_mode: bool = False
        self._alias_editor_widget: AliasEditor | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield ModelTree(self.registry, self.preset_store)
            with Vertical(id="right"):
                yield StatusPanel(id="status")
                yield ConfigPanel(id="config")
                yield LogPanel(id="logs")
        yield Footer()

    def _update_footer(self) -> None:
        """Update footer bindings based on mode."""
        if self._editor_mode or self._alias_editor_mode:
            bindings = [Binding("ctrl+s", "save_edit", "Save")]
            bindings.append(Binding("escape", "cancel_edit", "Cancel"))
            self.bindings = bindings
        else:
            self.bindings = [
                Binding("q", "quit", "Quit"),
                Binding("l", "launch", "Launch"),
                Binding("s", "stop", "Stop"),
                Binding("e", "edit", "Edit"),
                Binding("n", "new", "New"),
                Binding("d", "delete", "Delete"),
                Binding("a", "apply", "Apply"),
                Binding("t", "change_theme", "Theme"),
                Binding("h", "open_hub", "Hub"),
                Binding("f1", "help", "Help"),
            ]
        self.query_one(Footer).refresh()

    def on_mount(self) -> None:
        saved_theme = self.settings.theme
        if saved_theme and saved_theme in self.available_themes:
            self.theme = saved_theme
        tree = self.query_one(ModelTree)
        tree.focus()
        first = next(iter(self.registry.models), None)
        if first:
            self.query_one(ConfigPanel).selected = first
            self.query_one(ConfigPanel).registry = self.registry
        self._refresh_pid()
        self.set_interval(2.0, self._poll_metrics)
        self.set_interval(5.0, self._poll_gpu)
        self.set_interval(3.0, self._poll_log)
        self.query_one(LogPanel).tail_file(LOG_FILE)
        self._update_footer()

    def watch_theme(self, theme_name: str) -> None:
        """Persist theme choice and refresh panels that use theme colors."""
        if theme_name and theme_name in self.available_themes:
            self.settings.theme = theme_name
            save_settings(TUI_SETTINGS_JSON, self.settings)
        self.call_after_refresh(self._refresh_themed_widgets)

    def _refresh_themed_widgets(self) -> None:
        """Re-render custom panels so theme markup/CSS picks up the new palette."""
        if self._editor_mode or self._alias_editor_mode:
            return
        for selector in (StatusPanel, ConfigPanel):
            try:
                self.query_one(selector).refresh()
            except Exception:
                pass

    def _refresh_pid(self) -> None:
        info = read_pid_file(PID_FILE)
        if self._editor_mode or self._alias_editor_mode:
            return
        panel = self.query_one(StatusPanel)
        was_alive = panel.pid_info.alive if panel.pid_info else False
        alive = info.alive if info else False
        panel.pid_info = info if alive else None
        if alive and not was_alive:
            self._launch_time = time.time()
        if alive and info:
            if self.client is None or self.client.base != f"http://127.0.0.1:{info.port}":
                if self.client:
                    asyncio.ensure_future(self.client.close())
                self.client = ServerClient("127.0.0.1", info.port)
        panel.uptime = (time.time() - self._launch_time) if (alive and self._launch_time) else 0.0

    async def _poll_metrics(self) -> None:
        self._refresh_pid()
        panel = self.query_one(StatusPanel)
        if self.client and panel.pid_info:
            panel.metrics = await self.client.metrics()
            if panel.props is None:
                panel.props = await self.client.props()

    def _poll_gpu(self) -> None:
        self.query_one(StatusPanel).gpu = query_gpu()

    def _poll_log(self) -> None:
        try:
            size = LOG_FILE.stat().st_size
        except OSError:
            return
        if size != self._log_size:
            self._log_size = size
            self.query_one(LogPanel).tail_file(LOG_FILE)

    def _reload_registry(self) -> None:
        """Reload models.json and presets.json, refresh UI."""
        self.registry = load_registry(MODELS_JSON)
        self.preset_store = load_presets(PRESETS_JSON)
        tree = self.query_one(ModelTree)
        tree.registry = self.registry
        tree.preset_store = self.preset_store
        tree.refresh_tree()
        cfg = self.query_one(ConfigPanel)
        cfg.registry = self.registry

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        node: TreeNode = event.node
        cfg = self.query_one(ConfigPanel)
        data = node.data
        if data and data[0] == "model":
            cfg.selected = data[1]
        elif node.parent is not None and node.parent.data and node.parent.data[0] == "model":
            cfg.selected = node.parent.data[1]
        cfg.registry = self.registry

    def _selected_model(self) -> str | None:
        tree = self.query_one(ModelTree)
        node = tree.cursor_node
        if node is None:
            return None
        if node.data and node.data[0] == "model":
            return node.data[1]
        if node.data and node.data[0] == "alias":
            return self.registry.aliases.get(node.data[1])
        if node.parent and node.parent.data and node.parent.data[0] == "model":
            return node.parent.data[1]
        return None

    def action_launch(self) -> None:
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return
        
        # Check if we have an active preset for this model
        env_overrides = {}
        slot = get_active_slot(self.preset_store, model)
        if slot is not None:
            preset = get_preset(self.preset_store, model, slot)
            if preset:
                env_overrides = overrides_to_env(preset.overrides)
                self.notify(f"Launching {model} with preset [{slot}] {preset.name}")
        
        try:
            import os
            env = os.environ.copy()
            env.update(env_overrides)
            
            r = subprocess.run(
                [str(REPO_ROOT / "llm-serve"), model],
                capture_output=True, text=True, timeout=60, cwd=REPO_ROOT, env=env,
            )
            if r.returncode == 0:
                self.notify(f"Launched {model}")
                self._reload_registry()
            else:
                self.notify(f"Launch failed: {r.stderr.strip()[:200]}", severity="error")
        except Exception as e:
            self.notify(f"Launch error: {e}", severity="error")
        self._refresh_pid()

    def action_stop(self) -> None:
        try:
            r = subprocess.run([str(REPO_ROOT / "llm-serve"), "stop"],
                               capture_output=True, text=True, timeout=30, cwd=REPO_ROOT)
            self.notify("Stopped" if r.returncode == 0 else f"Stop: {r.stdout.strip()[:100]}")
        except Exception as e:
            self.notify(f"Stop error: {e}", severity="error")
        self._refresh_pid()

    def action_refresh(self) -> None:
        self._reload_registry()
        self._refresh_pid()
        self._poll_gpu()
        self._poll_log()
        self.notify("Refreshed")

    def on_param_focused(self, event: ParamFocused) -> None:
        self._focused_param = event.param
        if self._help_panel and self._help_visible:
            self._help_panel.show_param(event.param)

    def _enter_param_editor(self, editor: ModelEditor | PresetEditor) -> None:
        """Show model/preset editor with optional F2 help panel."""
        self.query_one("#status").display = False
        self.query_one("#config").display = False
        self.query_one("#logs").display = False

        editor.id = "editor-scroll"
        help_panel = ParamHelpPanel(id="param-help")

        right = self.query_one("#right")
        right.mount(editor)
        right.mount(help_panel)

        self._editor_widget = editor
        self._help_panel = help_panel
        self._help_visible = False
        self._focused_param = None
        self._editor_mode = True
        self._update_footer()

    def _current_focused_param(self) -> str | None:
        """Read the param name from whichever input is focused in the editor."""
        focused = self.focused
        if isinstance(focused, ParamInput):
            return focused.param_name
        return self._focused_param

    def action_toggle_param_help(self) -> None:
        if not self._editor_mode or not self._help_panel:
            return
        self._help_visible = not self._help_visible
        right = self.query_one("#right")
        if self._help_visible:
            right.add_class("help-open")
            param = self._current_focused_param()
            if param:
                self._focused_param = param
            self._help_panel.show_param(self._focused_param)
        else:
            right.remove_class("help-open")
        self._update_footer()

    def action_edit(self) -> None:
        if self._editor_mode or self._alias_editor_mode:
            self.notify("Already in edit mode", severity="warning")
            return
        
        tree = self.query_one(ModelTree)
        node = tree.cursor_node
        if node is None:
            self.notify("Select a model, preset, or alias first", severity="warning")
            return
        
        # Check if it's a preset
        if node.data and node.data[0] == "preset":
            _, model_name, slot = node.data
            self._edit_preset(model_name, slot)
            return
        
        # Check if it's an alias
        if node.data and node.data[0] == "alias":
            alias_name = node.data[1]
            target = self.registry.aliases.get(alias_name)
            if target:
                self._edit_alias(alias_name, target)
            return
        
        # Otherwise it's a model
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return
        if model not in self.registry.models:
            self.notify(f"Model '{model}' not found", severity="error")
            return
        
        cfg = self.registry.models[model]
        
        def on_save(new_params: dict) -> None:
            update_model(MODELS_JSON, model, new_params)
            self._reload_registry()
            self._exit_editor()
            self.notify(f"Saved {model}")
        
        def on_cancel() -> None:
            self._exit_editor()
            self.notify("Edit cancelled")

        editor = ModelEditor(model, cfg.params, self.registry, on_save, on_cancel)
        self._enter_param_editor(editor)

    def _edit_preset(self, model_name: str, slot: int) -> None:
        """Edit a preset."""
        base_params = self.registry.models[model_name].params
        preset = get_preset(self.preset_store, model_name, slot)
        
        def on_save(name: str, overrides: dict) -> None:
            set_preset(self.preset_store, model_name, slot, name, overrides)
            save_presets(PRESETS_JSON, self.preset_store)
            self._reload_registry()
            self._exit_editor()
            self.notify(f"Saved preset {model_name} [{slot}] {name}")
        
        def on_cancel() -> None:
            self._exit_editor()
            self.notify("Edit cancelled")

        editor = PresetEditor(model_name, slot, preset, base_params, on_save, on_cancel)
        self._enter_param_editor(editor)

    def _edit_alias(self, alias_name: str, current_target: str) -> None:
        """Edit an alias."""
        def on_save(new_target: str) -> None:
            self.registry.aliases[alias_name] = new_target
            save_registry(MODELS_JSON, self.registry)
            self._reload_registry()
            self._exit_alias_editor()
            self.notify(f"Saved alias {alias_name} → {new_target}")
        
        def on_cancel() -> None:
            self._exit_alias_editor()
            self.notify("Edit cancelled")
        
        # Hide status/config/logs, show editor
        self.query_one("#status").display = False
        self.query_one("#config").display = False
        self.query_one("#logs").display = False
        
        editor = AliasEditor(alias_name, current_target, self.registry, on_save, on_cancel)
        right = self.query_one("#right")
        right.mount(editor)
        self._alias_editor_widget = editor
        self._alias_editor_mode = True
        self._update_footer()

    def _exit_editor(self) -> None:
        """Exit editor mode and restore normal view."""
        if self._help_panel:
            self._help_panel.remove()
            self._help_panel = None
        if self._editor_widget:
            self._editor_widget.remove()
            self._editor_widget = None
        right = self.query_one("#right")
        right.remove_class("help-open")
        self._help_visible = False
        self._focused_param = None
        self.query_one("#status").display = True
        self.query_one("#config").display = True
        self.query_one("#logs").display = True
        self._editor_mode = False
        self._update_footer()
        self.query_one(ModelTree).focus()

    def _exit_alias_editor(self) -> None:
        """Exit alias editor mode and restore normal view."""
        if self._alias_editor_widget:
            self._alias_editor_widget.remove()
            self._alias_editor_widget = None
        self.query_one("#status").display = True
        self.query_one("#config").display = True
        self.query_one("#logs").display = True
        self._alias_editor_mode = False
        self._update_footer()
        self.query_one(ModelTree).focus()

    def action_save_edit(self) -> None:
        """Save current editor (called by Ctrl+S binding)."""
        if self._editor_widget:
            self._editor_widget.save()
        elif self._alias_editor_widget:
            self._alias_editor_widget.save()

    def action_cancel_edit(self) -> None:
        """Cancel current editor (called by Esc binding)."""
        if self._editor_widget:
            self._exit_editor()
            self.notify("Edit cancelled")
        elif self._alias_editor_widget:
            self._exit_alias_editor()
            self.notify("Edit cancelled")

    def action_new(self) -> None:
        # Check if an alias is selected
        tree = self.query_one(ModelTree)
        node = tree.cursor_node
        if node and node.data and node.data[0] == "alias":
            # Create new alias
            def handle_alias_result(result: tuple[str, str] | None) -> None:
                if result is not None:
                    name, target = result
                    self.registry.aliases[name] = target
                    save_registry(MODELS_JSON, self.registry)
                    self._reload_registry()
                    self.notify(f"Created alias {name} → {target}")
            
            self.push_screen(CreateAliasDialog(self.registry), handle_alias_result)
        else:
            # Create new model
            def handle_model_result(result: tuple[str, str] | None) -> None:
                if result is not None:
                    name, clone_from = result
                    base_params = dict(self.registry.models[clone_from].params)
                    create_model(MODELS_JSON, name, base_params)
                    self._reload_registry()
                    self.notify(f"Created {name} (cloned from {clone_from})")
            
            self.push_screen(CreateModelDialog(self.registry), handle_model_result)

    def action_delete(self) -> None:
        tree = self.query_one(ModelTree)
        node = tree.cursor_node
        if node is None:
            self.notify("Select a model, preset, or alias first", severity="warning")
            return
        
        # Check if it's a preset
        if node.data and node.data[0] == "preset":
            _, model_name, slot = node.data
            preset = get_preset(self.preset_store, model_name, slot)
            if not preset:
                self.notify("Preset not found", severity="error")
                return
            
            # Check if server is running with this preset
            pid_info = read_pid_file(PID_FILE)
            if pid_info and pid_info.alive and pid_info.model == model_name:
                # Check if this preset is active
                if self.preset_store.active.get(model_name) == slot:
                    self.notify(f"Cannot delete preset while server is running with it", severity="error")
                    return
            
            msg = f"Delete preset '{preset.name}' (slot {slot}) for {model_name}?"
            
            def handle_preset_confirm(confirmed: bool) -> None:
                if confirmed:
                    delete_preset(self.preset_store, model_name, slot)
                    if self.preset_store.active.get(model_name) == slot:
                        clear_active_preset(self.preset_store, model_name)
                    save_presets(PRESETS_JSON, self.preset_store)
                    self._reload_registry()
                    self.notify(f"Deleted preset {model_name} [{slot}]")
            
            self.push_screen(ConfirmDialog(msg), handle_preset_confirm)
            return
        
        # Check if it's an alias
        if node.data and node.data[0] == "alias":
            alias_name = node.data[1]
            msg = f"Delete alias '{alias_name}'?"
            
            def handle_alias_confirm(confirmed: bool) -> None:
                if confirmed:
                    del self.registry.aliases[alias_name]
                    save_registry(MODELS_JSON, self.registry)
                    self._reload_registry()
                    self.notify(f"Deleted alias {alias_name}")
            
            self.push_screen(ConfirmDialog(msg), handle_alias_confirm)
            return
        
        # Otherwise it's a model
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return
        
        # Check for aliases pointing to this model
        aliases_using = [a for a, t in self.registry.aliases.items() if t == model]
        msg = f"Delete model '{model}'?"
        if aliases_using:
            msg += f"\n\nWarning: aliases using this model: {', '.join(aliases_using)}"
        
        def handle_model_confirm(confirmed: bool) -> None:
            if confirmed:
                delete_model(MODELS_JSON, model)
                # Also delete all presets for this model
                if model in self.preset_store.presets:
                    del self.preset_store.presets[model]
                clear_active_preset(self.preset_store, model)
                save_presets(PRESETS_JSON, self.preset_store)
                self._reload_registry()
                self.notify(f"Deleted {model}")
        
        self.push_screen(ConfirmDialog(msg), handle_model_confirm)

    def action_apply(self) -> None:
        """Apply the selected preset (mark as active)."""
        tree = self.query_one(ModelTree)
        node = tree.cursor_node
        if node and node.data and node.data[0] == "preset":
            _, model_name, slot = node.data
            set_active_preset(self.preset_store, model_name, slot)
            save_presets(PRESETS_JSON, self.preset_store)
            self._reload_registry()
            preset = get_preset(self.preset_store, model_name, slot)
            self.notify(f"Applied preset {model_name} [{slot}] {preset.name if preset else ''}")
        else:
            self.notify("Select a preset first", severity="warning")

    def action_open_hub(self) -> None:
        if self._editor_mode or self._alias_editor_mode:
            self.notify("Close the editor first", severity="warning")
            return

        def on_complete() -> None:
            self.registry = load_registry(MODELS_JSON)
            self._reload_registry()

        self.push_screen(
            HubScreen(
                registry=self.registry,
                settings=self.settings,
                settings_path=TUI_SETTINGS_JSON,
                models_json_path=MODELS_JSON,
                models_dir=MODELS_DIR,
                on_complete=on_complete,
            )
        )

    def action_help(self) -> None:
        self.notify(
            "Tab: switch pane | ↑↓: navigate | L: launch | S: stop | E: edit | N: new | D: delete | A: apply preset | H: Hub download | T: theme | Q: quit",
            title="Help", timeout=10,
        )

    def notify(self, message: str, *, title: str = "", severity: str = "information",
               timeout: float | None = None, **kwargs):
        if timeout is not None:
            try:
                return super().notify(message, title=title, severity=severity,
                                      timeout=timeout, **kwargs)
            except TypeError:
                pass
        return super().notify(message, title=title, severity=severity, **kwargs)


def main() -> None:
    app = LLMServeApp()
    app.run()


if __name__ == "__main__":
    main()
