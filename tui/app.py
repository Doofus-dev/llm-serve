"""llm-serve TUI — Phase 1.5: config editor + JSON backend."""

from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, RichLog, Static, Tree,
    Collapsible, Select
)
from textual.widgets.tree import TreeNode
from textual.screen import ModalScreen
from textual.message import Message

from tui.data.models_json import Registry, ModelConfig, load_registry, save_registry, delete_model, create_model, update_model
from tui.data.gpu import GPUStats, query_gpu
from tui.data.pidfile import PidInfo, read_pid_file
from tui.data.stats import Metrics, ServerClient

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = REPO_ROOT / "models.json"
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


def fmt_uptime(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


class ModelTree(Tree):
    """Left panel: models and their aliases."""

    def __init__(self, registry: Registry):
        super().__init__("Models")
        self.registry = registry
        self.show_root = False

    def on_mount(self) -> None:
        self.refresh_tree()

    def refresh_tree(self) -> None:
        """Rebuild the tree from registry."""
        self.root.remove_children()
        aliases = self.registry.aliases
        for name, model in self.registry.models.items():
            node = self.root.add(name, data=("model", name))
            node.add_leaf(f"file: {model.file}")
            node.add_leaf(f"ctx: {model.params.get('ctx', '?')}  ngl: {model.params.get('gpu_layers', '?')}")
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
            lines.append(f"[bold green]● RUNNING[/]  [cyan]{info.model}[/]  (PID {info.pid}, port {info.port})")
            lines.append(f"Uptime: {fmt_uptime(self.uptime)}")
        else:
            lines.append("[bold red]○ NOT RUNNING[/] — press [bold]L[/] to launch selected model")

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
            params = self.registry.models[self.selected].params
            for k, v in params.items():
                val = str(v) if v != "" else "[dim]—[/]"
                lines.append(f"[cyan]{k:<18}[/] {val}")
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
            self.write(f"[red]log read error: {e}[/]")


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


class ModelEditor(VerticalScroll):
    """Editor panel for a model's parameters (takes over right side)."""

    def __init__(self, name: str, params: dict, registry: Registry, on_save, on_cancel):
        super().__init__()
        self.model_name = name
        self.params = dict(params)
        self.registry = registry
        self.inputs: dict[str, Input] = {}
        self.on_save_callback = on_save
        self.on_cancel_callback = on_cancel

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]Edit Model: {self.model_name}[/bold]  (Ctrl+S: save, Esc: cancel)")
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
                                yield Label(f"[cyan]{param}[/cyan]")
                                inp = Input(value=str(value), placeholder=param, id=f"input_{param}")
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
        
        self.on_save_callback(new_params)

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


class LLMServeApp(App):
    TITLE = "llm-serve TUI"
    CSS = """
    #main { height: 1fr; }
    #left { width: 32; border-right: solid $primary; }
    #right { layout: vertical; }
    #status { height: auto; max-height: 22; border-bottom: solid $secondary; padding: 0 1; }
    #config { height: 1fr; padding: 0 1; }
    #logs { height: 12; border-top: solid $secondary; }
    
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
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "launch", "Launch"),
        Binding("s", "stop", "Stop"),
        Binding("r", "refresh", "Refresh"),
        Binding("e", "edit", "Edit"),
        Binding("n", "new", "New"),
        Binding("d", "delete", "Delete"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.registry = load_registry(MODELS_JSON)
        self.client: ServerClient | None = None
        self._launch_time: float | None = None
        self._log_size: int = 0
        self._editor_mode: bool = False
        self._editor_widget: ModelEditor | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield ModelTree(self.registry)
            with Vertical(id="right"):
                yield StatusPanel(id="status")
                yield ConfigPanel(id="config")
                yield LogPanel(id="logs")
        yield Footer()

    def on_mount(self) -> None:
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

    def _refresh_pid(self) -> None:
        info = read_pid_file(PID_FILE)
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
        """Reload models.json and refresh UI."""
        self.registry = load_registry(MODELS_JSON)
        tree = self.query_one(ModelTree)
        tree.registry = self.registry
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
        try:
            r = subprocess.run(
                [str(REPO_ROOT / "llm-serve"), model],
                capture_output=True, text=True, timeout=60, cwd=REPO_ROOT,
            )
            if r.returncode == 0:
                self.notify(f"Launched {model}")
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

    def action_edit(self) -> None:
        if self._editor_mode:
            self.notify("Already in edit mode", severity="warning")
            return
        
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
        
        # Hide status/config/logs, show editor
        self.query_one("#status").display = False
        self.query_one("#config").display = False
        self.query_one("#logs").display = False
        
        editor = ModelEditor(model, cfg.params, self.registry, on_save, on_cancel)
        right = self.query_one("#right")
        right.mount(editor)
        self._editor_widget = editor
        self._editor_mode = True
        editor.focus()

    def _exit_editor(self) -> None:
        """Exit editor mode and restore normal view."""
        if self._editor_widget:
            self._editor_widget.remove()
            self._editor_widget = None
        self.query_one("#status").display = True
        self.query_one("#config").display = True
        self.query_one("#logs").display = True
        self._editor_mode = False
        self.query_one(ModelTree).focus()

    def action_new(self) -> None:
        def handle_result(result: tuple[str, str] | None) -> None:
            if result is not None:
                name, clone_from = result
                base_params = dict(self.registry.models[clone_from].params)
                create_model(MODELS_JSON, name, base_params)
                self._reload_registry()
                self.notify(f"Created {name} (cloned from {clone_from})")
        
        self.push_screen(CreateModelDialog(self.registry), handle_result)

    def action_delete(self) -> None:
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return
        
        # Check for aliases pointing to this model
        aliases_using = [a for a, t in self.registry.aliases.items() if t == model]
        msg = f"Delete model '{model}'?"
        if aliases_using:
            msg += f"\n\nWarning: aliases using this model: {', '.join(aliases_using)}"
        
        def handle_result(confirmed: bool) -> None:
            if confirmed:
                delete_model(MODELS_JSON, model)
                self._reload_registry()
                self.notify(f"Deleted {model}")
        
        self.push_screen(ConfirmDialog(msg), handle_result)

    def action_help(self) -> None:
        self.notify(
            "Tab: switch pane | ↑↓: navigate | L: launch | S: stop | E: edit | N: new | D: delete | R: refresh | Q: quit",
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
