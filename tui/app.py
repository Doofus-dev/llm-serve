"""llm-serve TUI — Phase 1: read-only live monitor."""

from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, RichLog, Static, Tree
from textual.widgets.tree import TreeNode

from tui.data.config_parser import Registry, parse_models_conf
from tui.data.gpu import GPUStats, query_gpu
from tui.data.pidfile import PidInfo, read_pid_file
from tui.data.stats import Metrics, ServerClient

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_CONF = REPO_ROOT / "models.conf"
LOG_FILE = REPO_ROOT / "logs" / "llm-serve.log"
PID_FILE = REPO_ROOT / "logs" / ".llm-serve.pid"

# Parameters to skip in the config table (not llama-server config / internal)
_SKIP_KEYS = {"notes", "tool_use_enforcement", "compression_enabled",
              "compression_threshold", "api_max_retries", "sync_aux"}


def fmt_uptime(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


class ModelTree(Tree):
    """Left panel: models and their aliases."""

    def __init__(self, registry: Registry):
        super().__init__("Models")
        self.registry = registry
        self.show_root = False

    def on_mount(self) -> None:
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
                if k in _SKIP_KEYS:
                    continue
                val = v if v != "" else "[dim]—[/]"
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
        except Exception as e:  # noqa: BLE001
            self.write(f"[red]log read error: {e}[/]")


class LLMServeApp(App):
    TITLE = "llm-serve TUI"
    CSS = """
    #main { height: 1fr; }
    #left { width: 32; border-right: solid $primary; }
    #right { layout: vertical; }
    #status { height: auto; max-height: 22; border-bottom: solid $secondary; padding: 0 1; }
    #config { height: 1fr; padding: 0 1; }
    #logs { height: 12; border-top: solid $secondary; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "launch", "Launch"),
        Binding("s", "stop", "Stop"),
        Binding("r", "refresh", "Refresh"),
        Binding("f1", "help", "Help"),
    ]

    def action_help(self) -> None:
        self.notify(
            "Tab: switch pane | ↑↓: navigate | L: launch | S: stop | R: refresh | Q: quit",
            title="Help", timeout=10,
        )

    def notify(self, message: str, *, title: str = "", severity: str = "information",
               timeout: float | None = None, **kwargs):
        # Older Textual: no timeout kwarg — drop it.
        if timeout is not None:
            try:
                return super().notify(message, title=title, severity=severity,
                                      timeout=timeout, **kwargs)
            except TypeError:
                pass
        return super().notify(message, title=title, severity=severity, **kwargs)

    def __init__(self):
        super().__init__()
        self.registry = parse_models_conf(MODELS_CONF)
        self.client: ServerClient | None = None
        self._launch_time: float | None = None
        self._log_size: int = 0

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
                [str(REPO_ROOT / "llm-serve"), model, "--no-hermes"],
                capture_output=True, text=True, timeout=60, cwd=REPO_ROOT,
            )
            if r.returncode == 0:
                self.notify(f"Launched {model}")
            else:
                self.notify(f"Launch failed: {r.stderr.strip()[:200]}", severity="error")
        except Exception as e:  # noqa: BLE001
            self.notify(f"Launch error: {e}", severity="error")
        self._refresh_pid()

    def action_stop(self) -> None:
        try:
            r = subprocess.run([str(REPO_ROOT / "llm-serve"), "stop"],
                               capture_output=True, text=True, timeout=30, cwd=REPO_ROOT)
            self.notify("Stopped" if r.returncode == 0 else f"Stop: {r.stdout.strip()[:100]}")
        except Exception as e:  # noqa: BLE001
            self.notify(f"Stop error: {e}", severity="error")
        self._refresh_pid()

    def action_refresh(self) -> None:
        self._refresh_pid()
        self._poll_gpu()
        self._poll_log()


def main() -> None:
    app = LLMServeApp()
    app.run()


if __name__ == "__main__":
    main()
