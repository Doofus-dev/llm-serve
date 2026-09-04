"""llm-serve TUI — Phase 1.5: config editor + JSON backend."""

from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Focus
from textual.reactive import reactive
from textual import work
from textual.widgets import (
    Button, Collapsible, Footer, Header, Input, Label, OptionList,
    ProgressBar, RichLog, Select, Static,
)
from textual.widgets.option_list import Option
from textual.screen import ModalScreen
from textual.message import Message

from tui.data.models_json import (
    AliasTarget,
    Registry,
    load_registry,
    save_registry,
    delete_model,
    unshared_model_file_paths,
    update_model,
    sync_gguf_architecture,
    clamp_preset_contexts,
    set_active_quant,
    add_or_update_quant,
    model_author_size_line,
    validate_display_name,
)
from tui.data.context_length import (
    fmt_ctx_compact,
    fmt_ctx_range,
    resolve_context_length,
)
from tui.data.preset_template import (
    PRESET_PARAM_GROUPS,
    default_preset_params,
)
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
from tui.data.presets import (
    PresetStore,
    Preset,
    load_presets,
    save_presets,
    get_preset,
    set_preset,
    delete_preset,
    merge_identity_and_preset,
    get_active_slot,
    set_active_preset,
    clear_active_preset,
    delete_all_presets_for_model,
    MAX_PRESETS_PER_MODEL,
    next_free_slot,
    list_presets_for_quant,
)
from tui.data.settings import load_settings, save_settings
from tui.screens.hub import HubScreen
from tui.screens.quant_picker import QuantPickerScreen
from tui.data.downloads import DownloadManager, DownloadJob
from tui.data.quant import quant_from_filename
from tui.data.hf import build_download_plan, build_source_metadata, fmt_size
from tui.data.baselines import RunBaseline, record_baseline
from tui.data.gguf import apply_architecture_from_gguf
from tui.data.gpu import GPUStats, query_gpu
from tui.data.pidfile import PidInfo, read_pid_file
from tui.data.stats import Metrics, ServerClient
from tui.data.throughput_history import (
    LiveThroughput,
    SPARKLINE_WIDTH,
    ThroughputHistory,
    ThroughputReader,
    format_avg_line,
    format_last_request,
    live_tps_from_metrics,
    render_prefill_progress,
    render_stage_strip,
    render_tps_sparkline,
    sample_tps_for_history,
)

METRICS_POLL_INTERVAL = 0.5
METRICS_HISTORY_SAMPLES = 120  # ~60s window at 0.5s polling

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = REPO_ROOT / "models.json"
MODELS_DIR = REPO_ROOT / "models"
PRESETS_JSON = REPO_ROOT / "presets.json"
TUI_SETTINGS_JSON = REPO_ROOT / "tui-settings.json"
TUI_BASELINES_JSON = REPO_ROOT / "tui-baselines.json"
LOG_FILE = REPO_ROOT / "logs" / "llm-serve.log"
PID_FILE = REPO_ROOT / "logs" / ".llm-serve.pid"

PROFILE_KEYS = ("display", "port", "host", "notes")

# Architecture facts from the GGUF header — shown but not editable.
LOCKED_PARAMS = frozenset({"total_layers", "context_length"})


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


def fmt_uptime(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def fmt_ctx(n) -> str:
    """Compact context size: 32768 → 32k."""
    return fmt_ctx_compact(n)


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


def fmt_model_runtime_line(params: dict) -> str:
    """Line 2: preset context vs model max, and GPU layers."""
    max_ctx = resolve_context_length(params, MODELS_DIR)
    ctx = fmt_ctx_range(params.get("ctx", "?"), max_ctx)
    ngl = params.get("gpu_layers", "?")
    return f"ctx: {ctx}  ngl: {ngl}"


def vram_health(percent: float) -> tuple[str, str]:
    """Return a concise VRAM pressure label and Rich style."""
    if percent >= 90:
        return "CRITICAL", "bold red"
    if percent >= 75:
        return "HIGH", "bold yellow"
    return "OK", "bold green"


def temperature_health(temp_c: float) -> tuple[str, str]:
    """Return a conservative GPU temperature label and Rich style."""
    if temp_c >= 85:
        return "HOT", "bold red"
    if temp_c >= 70:
        return "WARM", "bold yellow"
    return "COOL", "bold green"


def generation_health(tokens_per_second: float) -> tuple[str, str]:
    """Return a broad generation-speed indicator."""
    if tokens_per_second >= 20:
        return "FAST", "bold green"
    if tokens_per_second >= 5:
        return "MODERATE", "bold yellow"
    if tokens_per_second > 0:
        return "SLOW", "bold red"
    return "IDLE", "dim"


class DownloadBar(Vertical):
    """Progress strip shown only while a Hub download is running."""

    DEFAULT_CSS = """
    DownloadBar {
        height: 0;
        overflow: hidden;
        padding: 0;
        border: none;
    }

    DownloadBar.visible {
        height: 3;
        padding: 0 1;
        border-bottom: solid $warning;
        background: $surface-darken-1;
    }

    DownloadBar #download-label {
        height: 1;
        color: $warning;
    }

    DownloadBar ProgressBar {
        height: 1;
        width: 1fr;
    }
    """

    def apply_state(self, state) -> None:
        """Update visibility and progress from DownloadState."""
        if state.running:
            self.add_class("visible")
            label = self.query_one("#download-label", Label)
            label.update(state.status_line or f"Downloading {state.filename}…")
            bar = self.query_one("#download-progress", ProgressBar)
            total = state.progress_total
            if total:
                bar.update(total=float(total), progress=float(state.used_bytes))
            else:
                bar.update(total=None, progress=0.0)
        else:
            self.remove_class("visible")
            self.query_one("#download-progress", ProgressBar).update(total=None)

    def compose(self) -> ComposeResult:
        yield Label("", id="download-label")
        yield ProgressBar(total=None, id="download-progress", show_eta=False)


class ModelNav(OptionList):
    """Card-based model navigation with directly selectable presets."""

    def __init__(self, registry: Registry, preset_store: PresetStore):
        super().__init__(id="model-nav")
        self.registry = registry
        self.preset_store = preset_store
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
            card.append(fmt_model_runtime_line(runtime_params), style="dim")
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


class StatusPanel(Static):
    """Live status + throughput + GPU."""

    pid_info: reactive[PidInfo | None] = reactive(None)
    model_display: reactive[str | None] = reactive(None)
    preset_display: reactive[str | None] = reactive(None)
    next_remote: reactive[bool] = reactive(False)
    metrics: reactive[Metrics | None] = reactive(None)
    live_throughput: reactive[LiveThroughput | None] = reactive(None)
    gen_tps_history: reactive[list[float]] = reactive([])
    gpu: reactive[GPUStats | None] = reactive(None)
    props: reactive[dict | None] = reactive(None)
    uptime: reactive[float] = reactive(0.0)

    def render(self) -> Group:
        info = self.pid_info
        if info and info.alive:
            state = Text("● RUNNING", style="bold green")
            state.append(f"  {self.model_display or info.model}", style="bold cyan")
            if self.preset_display:
                state.append(f"  {self.preset_display}", style="bold yellow")
            if info.remote:
                state.append("  REMOTE", style="bold magenta")
        else:
            state = Text("○ NOT RUNNING", style="bold red")
            state.append("  Press L to launch the selected model", style="dim")

        launch_flag = Text("NEXT LAUNCH ", style="dim")
        if self.next_remote:
            launch_flag.append("[REMOTE]", style="bold magenta")
        else:
            launch_flag.append("[LOCAL]", style="bold green")

        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        header.add_row(state, launch_flag)

        renderables: list = [header]
        if info and info.alive:
            renderables.append(
                Text(
                    f"port {info.port}  •  PID {info.pid}  •  up {fmt_uptime(self.uptime)}",
                    style="dim",
                )
            )
        if self.props:
            alias = self.props.get("model_alias", "?")
            mp = self.props.get("model_path", "?")
            renderables.append(
                Text(f"alias {alias}  •  {Path(str(mp)).name}", style="dim")
            )

        throughput: list[Text] = []
        m = self.metrics
        if m:
            live = self.live_throughput or live_tps_from_metrics(m)
            gen = live.gen_tps
            history = self.gen_tps_history
            throughput.append(render_stage_strip(live))
            if live.stage == "prefill":
                throughput.append(render_prefill_progress(live))
            elif live.stage == "generating":
                gen_label, gen_style = generation_health(gen)
                speed = Text()
                speed.append(f"{gen:.1f} t/s", style=gen_style)
                speed.append(" generation  ", style="dim")
                speed.append(gen_label, style=gen_style)
                if gen > 0:
                    speed.append(f"  {(1000.0 / gen):.1f} ms/token", style="dim")
                throughput.append(speed)
            elif live.stage == "queued":
                throughput.append(Text("waiting for a free slot", style="bold yellow"))
            else:
                recap = format_last_request(live.last_request)
                throughput.append(recap if recap is not None else Text("idle", style="dim"))
            throughput.append(
                render_tps_sparkline(history, width=SPARKLINE_WIDTH)
                if history
                else Text(" " * SPARKLINE_WIDTH, style="dim", no_wrap=True)
            )
            avg_line = format_avg_line(history, METRICS_POLL_INTERVAL)
            if avg_line is None:
                avg_line = Text("avg —", style="dim")
                if live.stage == "prefill":
                    avg_line.append("  (waiting on generate)", style="dim")
            throughput.append(avg_line)
        else:
            throughput.append(
                Text("Metrics unavailable", style="dim")
            )

        gpu_lines: list[Text] = []
        g = self.gpu
        if g and g.available:
            gpu_lines.append(Text(g.name, style="bold"))
            vram_label, vram_style = vram_health(g.vram_pct)
            memory = Text(f"{g.memory_label} ")
            memory.append(
                f"{g.vram_used_mb/1024:.1f} / {g.vram_total_mb/1024:.1f} GB  "
                f"({g.vram_pct:.0f}%) {vram_label}",
                style=vram_style,
            )
            gpu_lines.append(memory)

            thermals = Text(f"{g.utilization_pct:.0f}% utilization  •  ")
            if g.temp_c > 0:
                temp_label, temp_style = temperature_health(g.temp_c)
                thermals.append(f"{g.temp_c:.0f}°C {temp_label}", style=temp_style)
            else:
                thermals.append("temperature unavailable", style="dim")
            gpu_lines.append(thermals)
            if g.unified and g.dedicated_total_mb:
                gpu_lines.append(
                    Text(
                        f"VRAM BAR {g.dedicated_used_mb/1024:.1f} / "
                        f"{g.dedicated_total_mb/1024:.1f} GB",
                        style="dim",
                    )
                )
        else:
            gpu_lines.append(Text("GPU stats unavailable", style="dim"))

        telemetry = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
            show_edge=False,
        )
        telemetry.add_column("THROUGHPUT", ratio=1, style="white")
        telemetry.add_column("GPU", ratio=1, style="white")
        telemetry.add_row(Group(*throughput), Group(*gpu_lines))
        renderables.append(telemetry)
        return Group(*renderables)


class ConfigPanel(Static):
    """Active preset config for the selected model."""

    selected: reactive[str | None] = reactive(None)
    registry: Registry | None = None
    preset_store: PresetStore | None = None

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
            max_ctx = resolve_context_length(identity, MODELS_DIR)
            file_status = "present" if model_file_exists(model.file) else "missing"

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
        with Horizontal():
            yield Button("Save", variant="success", id="save")
            yield Button("Cancel", variant="default", id="cancel")

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
            self.identity, MODELS_DIR / str(self.identity.get("file", ""))
        )
        self.max_ctx = resolve_context_length(self.identity, MODELS_DIR)
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

        with Horizontal():
            yield Button("Save", variant="success", id="save")
            yield Button("Cancel", variant="default", id="cancel")

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
    TITLE = "llm-serve"
    CSS = """
    Screen {
        background: $surface;
    }

    #app-body {
        height: 1fr;
        layout: vertical;
    }

    #main { height: 1fr; }
    #left {
        width: 40;
        border-right: solid $primary;
        background: $surface-darken-1;
    }
    #model-nav {
        height: 1fr;
        border: none;
        padding: 1;
        background: $surface-darken-1;
    }
    #model-nav > .option-list--option {
        padding: 0 1;
        background: $surface;
    }
    #model-nav > .option-list--option-highlighted {
        background: $primary-background;
        color: $foreground;
        text-style: none;
    }
    #model-nav > .option-list--separator {
        color: $surface-darken-1;
    }
    .nav-heading {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        text-style: bold;
        background: $surface-darken-1;
    }
    #alias-nav {
        height: auto;
        max-height: 8;
        min-height: 3;
        border: none;
        padding: 0 1 1 1;
        background: $surface-darken-1;
    }
    #alias-nav > .option-list--option {
        padding: 0 1;
        background: $surface;
    }
    #alias-nav > .option-list--option-highlighted {
        background: $primary-background;
        color: $foreground;
        text-style: none;
    }
    #right { layout: vertical; }
    #status {
        height: 13;
        min-height: 9;
        border-bottom: solid $secondary;
        padding: 0 1;
    }
    #config {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }
    #logs {
        height: 10;
        min-height: 6;
        border-top: solid $secondary;
    }
    #editor-scroll { height: 1fr; padding: 0 1; }
    #editor-scroll > Label {
        height: 1;
        margin: 0;
        padding: 0;
    }
    #param-help { height: 50%; border-top: solid $secondary; padding: 0 1; display: none; }
    #right.help-open #editor-scroll { height: 50%; }
    #right.help-open #param-help { display: block; }
    
    .param-field {
        width: 1fr;
        margin: 0 1;
        height: auto;
    }
    
    .param-field Label {
        height: 1;
        padding: 0;
        margin: 0;
    }
    
    .param-field Input,
    .param-field Select {
        margin: 0;
        padding: 0 1;
    }

    .param-field Label.param-locked {
        height: 1;
        width: 1fr;
        margin: 0;
        padding: 0 1;
        background: $surface-darken-1;
        color: $text-muted;
        content-align: left middle;
    }

    .param-field Select {
        height: 1;
        width: 1fr;
        padding: 0;
    }

    .param-field Select > SelectCurrent {
        height: 1;
        width: 1fr;
        padding: 0 1;
        border: none;
        background: $surface;
    }

    .param-field Select SelectCurrent Static#label {
        height: 1;
        width: 1fr;
        padding: 0;
    }

    .param-field Select SelectCurrent .arrow {
        height: 1;
        padding: 0;
    }

    .param-field Select:focus > SelectCurrent {
        background-tint: $foreground 15%;
        border: none;
    }

    .param-field Select:focus SelectCurrent Static#label,
    .param-field Select:focus SelectCurrent.-has-value Static#label {
        color: $accent;
    }

    .param-field Select:focus SelectCurrent .arrow {
        color: $accent;
    }

    .param-field Input:focus {
        background-tint: $foreground 15%;
    }

    #editor-scroll Input {
        margin: 0;
        padding: 0 1;
    }

    #editor-scroll Select {
        height: 1;
        margin: 0;
        padding: 0;
    }

    #editor-scroll Select > SelectCurrent {
        height: 1;
        padding: 0 1;
        border: none;
        background: $surface;
    }

    #editor-scroll Input:focus {
        background-tint: $foreground 15%;
    }

    #editor-scroll Select:focus > SelectCurrent {
        background-tint: $foreground 15%;
        border: none;
    }

    #editor-scroll Select:focus SelectCurrent Static#label,
    #editor-scroll Select:focus SelectCurrent.-has-value Static#label {
        color: $accent;
    }

    #editor-scroll Select:focus SelectCurrent .arrow {
        color: $accent;
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
    Input { margin: 0; }

    .param-label {
        color: $accent;
    }

    .field-label {
        color: $accent;
        height: 1;
        margin: 0;
        padding: 0;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "launch", "Launch"),
        Binding("s", "stop", "Stop"),
        Binding("e", "edit", "Edit"),
        Binding("n", "new", "New"),
        Binding("d", "delete", "Delete"),
        Binding("t", "change_theme", "Theme"),
        Binding("h", "open_hub", "Hub"),
        Binding("p", "pick_quant", "Quant"),
        Binding("r", "toggle_remote", "Remote OFF"),
        Binding("1", "activate_preset(1)", "Preset 1", show=False),
        Binding("2", "activate_preset(2)", "Preset 2", show=False),
        Binding("3", "activate_preset(3)", "Preset 3", show=False),
        Binding("4", "activate_preset(4)", "Preset 4", show=False),
        Binding("5", "activate_preset(5)", "Preset 5", show=False),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.registry = load_registry(MODELS_JSON, models_dir=MODELS_DIR)
        self.preset_store = load_presets(PRESETS_JSON)
        self.settings = load_settings(TUI_SETTINGS_JSON)
        self.download_manager = DownloadManager()
        self.client: ServerClient | None = None
        self.remote_launch: bool = False
        self._launch_time: float | None = None
        self._log_size: int = 0
        self._editor_mode: bool = False
        self._editor_widget: ProfileEditor | PresetEditor | None = None
        self._help_panel: ParamHelpPanel | None = None
        self._help_visible: bool = False
        self._focused_param: str | None = None
        self._gen_history = ThroughputHistory(max_samples=METRICS_HISTORY_SAMPLES)
        self._throughput_reader = ThroughputReader()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-body"):
            yield DownloadBar(id="download-bar")
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    yield Label("MODELS", classes="nav-heading")
                    yield ModelNav(self.registry, self.preset_store)
                    yield Label("ALIASES  Tab · 1-5 pin", classes="nav-heading")
                    yield AliasNav(self.registry)
                with Vertical(id="right"):
                    yield StatusPanel(id="status")
                    yield ConfigPanel(id="config")
                    yield LogPanel(id="logs")
        yield Footer()

    def _update_footer(self) -> None:
        """Update footer bindings based on mode."""
        if self._editor_mode:
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
                Binding("t", "change_theme", "Theme"),
                Binding("h", "open_hub", "Hub"),
                Binding("p", "pick_quant", "Quant"),
                Binding(
                    "r",
                    "toggle_remote",
                    "Remote ON" if self.remote_launch else "Remote OFF",
                ),
                Binding("1", "activate_preset(1)", "Preset 1", show=False),
                Binding("2", "activate_preset(2)", "Preset 2", show=False),
                Binding("3", "activate_preset(3)", "Preset 3", show=False),
                Binding("4", "activate_preset(4)", "Preset 4", show=False),
                Binding("5", "activate_preset(5)", "Preset 5", show=False),
                Binding("f1", "help", "Help"),
            ]
        self.query_one(Footer).refresh()

    def on_mount(self) -> None:
        saved_theme = self.settings.theme
        if saved_theme and saved_theme in self.available_themes:
            self.theme = saved_theme
        nav = self.query_one(ModelNav)
        nav.focus()
        first = next(iter(self.registry.models), None)
        if first:
            cfg = self.query_one(ConfigPanel)
            cfg.selected = first
            cfg.registry = self.registry
            cfg.preset_store = self.preset_store
        self._refresh_pid()
        self.set_interval(METRICS_POLL_INTERVAL, self._poll_metrics)
        self.set_interval(5.0, self._poll_gpu)
        self.set_interval(3.0, self._poll_log)
        self.query_one(LogPanel).tail_file(LOG_FILE)
        self._update_footer()
        self.download_manager.subscribe(self._on_download_state)
        self._reload_registry(notify_gguf=True)

    def _on_download_state(self, state) -> None:
        """Apply download progress on the UI thread."""

        def apply() -> None:
            try:
                self.query_one("#download-bar", DownloadBar).apply_state(state)
            except Exception:
                pass

        self.call_later(apply)

    @work(exclusive=True)
    async def _run_download_job(self, job: DownloadJob) -> None:
        ok, message = await self.download_manager.run(job)
        if ok:
            if job.on_success:
                job.on_success()
        else:
            if job.on_error:
                job.on_error(message)

    def _model_active_quant(self, model_name: str) -> str:
        cfg = self.registry.models.get(model_name)
        if not cfg:
            return "LOCAL"
        if cfg.active_quant:
            return cfg.active_quant
        return quant_from_filename(cfg.file)

    def _switch_quant(self, model_name: str, quant_id: str) -> None:
        if set_active_quant(MODELS_JSON, model_name, quant_id, models_dir=MODELS_DIR):
            self._reload_registry()
            self.notify(f"Switched to quant {quant_id}")
        else:
            self.notify(f"Quant {quant_id} not available", severity="error")

    def start_model_download(
        self,
        *,
        plan,
        filename: str,
        expected_bytes: int,
        clone_from: str,
        display: str | None = None,
        hf_context: int | None = None,
        on_complete=None,
        on_error=None,
    ) -> bool:
        if self.download_manager.busy:
            self.notify("Another download is already running", severity="warning")
            return False

        repo = plan.repo_id

        def _after_download() -> None:
            from tui.data.gguf import read_gguf_architecture

            base_params = dict(self.registry.models[clone_from].params)
            source = build_source_metadata(plan, filename)
            gguf_path = plan.local_dir / filename
            slug, quant_id = add_or_update_quant(
                MODELS_JSON,
                repo_id=repo,
                filename=filename,
                file_rel=plan.relative_file,
                source=source,
                base_params=base_params,
                gguf_path=gguf_path,
                models_dir=MODELS_DIR,
                display=display,
                hf_context=hf_context,
            )
            self._reload_registry()
            extra = ""
            info = read_gguf_architecture(gguf_path)
            if info.block_count is not None:
                extra = f" layers={info.block_count}"
            self.notify(f"Registered {slug} quant {quant_id}{extra}")
            if on_complete:
                on_complete()

        def _on_error(message: str) -> None:
            self.notify(message[:200], severity="error")
            if on_error:
                on_error(message)

        job = DownloadJob(
            plan=plan,
            filename=filename,
            expected_bytes=expected_bytes,
            clone_from=clone_from,
            display=display,
            on_success=_after_download,
            on_error=_on_error,
        )
        self._run_download_job(job)
        return True

    def action_pick_quant(self) -> None:
        model_name = self._selected_model()
        if not model_name:
            self.notify("Select a model first", severity="warning")
            return
        if model_name not in self.registry.models:
            return
        cfg = self.registry.models[model_name]

        def on_download(model: str, filename: str, expected_bytes: int = 0) -> None:
            source = cfg.params.get("source")
            if not isinstance(source, dict) or not source.get("repo"):
                self.notify("No Hub repo for this model", severity="error")
                return
            clone_from = model_name if model_name in self.registry.models else next(iter(self.registry.models))
            plan = build_download_plan(source["repo"], filename, MODELS_DIR)
            if not self.start_model_download(
                plan=plan,
                filename=filename,
                expected_bytes=expected_bytes,
                clone_from=clone_from,
                display=cfg.display,
            ):
                return
            self.notify(f"Downloading {filename}…", timeout=4)

        def handle(result: str | None) -> None:
            if result:
                self._switch_quant(model_name, result)

        self.push_screen(
            QuantPickerScreen(
                model_name,
                cfg,
                self.registry,
                MODELS_DIR,
                MODELS_JSON,
                baselines_path=TUI_BASELINES_JSON,
                on_download=on_download,
            ),
            handle,
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if not isinstance(event.option_list, (ModelNav, AliasNav)):
            return
        data = event.option_list.selected_data
        if data and data[0] == "preset":
            _, model_name, quant, slot = data
            set_active_preset(self.preset_store, model_name, quant, slot)
            save_presets(PRESETS_JSON, self.preset_store)
            self._reload_registry()

    def watch_theme(self, theme_name: str) -> None:
        """Persist theme choice and refresh panels that use theme colors."""
        if theme_name and theme_name in self.available_themes:
            self.settings.theme = theme_name
            save_settings(TUI_SETTINGS_JSON, self.settings)
        self.call_after_refresh(self._refresh_themed_widgets)

    def _refresh_themed_widgets(self) -> None:
        """Re-render custom panels so theme markup/CSS picks up the new palette."""
        if self._editor_mode:
            return
        for selector in (StatusPanel, ConfigPanel):
            try:
                self.query_one(selector).refresh()
            except Exception:
                pass

    def _refresh_pid(self) -> None:
        info = read_pid_file(PID_FILE)
        if self._editor_mode:
            return
        panel = self.query_one(StatusPanel)
        was_alive = panel.pid_info.alive if panel.pid_info else False
        alive = info.alive if info else False
        if not alive:
            self._gen_history.clear()
            self._throughput_reader.clear()
            panel.gen_tps_history = []
            panel.live_throughput = None
        panel.pid_info = info if alive else None
        running_key = None
        if alive and info:
            running_alias = self.registry.aliases.get(info.model)
            running_key = running_alias.model if running_alias else info.model
            if running_key not in self.registry.models:
                running_key = next(
                    (
                        key
                        for key, model in self.registry.models.items()
                        if model.display == info.model
                    ),
                    None,
                )
        panel.model_display = (
            self.registry.models[running_key].display
            if running_key in self.registry.models
            else None
        )
        panel.preset_display = None
        if running_key in self.registry.models:
            quant = info.quant or self._model_active_quant(running_key)
            slot = info.preset_slot
            if slot is None:
                slot = get_active_slot(self.preset_store, running_key, quant)
            if slot is not None:
                running_preset = get_preset(self.preset_store, running_key, quant, slot)
                if running_preset:
                    panel.preset_display = running_preset.name
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
            metrics, slots = await asyncio.gather(
                self.client.metrics(),
                self.client.slots(),
            )
            panel.metrics = metrics
            if metrics:
                live = self._throughput_reader.update(metrics, slots)
                panel.live_throughput = live
                self._gen_history.push(sample_tps_for_history(live))
                panel.gen_tps_history = self._gen_history.samples
            if panel.props is None:
                panel.props = await self.client.props()
            try:
                self._record_baseline(panel)
            except Exception:
                pass

    def _poll_gpu(self) -> None:
        self.query_one(StatusPanel).gpu = query_gpu()
        try:
            self._record_baseline(self.query_one(StatusPanel))
        except Exception:
            pass

    def _poll_log(self) -> None:
        try:
            size = LOG_FILE.stat().st_size
        except OSError:
            return
        if size != self._log_size:
            self._log_size = size
            self.query_one(LogPanel).tail_file(LOG_FILE)

    def _effective_model_params(self, model_name: str) -> dict | None:
        if model_name not in self.registry.models:
            return None
        identity = dict(self.registry.models[model_name].params)
        quant = self._model_active_quant(model_name)
        slot = get_active_slot(self.preset_store, model_name, quant)
        if slot is not None:
            preset = get_preset(self.preset_store, model_name, quant, slot)
            if preset:
                return merge_identity_and_preset(identity, preset)
        return identity

    def _record_baseline(self, panel: StatusPanel) -> None:
        """Persist observed VRAM / tok/s for the running model on this GPU."""
        info = panel.pid_info
        gpu = panel.gpu
        if not info or not info.alive or not gpu or not gpu.available:
            return
        if gpu.vram_used_mb <= 0:
            return
        params = self._effective_model_params(info.model)
        if not params:
            return

        file_rel = str(params.get("file", ""))
        path = MODELS_DIR / file_rel
        try:
            file_size = path.stat().st_size if path.is_file() else 0
        except OSError:
            file_size = 0

        gen_tps = None
        prompt_tps = None
        tokens = 0.0
        metrics = panel.metrics
        if metrics:
            tokens = metrics.tokens_predicted_total
            if metrics.avg_gen_tps > 0 and tokens >= 16:
                gen_tps = metrics.avg_gen_tps
            elif metrics.predicted_tokens_seconds > 0:
                gen_tps = metrics.predicted_tokens_seconds
            if metrics.avg_prompt_tps > 0:
                prompt_tps = metrics.avg_prompt_tps

        def as_int(value: object, default: int = 0) -> int:
            try:
                return int(float(str(value)))
            except (TypeError, ValueError):
                return default

        ctx = as_int(params.get("ctx"), 0)
        if ctx <= 0:
            return
        try:
            record_baseline(
                TUI_BASELINES_JSON,
                RunBaseline(
                    model=info.model,
                    file=file_rel,
                    file_size=file_size,
                    gpu_name=gpu.name,
                    ctx=ctx,
                    gpu_layers=as_int(params.get("gpu_layers"), 99),
                    total_layers=as_int(params.get("total_layers"), 0),
                    cache_k=str(params.get("cache_k") or "q4_0"),
                    cache_v=str(params.get("cache_v") or "q4_0"),
                    vram_used_mb=gpu.vram_used_mb,
                    gen_tps=gen_tps,
                    prompt_tps=prompt_tps,
                    tokens_predicted=tokens,
                ),
            )
        except OSError:
            return

    def _reload_registry(self, *, notify_gguf: bool = False) -> None:
        """Reload models.json and presets.json, refresh UI."""
        changes = sync_gguf_architecture(MODELS_JSON, MODELS_DIR)
        clamped = clamp_preset_contexts(MODELS_JSON, PRESETS_JSON, models_dir=MODELS_DIR)
        self.registry = load_registry(MODELS_JSON, models_dir=MODELS_DIR)
        self.preset_store = load_presets(PRESETS_JSON)
        nav = self.query_one(ModelNav)
        nav.registry = self.registry
        nav.preset_store = self.preset_store
        nav.refresh_cards()
        aliases = self.query_one(AliasNav)
        aliases.registry = self.registry
        aliases.refresh_aliases()
        cfg = self.query_one(ConfigPanel)
        cfg.registry = self.registry
        cfg.preset_store = self.preset_store
        if notify_gguf and changes:
            bits = ", ".join(
                f"{name} {field} {old}→{new}" for name, field, old, new in changes
            )
            self.notify(f"Updated from GGUF: {bits}")
        if clamped:
            bits = ", ".join(
                f"{model}/{quant}[{slot}] {old}→{new}"
                for model, quant, slot, old, new in clamped[:3]
            )
            extra = f" (+{len(clamped) - 3} more)" if len(clamped) > 3 else ""
            self.notify(f"Capped preset context: {bits}{extra}", severity="warning")

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if not isinstance(event.option_list, (ModelNav, AliasNav)):
            return
        cfg = self.query_one(ConfigPanel)
        data = event.option_list.selected_data
        if data and data[0] in ("model", "preset"):
            cfg.selected = data[1]
        elif data and data[0] == "alias":
            target = self.registry.aliases.get(data[1])
            cfg.selected = target.model if target else None
        cfg.registry = self.registry
        cfg.preset_store = self.preset_store

    def _selected_model(self) -> str | None:
        data = self._selected_data()
        if not data:
            return None
        if data[0] == "alias":
            target = self.registry.aliases.get(data[1])
            return target.model if target else None
        if data[0] in ("model", "preset"):
            return data[1]
        return None

    def _selected_data(self) -> tuple | None:
        if isinstance(self.focused, AliasNav):
            return self.query_one(AliasNav).selected_data
        return self.query_one(ModelNav).selected_data

    def action_launch(self) -> None:
        selected = self._selected_data()
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return

        if model not in self.registry.models:
            self.notify(f"Model '{model}' not found", severity="error")
            return

        cfg = self.registry.models[model]
        alias_name = selected[1] if selected and selected[0] == "alias" else None
        alias_target = self.registry.aliases.get(alias_name) if alias_name else None
        quant = (
            alias_target.quant
            if alias_target and alias_target.quant is not None
            else self._model_active_quant(model)
        )
        slot = (
            alias_target.preset_slot
            if alias_target and alias_target.preset_slot is not None
            else get_active_slot(self.preset_store, model, quant)
        )
        if slot is None:
            self.notify(
                f"No active preset for {cfg.display} ({quant}). Create and apply one first.",
                severity="error",
            )
            return

        preset = get_preset(self.preset_store, model, quant, slot)
        if not preset:
            self.notify(
                f"Active preset slot {slot} is missing for {cfg.display} ({quant})",
                severity="error",
            )
            return

        launch_name = alias_name or cfg.display
        access = "remote" if self.remote_launch else "local"
        self.notify(
            f"Launching {launch_name} ({quant}) preset [{slot}] {preset.name} — {access}"
        )

        try:
            command = [str(REPO_ROOT / "llm-serve"), launch_name]
            if self.remote_launch:
                command.append("--remote")
            r = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=REPO_ROOT,
            )
            if r.returncode == 0:
                self.notify(f"Launched {launch_name}")
                self._reload_registry()
            else:
                err = (r.stderr or r.stdout or "").strip()[:200]
                self.notify(f"Launch failed: {err}", severity="error")
        except Exception as e:
            self.notify(f"Launch error: {e}", severity="error")
        self._refresh_pid()

    def action_toggle_remote(self) -> None:
        """Toggle Meshnet/LAN binding for the next launch."""
        self.remote_launch = not self.remote_launch
        self.query_one(StatusPanel).next_remote = self.remote_launch
        self._update_footer()

    def action_stop(self) -> None:
        try:
            r = subprocess.run(
                [str(REPO_ROOT / "llm-serve"), "stop"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=REPO_ROOT,
            )
            self.notify("Stopped" if r.returncode == 0 else f"Stop: {r.stdout.strip()[:100]}")
        except Exception as e:
            self.notify(f"Stop error: {e}", severity="error")
        self._refresh_pid()

    def on_param_focused(self, event: ParamFocused) -> None:
        self._focused_param = event.param
        if self._help_panel and self._help_visible:
            self._help_panel.show_param(event.param)

    def _enter_param_editor(self, editor: ProfileEditor | PresetEditor) -> None:
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
        """Read the param name from whichever field is focused in the editor."""
        focused = self.focused
        if isinstance(focused, (ParamInput, ParamSelect)):
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
        if self._editor_mode:
            self.notify("Already in edit mode", severity="warning")
            return
        
        data = self._selected_data()
        if data is None:
            self.notify("Select a model, preset, or alias first", severity="warning")
            return

        # Check if it's a preset
        if data[0] == "preset":
            _, model_name, quant, slot = data
            self._edit_preset(model_name, quant, slot)
            return
        
        # Check if it's an alias
        if data[0] == "alias":
            self.notify("Use ←/→ to change this alias target")
            return
        
        # Otherwise it's a model
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return
        if model not in self.registry.models:
            self.notify(f"Model '{model}' not found", severity="error")
            return

        self._reload_registry()
        cfg = self.registry.models[model]
        
        def on_save(new_params: dict) -> None:
            update_model(MODELS_JSON, model, new_params)
            self._reload_registry()
            self._exit_editor()
            self.notify(f"Saved profile {cfg.display}")

        def on_cancel() -> None:
            self._exit_editor()
            self.notify("Edit cancelled")

        editor = ProfileEditor(model, cfg.params, self.registry, on_save, on_cancel)
        self._enter_param_editor(editor)

    def _edit_preset(self, model_name: str, quant: str, slot: int) -> None:
        """Edit a preset."""
        self._reload_registry()
        identity = self.registry.models[model_name].params
        preset = get_preset(self.preset_store, model_name, quant, slot)
        is_new = preset is None

        def on_save(name: str, params: dict) -> None:
            set_preset(self.preset_store, model_name, quant, slot, name, params)
            if is_new or get_active_slot(self.preset_store, model_name, quant) is None:
                set_active_preset(self.preset_store, model_name, quant, slot)
            save_presets(PRESETS_JSON, self.preset_store)
            self._reload_registry()
            self._exit_editor()
            self.notify(f"Saved preset {model_name}/{quant} [{slot}] {name}")

        def on_cancel() -> None:
            self._exit_editor()
            self.notify("Edit cancelled")

        editor = PresetEditor(
            model_name,
            slot,
            preset,
            identity,
            is_new=is_new,
            on_save=on_save,
            on_cancel=on_cancel,
        )
        self._enter_param_editor(editor)

    def cycle_selected_alias(self, direction: int) -> None:
        """Immediately point the selected alias at the adjacent model."""
        nav = self.query_one(AliasNav)
        data = nav.selected_data
        if not data or data[0] != "alias":
            return
        alias_name = data[1]
        model_names = list(self.registry.models)
        if not model_names:
            self.notify("No models are available", severity="warning")
            return
        current_target = self.registry.aliases.get(alias_name)
        current = current_target.model if current_target else None
        try:
            index = model_names.index(current)
        except ValueError:
            index = 0
        target = model_names[(index + direction) % len(model_names)]
        if target == current:
            return
        self.registry.aliases[alias_name] = AliasTarget(model=target)
        save_registry(MODELS_JSON, self.registry)
        self._reload_registry()

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
        self.query_one(ModelNav).focus()

    def action_save_edit(self) -> None:
        """Save current editor (called by Ctrl+S binding)."""
        if self._editor_widget:
            self._editor_widget.save()

    def action_cancel_edit(self) -> None:
        """Cancel current editor (called by Esc binding)."""
        if self._editor_widget:
            self._exit_editor()
            self.notify("Edit cancelled")

    def action_new(self) -> None:
        if self._editor_mode:
            self.notify("Close the editor first", severity="warning")
            return
        data = self._selected_data()
        if data and data[0] == "alias":
            def handle_alias_result(result: tuple[str, str] | None) -> None:
                if result is not None:
                    name, target = result
                    self.registry.aliases[name] = AliasTarget(model=target)
                    save_registry(MODELS_JSON, self.registry)
                    self._reload_registry()
                    self.notify(f"Created alias {name} → {target}")

            self.push_screen(CreateAliasDialog(self.registry), handle_alias_result)
            return

        model = self._selected_model()
        if not model:
            self.notify("Select a model first to create a preset", severity="warning")
            return
        self._new_preset_for_model(model)

    def _new_preset_for_model(self, model_name: str) -> None:
        """Open the preset editor on the next free slot for this model's active quant."""
        if model_name not in self.registry.models:
            self.notify(f"Model '{model_name}' not found", severity="error")
            return
        quant = self._model_active_quant(model_name)
        slot = next_free_slot(self.preset_store, model_name, quant)
        if slot is None:
            self.notify(
                f"{model_name}/{quant} already has {MAX_PRESETS_PER_MODEL} presets",
                severity="warning",
            )
            return
        self._edit_preset(model_name, quant, slot)

    def action_delete(self) -> None:
        data = self._selected_data()
        if data is None:
            self.notify("Select a model, preset, or alias first", severity="warning")
            return
        
        # Check if it's a preset
        if data[0] == "preset":
            _, model_name, quant, slot = data
            preset = get_preset(self.preset_store, model_name, quant, slot)
            if not preset:
                self.notify("Preset not found", severity="error")
                return

            pid_info = read_pid_file(PID_FILE)
            if pid_info and pid_info.alive and pid_info.model == model_name:
                if get_active_slot(self.preset_store, model_name, quant) == slot:
                    self.notify("Cannot delete preset while server is running with it", severity="error")
                    return

            msg = f"Delete preset '{preset.name}' (slot {slot}) for {model_name}/{quant}?"

            def handle_preset_confirm(confirmed: bool) -> None:
                if confirmed:
                    delete_preset(self.preset_store, model_name, quant, slot)
                    if get_active_slot(self.preset_store, model_name, quant) == slot:
                        clear_active_preset(self.preset_store, model_name, quant)
                    save_presets(PRESETS_JSON, self.preset_store)
                    self._reload_registry()
                    self.notify(f"Deleted preset {model_name}/{quant} [{slot}]")

            self.push_screen(ConfirmDialog(msg), handle_preset_confirm)
            return
        
        # Check if it's an alias
        if data[0] == "alias":
            alias_name = data[1]
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
        pid_info = read_pid_file(PID_FILE)
        if pid_info and pid_info.alive and pid_info.model == model:
            self.notify("Stop this model before deleting it", severity="error")
            return
        
        # Check for aliases pointing to this model
        aliases_using = [
            alias
            for alias, target in self.registry.aliases.items()
            if target.model == model
        ]
        files_to_delete = unshared_model_file_paths(self.registry, model, MODELS_DIR)
        bytes_to_free = sum(path.stat().st_size for path in files_to_delete)
        display_name = self.registry.models[model].display
        msg = f"Delete model '{display_name}' and all of its presets?"
        if files_to_delete:
            msg += (
                f"\n\nThis also permanently deletes {len(files_to_delete)} GGUF "
                f"file{'s' if len(files_to_delete) != 1 else ''} "
                f"({fmt_size(bytes_to_free)}) from disk."
            )
        else:
            msg += "\n\nNo unshared GGUF files are stored on disk."
        if aliases_using:
            msg += f"\n\nAliases also removed: {', '.join(aliases_using)}"
        
        def handle_model_confirm(confirmed: bool) -> None:
            if confirmed:
                try:
                    for path in files_to_delete:
                        path.unlink()
                except OSError as exc:
                    self.notify(f"Could not delete model file: {exc}", severity="error")
                    return
                delete_model(MODELS_JSON, model)
                delete_all_presets_for_model(self.preset_store, model)
                save_presets(PRESETS_JSON, self.preset_store)
                self._reload_registry()
                freed = f" and freed {fmt_size(bytes_to_free)}" if files_to_delete else ""
                self.notify(f"Deleted {display_name}{freed}")
        
        self.push_screen(ConfirmDialog(msg), handle_model_confirm)

    def action_activate_preset(self, slot: int) -> None:
        """Activate a family default, or pin/unpin a preset on an alias."""
        selected = self._selected_data()
        model_name = self._selected_model()
        if not model_name:
            self.notify("Select a model card first", severity="warning")
            return
        if selected and selected[0] == "alias":
            alias_name = selected[1]
            target = self.registry.aliases[alias_name]
            if target.preset_slot == slot:
                target.quant = None
                target.preset_slot = None
                save_registry(MODELS_JSON, self.registry)
                self._reload_registry()
                return
        quant = self._model_active_quant(model_name)
        preset = get_preset(self.preset_store, model_name, quant, slot)
        if preset is None:
            self.notify(
                f"No preset [{slot}] for {self.registry.models[model_name].display}",
                severity="warning",
            )
            return
        if selected and selected[0] == "alias":
            alias_name = selected[1]
            target = self.registry.aliases[alias_name]
            target.quant = quant
            target.preset_slot = slot
            save_registry(MODELS_JSON, self.registry)
            self._reload_registry()
            return
        set_active_preset(self.preset_store, model_name, quant, slot)
        save_presets(PRESETS_JSON, self.preset_store)
        self._reload_registry()

    def action_open_hub(self) -> None:
        if self._editor_mode:
            self.notify("Close the editor first", severity="warning")
            return

        def on_complete() -> None:
            self.registry = load_registry(MODELS_JSON, models_dir=MODELS_DIR)
            self._reload_registry()

        self.push_screen(
            HubScreen(
                registry=self.registry,
                settings=self.settings,
                settings_path=TUI_SETTINGS_JSON,
                models_json_path=MODELS_JSON,
                models_dir=MODELS_DIR,
                baselines_path=TUI_BASELINES_JSON,
                on_complete=on_complete,
            )
        )

    def action_help(self) -> None:
        self.notify(
            "Tab: switch pane | ↑↓: navigate | P/Enter: quant | L: launch | S: stop | E: edit | N: new preset | D: delete | A: apply preset | H: Hub | T: theme | Q: quit",
            title="Help",
            timeout=10,
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
