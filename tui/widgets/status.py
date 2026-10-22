"""Live status, throughput, and GPU panel."""

from __future__ import annotations

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static, Switch

from tui.data.gpu import GPUStats, ProcessMem
from tui.data.pidfile import PidInfo
from tui.data.settings import log_verbosity_label
from tui.data.stats import Metrics
from tui.data.throughput_history import (
    LiveThroughput,
    SPARKLINE_WIDTH,
    format_avg_line,
    format_last_request,
    live_tps_from_metrics,
    render_prefill_progress,
    render_stage_strip,
    render_tps_sparkline,
)
from tui.paths import METRICS_POLL_INTERVAL
from tui.theme import ACCENT, ACCENT_STYLE, BOLD_STYLE, DIM_STYLE, ERR_STYLE, OK_STYLE, TEXT_MUTED, WARN_STYLE
from tui.widgets.health import (
    fmt_uptime,
    generation_health,
    health_dot,
    temperature_health,
    vram_health,
)


# --- VRAM bar gauge -------------------------------------------------------
# GPU column is ~45 cells. Gauge + process rows must stay on one line.
VRAM_GAUGE_BAR_WIDTH = 24
PROCESS_NAME_WIDTH = 14


def fmt_mem_mb(mb: float) -> str:
    if mb <= 0:
        return "—"
    if mb >= 1024:
        return f"{mb / 1024:.1f}G"
    return f"{mb:.0f}M"


def render_process_rows(processes: list[ProcessMem]) -> Table:
    table = Table(
        box=None,
        expand=True,
        padding=(0, 1),
        show_edge=False,
        show_header=True,
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("PROC", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("PID", justify="right", no_wrap=True, width=8)
    table.add_column("VRAM", justify="right", no_wrap=True, width=6)
    table.add_column("RAM", justify="right", no_wrap=True, width=6)
    for proc in processes:
        mark = "*" if proc.tracked else " "
        name = proc.name or str(proc.pid)
        if len(name) > PROCESS_NAME_WIDTH:
            name = name[: PROCESS_NAME_WIDTH - 1] + "…"
        table.add_row(
            Text(f"{mark}{name}", style=ACCENT_STYLE if proc.tracked else None),
            str(proc.pid),
            fmt_mem_mb(proc.vram_mb),
            fmt_mem_mb(proc.ram_mb),
        )
    return table


def render_vram_gauge(g: GPUStats, label: str, style: str) -> Text:
    """Render a horizontal bar gauge filled to the VRAM percentage.

    The bar is colored by the health state. The percentage and health
    label are centered in the middle of the bar, all colored with the
    bar's health color. The GPU name is rendered above the gauge by
    the caller.
    """
    pct = g.vram_pct
    bar_width = VRAM_GAUGE_BAR_WIDTH
    filled = int(round(pct / 100 * bar_width))
    filled = max(0, min(bar_width, filled))

    text = f"{pct:.0f}% {label}"
    text_len = len(text)
    text_start = (bar_width - text_len) // 2
    text_start = max(0, min(bar_width - text_len, text_start))

    line = Text(no_wrap=True)
    for i in range(bar_width):
        if text_start <= i < text_start + text_len:
            line.append(text[i - text_start], style=style)
        else:
            char = "▓" if i < filled else "░"
            line.append(char, style=style if i < filled else None)
    return line


def format_generation_speed(gen_tps: float, width: int = SPARKLINE_WIDTH) -> Text:
    """One-line live rate: t/s, health, ms/token, sized to the sparkline column."""
    gen_label, gen_style = generation_health(gen_tps)
    rate = f"{gen_tps:.1f} t/s"
    extras = f" {gen_label} {(1000.0 / gen_tps):.1f} ms/token" if gen_tps > 0 else ""
    speed = Text()
    speed.append(health_dot(gen_style))
    speed.append(" ")
    speed.append(rate, style=gen_style)
    if extras:
        speed.append(extras, style=gen_style)
    speed.no_wrap = speed.cell_len <= width
    return speed


def remote_toggle_label(remote_on: bool) -> str:
    """Always-visible STATUS switch caption for the next launch bind."""
    if remote_on:
        return "Remote ON   Meshnet/LAN  0.0.0.0  (R)"
    return "Remote OFF  localhost only  (R)"


def format_preset_display(slot: int | None, name: str | None = None) -> str | None:
    """Preset slot label for STATUS, e.g. ``[5] 120K-HighPower``."""
    if slot is None:
        return None
    label = f"[{slot}]"
    cleaned = (name or "").strip()
    if cleaned:
        return f"{label} {cleaned}"
    return label


def format_instance_preset_column(
    quant: str | None,
    slot: int | None,
    name: str | None = None,
) -> str:
    """PRESET column for the multi-instance STATUS table."""
    quant_text = quant or "-"
    preset = format_preset_display(slot, name)
    if preset:
        return f"{quant_text} {preset}"
    return quant_text


def fmt_server_status_line(
    info: PidInfo | None,
    uptime: float,
    instances: list[PidInfo] | None = None,
) -> str:
    """Live `port • PID • up` line shown in the top header."""
    extras = [item for item in (instances or []) if item.alive]
    if info and info.alive:
        extras = [item for item in extras if item.pid != info.pid]
        if not extras:
            return f"port {info.port}  •  PID {info.pid}  •  up {fmt_uptime(uptime)}"
        ports = " ".join(f":{item.port}" for item in [info, *extras])
        return f"{1 + len(extras)} running  •  {ports}"
    if extras:
        if len(extras) == 1:
            only = extras[0]
            return f"port {only.port}  •  PID {only.pid}  •  up {fmt_uptime(uptime)}"
        ports = " ".join(f":{item.port}" for item in extras)
        return f"{len(extras)} running  •  {ports}"
    return "not running"


class StatusHeader(Static):
    """Top bar: live server port, PID, and uptime (or a not-running placeholder)."""

    DEFAULT_CSS = """
    StatusHeader {
        dock: top;
        width: 100%;
        height: 1;
        background: $panel;
        color: $foreground;
        text-style: bold;
        padding: 0 1;
        content-align: left middle;
    }
    """

    pid_info: reactive[PidInfo | None] = reactive(None)
    instances: reactive[list[PidInfo]] = reactive([])
    uptime: reactive[float] = reactive(0.0)

    def render(self) -> Text:
        info = self.pid_info
        line = fmt_server_status_line(info, self.uptime, self.instances)
        if line == "not running":
            return Text("not running", style=TEXT_MUTED)
        return Text(line, style=ACCENT)


class StatusPanel(Vertical):
    """Live status + throughput + GPU, plus the next-launch remote switch."""

    DEFAULT_CSS = """
    StatusPanel {
        layout: vertical;
        height: auto;
    }
    #remote-toggle-row {
        height: 3;
        min-height: 3;
        padding: 0 1;
        align: left middle;
    }
    #remote-switch {
        width: auto;
        height: 1;
        margin: 0 1 0 0;
    }
    #remote-switch-label {
        height: 1;
        color: $text-muted;
    }
    #status-body {
        height: auto;
    }
    """

    pid_info: reactive[PidInfo | None] = reactive(None)
    instances: reactive[list[PidInfo]] = reactive([])
    instance_labels: reactive[dict[int, str]] = reactive({})
    instance_preset_names: reactive[dict[int, str]] = reactive({})
    model_display: reactive[str | None] = reactive(None)
    quant_display: reactive[str | None] = reactive(None)
    preset_display: reactive[str | None] = reactive(None)
    next_remote: reactive[bool] = reactive(False)
    next_log_verbosity: reactive[int] = reactive(4)
    metrics: reactive[Metrics | None] = reactive(None)
    live_throughput: reactive[LiveThroughput | None] = reactive(None)
    gen_tps_history: reactive[list[float]] = reactive([])
    gpu: reactive[GPUStats | None] = reactive(None)
    props: reactive[dict | None] = reactive(None)
    uptime: reactive[float] = reactive(0.0)

    def compose(self) -> ComposeResult:
        yield Static("STATUS", classes="card-title", id="status-title")
        with Horizontal(id="remote-toggle-row"):
            yield Switch(value=False, id="remote-switch")
            yield Label(
                remote_toggle_label(False),
                id="remote-switch-label",
            )
        yield Static(id="status-body")

    def on_mount(self) -> None:
        self._sync_remote_switch()
        self._refresh_body()

    def _refresh_body(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#status-body", Static).update(self.render())

    def _sync_remote_switch(self) -> None:
        if not self.is_mounted:
            return
        switch = self.query_one("#remote-switch", Switch)
        if switch.value != self.next_remote:
            with switch.prevent(Switch.Changed):
                switch.value = self.next_remote
        self.query_one("#remote-switch-label", Label).update(
            remote_toggle_label(self.next_remote)
        )

    def watch_next_remote(self) -> None:
        self._sync_remote_switch()
        self._refresh_body()

    def watch_pid_info(self) -> None:
        self._refresh_body()

    def watch_instances(self) -> None:
        self._refresh_body()

    def watch_instance_labels(self) -> None:
        self._refresh_body()

    def watch_instance_preset_names(self) -> None:
        self._refresh_body()

    def watch_model_display(self) -> None:
        self._refresh_body()

    def watch_quant_display(self) -> None:
        self._refresh_body()

    def watch_preset_display(self) -> None:
        self._refresh_body()

    def watch_next_log_verbosity(self) -> None:
        self._refresh_body()

    def watch_metrics(self) -> None:
        self._refresh_body()

    def watch_live_throughput(self) -> None:
        self._refresh_body()

    def watch_gen_tps_history(self) -> None:
        self._refresh_body()

    def watch_gpu(self) -> None:
        self._refresh_body()

    def watch_props(self) -> None:
        self._refresh_body()

    def watch_uptime(self) -> None:
        self._refresh_body()

    def render(self) -> Group:
        info = self.pid_info
        if info and info.alive:
            state = Text("● RUNNING", style=OK_STYLE)
            state.append(f"  {self.model_display or info.model}", style=ACCENT_STYLE)
            if self.quant_display:
                state.append(f"  {self.quant_display}", style=WARN_STYLE)
            if self.preset_display:
                state.append(f"  {self.preset_display}", style=WARN_STYLE)
            if info.remote:
                state.append("  REMOTE", style=ACCENT_STYLE)
        else:
            state = Text("○ NOT RUNNING", style=ERR_STYLE)
            state.append("  Press L to launch the selected model", style=DIM_STYLE)

        launch_flag = Text("NEXT LAUNCH ", style=DIM_STYLE)
        if self.next_remote:
            launch_flag.append("[REMOTE]", style=ACCENT_STYLE)
        else:
            launch_flag.append("[LOCAL]", style=OK_STYLE)
        launch_flag.append(f"  [LOG {log_verbosity_label(self.next_log_verbosity)}]", style=ACCENT_STYLE)

        header = Table.grid(expand=True, padding=(0, 0))
        header.add_column(ratio=1)
        header.add_column(justify="right")
        header.add_row(state, launch_flag)

        renderables: list = [header]
        if info and info.alive:
            family = self.model_display or info.model
            bits = [family]
            if self.quant_display:
                bits.append(self.quant_display)
            if self.preset_display:
                bits.append(self.preset_display)
            renderables.append(Text("  •  ".join(bits), style=TEXT_MUTED))

        alive_instances = [item for item in self.instances if item.alive]
        if len(alive_instances) > 1:
            vram_by_pid: dict[int, float] = {}
            if self.gpu:
                for proc in self.gpu.processes:
                    vram_by_pid[proc.pid] = vram_by_pid.get(proc.pid, 0.0) + proc.vram_mb
            table = Table(
                box=None,
                expand=True,
                padding=(0, 1),
                show_edge=False,
                show_header=True,
                pad_edge=False,
                collapse_padding=True,
            )
            table.add_column("MODEL", ratio=2, no_wrap=True, overflow="ellipsis")
            table.add_column("PORT", justify="right", no_wrap=True, width=6)
            table.add_column("PRESET", no_wrap=True, overflow="ellipsis", min_width=16, ratio=2)
            table.add_column("VRAM", justify="right", no_wrap=True, width=6)
            for item in alive_instances:
                label = self.instance_labels.get(item.pid) or item.model
                preset = format_instance_preset_column(
                    item.quant,
                    item.preset_slot,
                    self.instance_preset_names.get(item.pid),
                )
                used = vram_by_pid.get(item.pid)
                vram = fmt_mem_mb(used) if used else "n/a"
                table.add_row(label, str(item.port), preset, vram)
            renderables.append(table)

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
                throughput.append(format_generation_speed(gen))
            elif live.stage == "queued":
                throughput.append(Text("waiting for a free slot", style=WARN_STYLE))
            else:
                recap = format_last_request(live.last_request)
                throughput.append(recap if recap is not None else Text("idle", style=DIM_STYLE))
            throughput.append(render_tps_sparkline(history, width=SPARKLINE_WIDTH))
            avg_line = format_avg_line(history, METRICS_POLL_INTERVAL)
            if avg_line is None:
                avg_line = Text("avg —", style=DIM_STYLE)
                if live.stage == "prefill":
                    avg_line.append("  (waiting on generate)", style=DIM_STYLE)
            throughput.append(avg_line)
        else:
            empty = Text(justify="center")
            empty.append("○ ", style=DIM_STYLE)
            empty.append("Metrics unavailable", style=TEXT_MUTED)
            empty.append("\n")
            empty.append("Press L to launch the selected model", style=DIM_STYLE)
            throughput.append(empty)
        gpu_lines: list = []
        g = self.gpu
        if g and g.available:
            title = Text(g.name, style=BOLD_STYLE)
            title.append(
                f"  {fmt_mem_mb(g.vram_used_mb)} / {fmt_mem_mb(g.vram_total_mb)} total",
                style=DIM_STYLE,
            )
            gpu_lines.append(title)
            vram_label, vram_style = vram_health(g.vram_pct)
            gpu_lines.append(render_vram_gauge(g, vram_label, vram_style))

            thermals = Text(f"{g.utilization_pct:.0f}% util  •  ", no_wrap=True, style=TEXT_MUTED)
            if g.temp_c > 0:
                temp_label, temp_style = temperature_health(g.temp_c)
                thermals.append(health_dot(temp_style))
                thermals.append(" ")
                thermals.append(f"{g.temp_c:.0f}°C ", style=temp_style)
                thermals.append(temp_label, style=temp_style)
            else:
                thermals.append("temp n/a", style=DIM_STYLE)
            gpu_lines.append(thermals)
            if g.unified and g.dedicated_total_mb:
                gpu_lines.append(
                    Text(
                        f"BAR {fmt_mem_mb(g.dedicated_used_mb)} / {fmt_mem_mb(g.dedicated_total_mb)}",
                        style=DIM_STYLE,
                        no_wrap=True,
                    )
                )
            if g.processes:
                gpu_lines.append(render_process_rows(g.processes))
            else:
                gpu_lines.append(Text("no GPU processes", style=DIM_STYLE))
        else:
            gpu_empty = Text(justify="center")
            gpu_empty.append("○ GPU stats unavailable", style=TEXT_MUTED)
            gpu_lines.append(gpu_empty)

        telemetry = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
            show_edge=False,
            pad_edge=False,
            collapse_padding=True,
            header_style=DIM_STYLE,
            border_style=DIM_STYLE,
        )
        telemetry.add_column("THROUGHPUT", ratio=1)
        telemetry.add_column("GPU", ratio=1)
        telemetry.add_row(Group(*throughput), Group(*gpu_lines))
        renderables.append(telemetry)
        return Group(*renderables)
