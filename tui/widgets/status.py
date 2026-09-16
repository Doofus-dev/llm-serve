"""Live status, throughput, and GPU panel."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static

from tui.data.gpu import GPUStats
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
from tui.theme import ACCENT_STYLE, OK_STYLE, WARN_STYLE
from tui.widgets.health import (
    fmt_uptime,
    generation_health,
    health_dot,
    temperature_health,
    vram_health,
)


# --- VRAM bar gauge -------------------------------------------------------
# The GPU column of the telemetry table is ~45 cells wide. The gauge row
# must fit: bar + " NN% LABEL" without wrapping.
VRAM_GAUGE_BAR_WIDTH = 30


def render_vram_gauge(g: GPUStats, label: str, style: str) -> Text:
    """Render a horizontal bar gauge filled to the VRAM percentage.

    The bar is colored by the health state. The percentage and health
    label are centered in the middle of the bar, with reverse coloring
    so they stay readable against either the filled or empty region.
    The GPU name is rendered above the gauge by the caller.
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
            # Percentage digits: black over filled bar, white over empty bar.
            line.append(text[i - text_start], style="black" if i < filled else "white")
        else:
            char = "▓" if i < filled else "░"
            line.append(char, style=style if i < filled else None)
    return line


class StatusPanel(Static):
    """Live status + throughput + GPU."""

    pid_info: reactive[PidInfo | None] = reactive(None)
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
            state = Text("○ NOT RUNNING", style="bold red")
            state.append("  Press L to launch the selected model", style="dim")

        launch_flag = Text("NEXT LAUNCH ", style="dim")
        if self.next_remote:
            launch_flag.append("[REMOTE]", style=ACCENT_STYLE)
        else:
            launch_flag.append("[LOCAL]", style=OK_STYLE)
        launch_flag.append(f"  [LOG {log_verbosity_label(self.next_log_verbosity)}]", style=ACCENT_STYLE)

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
                speed.append(health_dot(gen_style))
                speed.append(" ")
                speed.append(f"{gen:.1f} t/s ", style=gen_style)
                speed.append("generation", style="dim")
                if gen > 0:
                    speed.append(f"  {gen_label}  {(1000.0 / gen):.1f} ms/token", style=gen_style)
                throughput.append(speed)
            elif live.stage == "queued":
                throughput.append(Text("waiting for a free slot", style=WARN_STYLE))
            else:
                recap = format_last_request(live.last_request)
                throughput.append(recap if recap is not None else Text("idle", style="dim"))
            throughput.append(render_tps_sparkline(history, width=SPARKLINE_WIDTH))
            avg_line = format_avg_line(history, METRICS_POLL_INTERVAL)
            if avg_line is None:
                avg_line = Text("avg —", style="dim")
                if live.stage == "prefill":
                    avg_line.append("  (waiting on generate)", style="dim")
            throughput.append(avg_line)
        else:
            empty = Text(justify="center")
            empty.append("○ ", style="dim")
            empty.append("Metrics unavailable", style="dim")
            empty.append("\n")
            empty.append("Press L to launch the selected model", style="bold dim")
            throughput.append(empty)
        gpu_lines: list[Text] = []
        g = self.gpu
        if g and g.available:
            gpu_lines.append(Text(g.name, style="bold"))
            vram_label, vram_style = vram_health(g.vram_pct)
            memory = render_vram_gauge(g, vram_label, vram_style)
            gpu_lines.append(memory)

            thermals = Text(f"{g.utilization_pct:.0f}% utilization  •  ")
            if g.temp_c > 0:
                temp_label, temp_style = temperature_health(g.temp_c)
                thermals.append(health_dot(temp_style))
                thermals.append(" ")
                thermals.append(f"{g.temp_c:.0f}°C ", style=temp_style)
                thermals.append(temp_label, style=temp_style)
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
            gpu_empty = Text(justify="center")
            gpu_empty.append("○ GPU stats unavailable", style="dim")
            gpu_lines.append(gpu_empty)

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
