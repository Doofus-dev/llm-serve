"""Live status, throughput, and GPU panel."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text
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
from tui.widgets.health import (
    fmt_uptime,
    generation_health,
    temperature_health,
    vram_health,
)


class StatusPanel(Static):
    """Live status + throughput + GPU."""

    pid_info: reactive[PidInfo | None] = reactive(None)
    model_display: reactive[str | None] = reactive(None)
    preset_display: reactive[str | None] = reactive(None)
    next_remote: reactive[bool] = reactive(False)
    next_log_verbosity: reactive[int] = reactive(4)
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
        launch_flag.append(f"  [LOG {log_verbosity_label(self.next_log_verbosity)}]", style="bold cyan")

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
