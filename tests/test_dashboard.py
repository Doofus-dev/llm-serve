"""Tests for the compact main-page dashboard renderers."""

from __future__ import annotations

import io
import os
import unittest

from rich.console import Console

from tui.widgets.config import ConfigPanel
from tui.widgets.health import generation_health, temperature_health, vram_health
from tui.widgets.status import StatusHeader, StatusPanel, format_generation_speed
from tui.data.throughput_history import LastRequest, LiveThroughput, SPARKLINE_WIDTH
from tui.data.gpu import GPUStats, ProcessMem
from tui.data.pidfile import PidInfo, write_pid_file
from tui.data.stats import Metrics


def render_text(renderable) -> str:
    output = io.StringIO()
    Console(file=output, width=100, color_system=None).print(renderable)
    return output.getvalue()


class DashboardTests(unittest.TestCase):
    def test_config_labels_and_values_use_distinct_styles(self) -> None:
        self.assertEqual(str(ConfigPanel._label("ctx").style), "cyan")
        self.assertEqual(str(ConfigPanel._value(40960).style), "bold yellow")

    def test_status_panel_renders_populated_metrics(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics(
            prompt_tokens_total=59_916,
            tokens_predicted_total=3_732,
            prompt_tokens_seconds=120.5,
            predicted_tokens_seconds=49.5,
            requests_processing=1,
        )
        panel.live_throughput = LiveThroughput(
            gen_tps=49.5,
            prompt_tps=120.5,
            source="metrics_gauge",
            stage="generating",
        )

        rendered = render_text(panel.render())

        self.assertIn("GENERATE", rendered)
        self.assertIn("49.5 t/s FAST", rendered)
        self.assertIn("20.2 ms/token", rendered)
        self.assertNotIn("t/s generation", rendered)
        self.assertNotIn("59,916 prompt tokens", rendered)

    def test_generation_speed_fits_sparkline_width(self) -> None:
        cases = (
            (10.0, "MODERATE", "100.0"),
            (100.0, "FAST", "10.0"),
            (49.5, "FAST", "20.2"),
        )
        for tps, label, ms_token in cases:
            with self.subTest(tps=tps):
                line = format_generation_speed(tps)
                self.assertLessEqual(line.cell_len, SPARKLINE_WIDTH)
                self.assertTrue(line.no_wrap)
                output = io.StringIO()
                Console(file=output, width=SPARKLINE_WIDTH, color_system=None).print(line)
                rendered = output.getvalue()
                self.assertEqual(rendered.count("\n"), 1)
                self.assertIn(f"{tps:.1f} t/s {label}", rendered)
                self.assertIn(f"{ms_token} ms/token", rendered)
                self.assertNotIn("generation", rendered)

    def test_status_panel_shows_prefill_phase(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics(requests_processing=1, prompt_tokens_seconds=80.0)
        panel.live_throughput = LiveThroughput(
            gen_tps=0.0,
            prompt_tps=3800.0,
            source="slots",
            stage="prefill",
            n_prompt_processed=2431,
            n_prompt_total=4096,
            n_prompt_cache=1200,
        )

        rendered = render_text(panel.render())

        self.assertIn("PREFILL", rendered)
        self.assertIn("CACHE", rendered)
        self.assertIn("3,631/4,096", rendered)
        self.assertIn("3.8k t/s", rendered)
        self.assertIn("waiting on generate", rendered)
        self.assertNotIn("3800", rendered)

    def test_status_panel_shows_last_request_when_idle(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics()
        panel.live_throughput = LiveThroughput(
            stage="idle",
            last_request=LastRequest(
                prompt_tokens=4096,
                cache_tokens=1200,
                gen_tokens=142,
                gen_tps=41.0,
            ),
        )

        rendered = render_text(panel.render())
        collapsed = " ".join(rendered.split())

        self.assertIn("IDLE queue cache prefill generate", collapsed)
        self.assertIn("last: 4,096 prompt (1,200 cache) · 142 gen @ 41 t/s", collapsed)

    def test_status_panel_prefers_friendly_model_name(self) -> None:
        panel = StatusPanel()
        panel.pid_info = PidInfo(
            pid=os.getpid(),
            model="qwen36-27b-bartowski",
            port=8080,
            ts="",
            remote=True,
        )
        panel.model_display = "Qwen 3.6"
        panel.quant_display = "Q8_0"
        panel.preset_display = "[1]"

        rendered = render_text(panel.render())

        self.assertIn("RUNNING  Qwen 3.6  Q8_0  [1]  REMOTE", rendered)
        self.assertNotIn("qwen36-27b-bartowski", rendered)

    def test_status_panel_hot_bar_shows_family_and_slot_not_path(self) -> None:
        panel = StatusPanel()
        panel.pid_info = PidInfo(
            pid=os.getpid(),
            model="qwen38-27b-bartowski",
            port=8080,
            ts="",
        )
        panel.model_display = "Qwen 3.8"
        panel.quant_display = "Q8_0"
        panel.preset_display = "[2]"
        panel.uptime = 65
        panel.props = {
            "model_alias": "coding",
            "model_path": "/home/doofus/models/Qwen3.8-27B-Q8_0.gguf",
        }

        rendered = render_text(panel.render())

        self.assertIn("Qwen 3.8  •  slot 2", rendered)
        self.assertNotIn("alias coding", rendered)
        self.assertNotIn("Qwen3.8-27B-Q8_0.gguf", rendered)
        self.assertNotIn("port 8080", rendered)
        self.assertNotIn(f"PID {os.getpid()}", rendered)
        self.assertNotIn("up 0:01:05", rendered)

    def test_status_header_shows_live_status_or_not_running(self) -> None:
        header = StatusHeader()
        self.assertIn("not running", render_text(header.render()))

        header.pid_info = PidInfo(
            pid=os.getpid(),
            model="qwen38-27b-bartowski",
            port=8080,
            ts="",
        )
        header.uptime = 65
        rendered = render_text(header.render())
        self.assertIn(f"port 8080  •  PID {os.getpid()}  •  up 0:01:05", rendered)

    def test_status_health_thresholds(self) -> None:
        self.assertEqual(vram_health(74.9), ("OK", "bold green"))
        self.assertEqual(vram_health(75), ("HIGH", "bold yellow"))
        self.assertEqual(vram_health(90), ("CRITICAL", "bold red"))
        self.assertEqual(temperature_health(69.9), ("COOL", "bold green"))
        self.assertEqual(temperature_health(70), ("WARM", "bold yellow"))
        self.assertEqual(temperature_health(85), ("HOT", "bold red"))
        self.assertEqual(generation_health(20), ("FAST", "bold green"))
        self.assertEqual(generation_health(5), ("MODERATE", "bold yellow"))
        self.assertEqual(generation_health(1), ("SLOW", "bold red"))

    def test_status_panel_renders_gpu_pressure_indicators(self) -> None:
        panel = StatusPanel()
        panel.gpu = GPUStats(
            name="Test GPU",
            vram_used_mb=15_360,
            vram_total_mb=16_384,
            utilization_pct=96,
            temp_c=87,
            available=True,
        )

        rendered = render_text(panel.render())

        self.assertIn("CRITICAL", rendered)
        self.assertIn("87°C HOT", rendered)

    def test_status_panel_renders_throughput_graph(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics(predicted_tokens_seconds=25.0)
        panel.live_throughput = LiveThroughput(gen_tps=25.0, source="metrics_gauge")
        panel.gen_tps_history = [10.0, 20.0, 25.0, 30.0]

        rendered = render_text(panel.render())

        self.assertIn("avg 21.2 tok/s", rendered)
        self.assertTrue(any(ch in rendered for ch in "▁▂▃▄▅▆▇█"))


    def test_status_panel_renders_multiple_gpu_processes(self) -> None:
        panel = StatusPanel()
        panel.gpu = GPUStats(
            name="Test GPU",
            vram_used_mb=12_288,
            vram_total_mb=16_384,
            utilization_pct=40,
            temp_c=55,
            available=True,
            processes=[
                ProcessMem(pid=100, name="llama-server", vram_mb=8192, ram_mb=1500, tracked=True),
                ProcessMem(pid=200, name="other-llama", vram_mb=4096, ram_mb=900, tracked=False),
            ],
        )

        rendered = render_text(panel.render())
        collapsed = " ".join(rendered.split())

        self.assertIn("12.0G / 16.0G total", collapsed)
        self.assertIn("llama-server", collapsed)
        self.assertIn("other-llama", collapsed)
        self.assertIn("100", collapsed)
        self.assertIn("200", collapsed)
        self.assertIn("8.0G", collapsed)
        self.assertIn("4.0G", collapsed)
        self.assertIn("VRAM", collapsed)
        self.assertIn("RAM", collapsed)

    def test_status_panel_shows_full_seven_digit_pids(self) -> None:
        panel = StatusPanel()
        panel.gpu = GPUStats(
            name="Test GPU",
            vram_used_mb=12_288,
            vram_total_mb=16_384,
            utilization_pct=40,
            temp_c=55,
            available=True,
            processes=[
                ProcessMem(
                    pid=1120647,
                    name="llama-server",
                    vram_mb=13902,
                    ram_mb=1251,
                    tracked=True,
                ),
                ProcessMem(
                    pid=4155563,
                    name="llama-server",
                    vram_mb=0,
                    ram_mb=2.5,
                    tracked=False,
                ),
            ],
        )

        rendered = render_text(panel.render())

        self.assertIn("1120647", rendered)
        self.assertIn("4155563", rendered)
        self.assertNotIn("11206…", rendered)
        self.assertNotIn("41555…", rendered)

    def test_status_panel_shows_next_launch_remote_toggle(self) -> None:
        panel = StatusPanel()
        self.assertIn("NEXT LAUNCH [LOCAL]  [LOG TRACE]", render_text(panel.render()))

        panel.next_remote = True
        panel.next_log_verbosity = 3

        rendered = render_text(panel.render())
        self.assertIn("NEXT LAUNCH [REMOTE]", rendered)
        self.assertIn("[LOG INFO]", rendered)


class StatusHeaderComposeTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_top_bar_is_live_status_not_title(self) -> None:
        from textual.widgets import Header

        from tests.support import Harness

        harness = Harness()
        try:
            app = harness.app()
            async with app.run_test(size=(120, 40)):
                self.assertEqual(len(list(app.query(Header))), 0)
                header = app.query_one(StatusHeader)
                self.assertIn("not running", render_text(header.render()))
        finally:
            harness.cleanup()

    async def test_header_stays_live_in_editor_mode(self) -> None:
        from tests.support import MODEL_SLUG, QUANT, Harness

        harness = Harness()
        try:
            app = harness.app()
            async with app.run_test(size=(120, 40)):
                write_pid_file(
                    harness.paths.pid_file,
                    pid=os.getpid(),
                    model=MODEL_SLUG,
                    port=8080,
                    quant=QUANT,
                    preset_slot=1,
                    remote=False,
                )
                app._editor_mode = True
                app._refresh_pid()
                header = app.query_one(StatusHeader)
                self.assertIn(
                    f"port 8080  •  PID {os.getpid()}",
                    render_text(header.render()),
                )

                harness.paths.pid_file.unlink()
                app._refresh_pid()
                self.assertIn("not running", render_text(header.render()))
        finally:
            harness.cleanup()


if __name__ == "__main__":
    unittest.main()
