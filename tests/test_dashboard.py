"""Tests for the compact main-page dashboard renderers."""

from __future__ import annotations

import io
import os
import unittest

from rich.console import Console

from tui.app import (
    ConfigPanel,
    StatusPanel,
    generation_health,
    temperature_health,
    vram_health,
)
from tui.data.gpu import GPUStats
from tui.data.pidfile import PidInfo
from tui.data.stats import Metrics
from tui.data.throughput_history import LastRequest, LiveThroughput


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
        self.assertIn("49.5 t/s generation", rendered)
        self.assertNotIn("59,916 prompt tokens", rendered)

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
        panel.preset_display = "default"

        rendered = render_text(panel.render())

        self.assertIn("RUNNING  Qwen 3.6  default  REMOTE", rendered)
        self.assertNotIn("qwen36-27b-bartowski", rendered)

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

        self.assertIn("(94%) CRITICAL", rendered)
        self.assertIn("87°C HOT", rendered)

    def test_status_panel_renders_throughput_graph(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics(predicted_tokens_seconds=25.0)
        panel.live_throughput = LiveThroughput(gen_tps=25.0, source="metrics_gauge")
        panel.gen_tps_history = [10.0, 20.0, 25.0, 30.0]

        rendered = render_text(panel.render())

        self.assertIn("avg 21.2 tok/s", rendered)
        self.assertTrue(any(ch in rendered for ch in "▁▂▃▄▅▆▇█"))

    def test_status_panel_shows_next_launch_remote_toggle(self) -> None:
        panel = StatusPanel()
        self.assertIn("NEXT LAUNCH [LOCAL]  [LOG TRACE]", render_text(panel.render()))

        panel.next_remote = True
        panel.next_log_verbosity = 3

        rendered = render_text(panel.render())
        self.assertIn("NEXT LAUNCH [REMOTE]", rendered)
        self.assertIn("[LOG INFO]", rendered)


if __name__ == "__main__":
    unittest.main()
