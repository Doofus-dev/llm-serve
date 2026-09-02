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

        rendered = render_text(panel.render())

        self.assertIn("3,732 gen / 59,916 prompt tokens", rendered)
        self.assertIn("49.5 tok/s generation", rendered)
        self.assertIn("1 active", rendered)

    def test_status_panel_prefers_friendly_model_name(self) -> None:
        panel = StatusPanel()
        panel.pid_info = PidInfo(
            pid=os.getpid(),
            model="qwen36-27b-bartowski",
            port=8080,
            ts="",
        )
        panel.model_display = "Qwen 3.6"
        panel.preset_display = "default"

        rendered = render_text(panel.render())

        self.assertIn("RUNNING  Qwen 3.6  default", rendered)
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


if __name__ == "__main__":
    unittest.main()
