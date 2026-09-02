"""Tests for the compact main-page dashboard renderers."""

from __future__ import annotations

import io
import unittest

from rich.console import Console

from tui.app import ConfigPanel, StatusPanel
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


if __name__ == "__main__":
    unittest.main()
