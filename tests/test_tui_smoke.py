"""Headless smoke test for the TUI app using Textual's Pilot."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from rich.console import Console

from tests.support import DISPLAY, MODEL_SLUG, Harness
from tui.widgets.config import ConfigPanel
from tui.widgets.nav import ModelNav
from tui.widgets.status import StatusPanel


def render_text(renderable) -> str:
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)
    console.print(renderable)
    return output.getvalue()


class TuiSmokeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    async def test_dashboard_renders_fixture_model(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.5)

            nav = app.query_one(ModelNav)
            model_slugs = [
                data[1]
                for data in nav._option_data.values()
                if data and data[0] == "model"
            ]
            self.assertIn(MODEL_SLUG, model_slugs)

            labels = [
                str(nav.get_option(option_id).prompt)
                for option_id, data in nav._option_data.items()
                if data and data[0] == "model"
            ]
            self.assertTrue(any(DISPLAY.lower() in label.lower() for label in labels), labels)
            self.assertTrue(any("Quant" in label and "ctx:" in label for label in labels), labels)

            status = app.query_one(StatusPanel)
            rendered = render_text(status.render())
            self.assertTrue("NOT RUNNING" in rendered or "RUNNING" in rendered)

            cfg = app.query_one(ConfigPanel)
            cfg.selected = MODEL_SLUG
            cfg.registry = app.registry
            cfg.preset_store = app.preset_store
            cfg.models_dir = app.paths.models_dir
            text = render_text(cfg.render())
            self.assertIn("Model", text)
            self.assertIn(DISPLAY, text)
            self.assertIn("Preset", text)

            app._refresh_pid()
            app._poll_gpu()
            await pilot.pause(0.3)
            r2 = render_text(app.query_one(StatusPanel).render())
            self.assertIn("GPU", r2)

            with patch("tui.app.save_settings"):
                await pilot.press("r")
            await pilot.pause(0.2)
