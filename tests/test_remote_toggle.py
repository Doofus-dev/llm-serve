"""Visible next-launch remote toggle: STATUS switch, footer, and launch flag."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual.widgets import Switch

from tui.bindings import build_app_bindings
from tui.data.settings import load_settings
from tui.widgets.nav import ModelNav
from tui.widgets.status import StatusPanel, remote_toggle_label
from tests.support import Harness
from tests.test_dashboard import render_text
from tests.test_footer import shown_actions


class RemoteToggleLabelTests(unittest.TestCase):
    def test_status_render_keeps_next_launch_flag(self) -> None:
        panel = StatusPanel()
        self.assertIn("NEXT LAUNCH [LOCAL]", render_text(panel.render()))
        panel.next_remote = True
        self.assertIn("NEXT LAUNCH [REMOTE]", render_text(panel.render()))

    def test_switch_caption_names_bind_address(self) -> None:
        self.assertIn("localhost only", remote_toggle_label(False))
        self.assertIn("0.0.0.0", remote_toggle_label(True))
        self.assertIn("(R)", remote_toggle_label(False))

    def test_footer_keeps_remote_next_to_run_actions(self) -> None:
        actions = [b.action for b in build_app_bindings() if b.show]
        self.assertIn("toggle_remote", actions)
        self.assertEqual(actions[actions.index("stop") + 1], "toggle_remote")


class RemoteToggleAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    async def test_status_switch_is_mounted(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.2)
            switch = app.query_one("#remote-switch", Switch)
            self.assertTrue(switch.display)
            self.assertFalse(switch.value)
            self.assertIn("Remote OFF", render_text(app.query_one("#remote-switch-label").render()))

    async def test_hotkey_and_switch_stay_in_sync(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.2)
            nav = app.query_one(ModelNav)
            nav.focus()
            await pilot.press("r")
            await pilot.pause(0.1)
            self.assertTrue(app.remote_launch)
            self.assertTrue(app.query_one("#remote-switch", Switch).value)
            self.assertTrue(load_settings(self.harness.paths.settings_json).remote_launch)
            self.assertIn("Remote ON", render_text(app.query_one("#remote-switch-label").render()))

            app.query_one("#remote-switch", Switch).toggle()
            await pilot.pause(0.1)
            self.assertFalse(app.remote_launch)
            self.assertFalse(app.query_one("#remote-switch", Switch).value)

    async def test_narrow_footer_still_shows_remote(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.2)
            app.query_one(ModelNav).focus()
            await pilot.pause(0.1)
            self.assertIn("toggle_remote", shown_actions(app))

    async def test_launch_passes_remote_flag(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.2)
            nav = app.query_one(ModelNav)
            nav.highlighted = 0
            nav.focus()
            app.remote_launch = False
            with patch("tui.app.save_settings"):
                await pilot.press("r")
            with (
                patch("tui.app.prepare_launch") as prep,
                patch("tui.app.launch_background"),
                patch.object(app, "_reload_registry"),
                patch.object(app, "_refresh_pid"),
            ):
                prep.return_value = SimpleNamespace()
                await pilot.press("l")
                await pilot.pause(0.4)
            self.assertTrue(prep.call_args.kwargs.get("remote"))


if __name__ == "__main__":
    unittest.main()
