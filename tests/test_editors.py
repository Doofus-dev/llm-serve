"""Tests for the model/preset editor screens."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from textual.widgets import Button

from tests.support import Harness
from tui.data.models_json import ModelConfig, Registry
from tui.screens.editors import ParamHelpPanel, ProfileEditor
from tui.widgets.action_bar import ActionBar
from tui.widgets.nav import ModelNav


def _make_editor(port_value: str) -> ProfileEditor:
    reg = Registry()
    reg.models["m1"] = ModelConfig(name="m1", params={"display": "M1"})
    editor = ProfileEditor(
        model_name="m1",
        params={"display": "M1", "port": 8081, "host": "127.0.0.1", "notes": ""},
        registry=reg,
        on_save=Mock(),
        on_cancel=Mock(),
    )
    editor.display_input = Mock()
    editor.display_input.value = "M1"
    editor.port_input = Mock()
    editor.port_input.value = port_value
    editor.host_input = Mock()
    editor.host_input.value = "127.0.0.1"
    editor.notes_input = Mock()
    editor.notes_input.value = ""
    return editor


class ProfileEditorPortTests(unittest.TestCase):
    def test_non_numeric_port_aborts_save(self) -> None:
        editor = _make_editor("abc")
        with patch.object(ProfileEditor, "app", new_callable=Mock) as app:
            editor.save()
            self.assertEqual(editor.on_save_callback.call_count, 0)
            app.notify.assert_called_once()
            self.assertEqual(app.notify.call_args[0][0], "Port must be a number")

    def test_empty_port_aborts_save(self) -> None:
        editor = _make_editor("")
        with patch.object(ProfileEditor, "app", new_callable=Mock) as app:
            editor.save()
            self.assertEqual(editor.on_save_callback.call_count, 0)
            app.notify.assert_called_once()

    def test_valid_port_saves(self) -> None:
        editor = _make_editor("9090")
        editor.on_save_callback = Mock()
        with patch.object(ProfileEditor, "app", new_callable=Mock):
            editor.save()
            self.assertEqual(editor.on_save_callback.call_count, 1)
            self.assertEqual(editor.on_save_callback.call_args[0][0]["port"], 9090)


class EditorLayoutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    async def test_preset_save_cancel_visible_without_scrolling(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(100, 28)) as pilot:
            model_name = next(iter(app.registry.models))
            quant = app._model_active_quant(model_name)
            app._edit_preset(model_name, quant, 1)
            await pilot.pause()
            editor = app.query_one("#param-editor")
            save = editor.query_one("#save", Button)
            cancel = editor.query_one("#cancel", Button)
            bar = editor.query_one("ActionBar.editor-actions", ActionBar)
            scroll = editor.query_one("#editor-scroll")
            self.assertIs(bar.parent, editor)
            self.assertNotIn(save, list(scroll.query(Button)))
            self.assertGreater(save.region.height, 0)
            self.assertGreater(cancel.region.height, 0)
            self.assertLessEqual(bar.region.bottom, editor.region.bottom)
            app.query_one("#input_ctx").focus()
            await pilot.pause()
            app.action_toggle_param_help()
            await pilot.pause()
            help_panel = app.query_one("#help_ctx", ParamHelpPanel)
            self.assertIn("visible", help_panel.classes)
            self.assertGreater(save.region.height, 0)
            self.assertLessEqual(bar.region.bottom, editor.region.bottom)

    async def test_profile_editor_has_identity_toggles(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(100, 28)) as pilot:
            nav = app.query_one(ModelNav)
            model_name = next(iter(app.registry.models))
            for index in range(nav.option_count):
                option = nav.get_option_at_index(index)
                if nav._option_data.get(option.id or "") == ("model", model_name):
                    nav.highlighted = index
                    break
            nav.focus()
            await pilot.press("e")
            await pilot.pause()
            editor = app.query_one("#param-editor")
            self.assertTrue(editor.query_one("#select_vision_enabled"))
            self.assertTrue(editor.query_one("#select_mtp_enabled"))
            save = editor.query_one("#save", Button)
            self.assertGreater(save.region.height, 0)


if __name__ == "__main__":
    unittest.main()
