"""Tests for the model/preset editor screens."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from tui.data.models_json import ModelConfig, Registry
from tui.screens.editors import ProfileEditor


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


if __name__ == "__main__":
    unittest.main()
