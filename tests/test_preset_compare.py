"""Preset compare diffs and independent saves."""

from __future__ import annotations

import json
import unittest

from textual.widgets import Select

from tests.support import MODEL_SLUG, QUANT, Harness
from tui.data.preset_template import DEFAULT_PRESET_PARAMS
from tui.data.presets import (
    diff_preset_params,
    get_preset,
    load_presets,
    save_presets,
    set_preset,
)
from tui.screens.compare import PresetCompareScreen, format_diff_summary
from tui.screens.editors import ConfirmDialog, PresetEditor
from tui.widgets.nav import ModelNav


class PresetDiffTests(unittest.TestCase):
    def test_classifies_changed_added_removed(self) -> None:
        left = {"ctx": 32768, "gpu_layers": 99, "temp": 0.8}
        right = {"ctx": 131072, "gpu_layers": 99, "flash_attn": "on"}
        diff = diff_preset_params(left, right)
        self.assertEqual(diff["changed"]["ctx"], ("32768", "131072"))
        self.assertEqual(diff["added"]["flash_attn"], ("", "on"))
        self.assertEqual(diff["removed"]["temp"], ("0.8", ""))
        self.assertNotIn("gpu_layers", diff["changed"])

    def test_empty_matches_missing(self) -> None:
        diff = diff_preset_params({"thinking": ""}, {})
        self.assertEqual(diff["added"], {})
        self.assertEqual(diff["removed"], {})
        self.assertEqual(diff["changed"], {})

    def test_summary_lists_differing_params(self) -> None:
        summary = format_diff_summary(
            "default",
            "max-ctx",
            {"ctx": 32768, "gpu_layers": 99},
            {"ctx": 131072, "gpu_layers": 99},
        )
        self.assertIn("name", summary)
        self.assertIn("ctx", summary)
        self.assertIn("32768", summary)
        self.assertIn("131072", summary)
        self.assertNotIn("gpu_layers", summary)


class PresetCompareScreenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.harness = Harness()
        store = load_presets(self.harness.paths.presets_json)
        max_params = dict(DEFAULT_PRESET_PARAMS)
        max_params["ctx"] = 16384
        max_params["gpu_layers"] = 40
        set_preset(store, MODEL_SLUG, QUANT, 2, "max-ctx", max_params)
        save_presets(self.harness.paths.presets_json, store)

    def tearDown(self) -> None:
        self.harness.cleanup()

    async def test_compare_highlights_and_saves_independently(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app.query_one(ModelNav).focus()
            app.action_compare_presets()
            await pilot.pause(0.5)
            screen = app.screen
            self.assertIsInstance(screen, PresetCompareScreen)
            self.assertEqual(screen.left_slot, 1)
            self.assertEqual(screen.right_slot, 2)

            summary = str(screen.query_one("#diff-summary").content)
            self.assertIn("ctx", summary)
            self.assertIn("gpu_layers", summary)
            self.assertTrue(screen.query_one("#cmp-left-field_ctx").has_class("diff-changed"))
            self.assertTrue(screen.query_one("#cmp-right-field_ctx").has_class("diff-changed"))

            screen.query_one("#cmp-left-input_ctx").value = "8192"
            screen.query_one("#cmp-right-input_gpu_layers").value = "20"
            await pilot.pause(0.2)
            screen.query_one("#cmp-left-editor").save()
            await pilot.pause(0.2)
            screen.query_one("#cmp-right-editor").save()
            await pilot.pause(0.2)

        store = load_presets(self.harness.paths.presets_json)
        left = get_preset(store, MODEL_SLUG, QUANT, 1)
        right = get_preset(store, MODEL_SLUG, QUANT, 2)
        self.assertEqual(left.params["ctx"], 8192)
        self.assertEqual(left.params["gpu_layers"], DEFAULT_PRESET_PARAMS["gpu_layers"])
        self.assertEqual(right.params["gpu_layers"], 20)
        self.assertEqual(right.params["ctx"], 16384)
        self.assertEqual(left.name, "default")
        self.assertEqual(right.name, "max-ctx")

        on_disk = json.loads(self.harness.paths.presets_json.read_text())
        self.assertEqual(on_disk[MODEL_SLUG][QUANT]["1"]["params"]["ctx"], 8192)
        self.assertEqual(on_disk[MODEL_SLUG][QUANT]["2"]["params"]["gpu_layers"], 20)
        other = on_disk["other-model"][QUANT]["1"]["params"]
        self.assertEqual(other["ctx"], DEFAULT_PRESET_PARAMS["ctx"])
        self.assertEqual(other["gpu_layers"], DEFAULT_PRESET_PARAMS["gpu_layers"])

    async def test_compare_requires_two_presets(self) -> None:
        store = load_presets(self.harness.paths.presets_json)
        del store.presets[MODEL_SLUG][QUANT][2]
        save_presets(self.harness.paths.presets_json, store)
        app = self.harness.app()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.3)
            app.query_one(ModelNav).focus()
            app.action_compare_presets()
            await pilot.pause(0.2)
            self.assertNotIsInstance(app.screen, PresetCompareScreen)

    def _add_slot(self, slot: int, name: str) -> None:
        store = load_presets(self.harness.paths.presets_json)
        params = dict(DEFAULT_PRESET_PARAMS)
        set_preset(store, MODEL_SLUG, QUANT, slot, name, params)
        save_presets(self.harness.paths.presets_json, store)

    async def test_compare_blocks_dashboard_delete(self) -> None:
        app = self.harness.app()
        gguf = self.harness.paths.models_dir / "demo" / "Demo-Q4_K_M.gguf"
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app.query_one(ModelNav).focus()
            app.action_compare_presets()
            await pilot.pause(0.5)
            self.assertIsInstance(app.screen, PresetCompareScreen)
            actions = {
                binding.action
                for _key, binding, _enabled, _tooltip in app.screen.active_bindings.values()
            }
            self.assertNotIn("delete", actions)
            self.assertNotIn("launch", actions)
            self.assertNotIn("edit", actions)
            await pilot.press("d")
            await pilot.pause(0.2)
            self.assertIsInstance(app.screen, PresetCompareScreen)
            self.assertNotIsInstance(app.screen, ConfirmDialog)
            self.assertIn(MODEL_SLUG, app.registry.models)
            self.assertTrue(gguf.exists())

    async def test_compare_excludes_the_other_side_slot(self) -> None:
        self._add_slot(3, "coding")
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app.query_one(ModelNav).focus()
            app.action_compare_presets()
            await pilot.pause(0.5)
            screen = app.screen
            self.assertIsInstance(screen, PresetCompareScreen)
            left = screen.query_one("#cmp-left-slot", Select)
            with self.assertRaises(Exception):
                left.value = str(screen.right_slot)
            self.assertEqual(screen.query_one("#cmp-left-editor", PresetEditor).slot, 1)
            self.assertEqual(screen.query_one("#cmp-right-editor", PresetEditor).slot, 2)
            left.value = "3"
            await app.workers.wait_for_complete()
            await pilot.pause(0.3)
            self.assertEqual(screen.query_one("#cmp-left-editor", PresetEditor).slot, 3)
            self.assertEqual(screen.query_one("#cmp-right-editor", PresetEditor).slot, 2)
            right = screen.query_one("#cmp-right-slot", Select)
            with self.assertRaises(Exception):
                right.value = "3"
            right.value = "1"
            await app.workers.wait_for_complete()
            await pilot.pause(0.3)
            self.assertEqual(screen.query_one("#cmp-left-editor", PresetEditor).slot, 3)
            self.assertEqual(screen.query_one("#cmp-right-editor", PresetEditor).slot, 1)

    async def test_compare_updates_labels_after_rename_save(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app.query_one(ModelNav).focus()
            app.action_compare_presets()
            await pilot.pause(0.5)
            screen = app.screen
            self.assertIsInstance(screen, PresetCompareScreen)
            editor = screen.query_one("#cmp-left-editor", PresetEditor)
            self.assertIsNotNone(editor.name_input)
            editor.name_input.value = "renamed-default"
            editor.save()
            await pilot.pause(0.3)
            title = str(screen.query_one("#cmp-left-title").content)
            self.assertIn("renamed-default", title)
            prompt = str(screen.query_one("#cmp-left-slot #label").content)
            self.assertIn("renamed-default", prompt)
            stored = get_preset(
                load_presets(self.harness.paths.presets_json), MODEL_SLUG, QUANT, 1
            )
            self.assertEqual(stored.name, "renamed-default")

    async def test_compare_applies_latest_slot_during_replace(self) -> None:
        self._add_slot(3, "coding")
        self._add_slot(4, "fast")
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app.query_one(ModelNav).focus()
            app.action_compare_presets()
            await pilot.pause(0.5)
            screen = app.screen
            self.assertIsInstance(screen, PresetCompareScreen)
            left = screen.query_one("#cmp-left-slot", Select)
            left.value = "3"
            left.value = "4"
            await pilot.pause(0.2)
            await app.workers.wait_for_complete()
            await pilot.pause(0.3)
            self.assertEqual(screen.query_one("#cmp-left-editor", PresetEditor).slot, 4)
            self.assertEqual(screen.query_one("#cmp-right-editor", PresetEditor).slot, 2)

    async def test_compare_help_follows_field_focus(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app.query_one(ModelNav).focus()
            app.action_compare_presets()
            await pilot.pause(0.5)
            screen = app.screen
            self.assertIsInstance(screen, PresetCompareScreen)
            screen.query_one("#cmp-left-input_ctx").focus()
            await pilot.pause(0.2)
            screen.action_toggle_param_help()
            help_text = str(screen.query_one("#compare-param-help #param-help-text").content)
            self.assertIn("ctx", help_text)
            screen.query_one("#cmp-left-input_gpu_layers").focus()
            await pilot.pause(0.2)
            help_text = str(screen.query_one("#compare-param-help #param-help-text").content)
            self.assertIn("gpu_layers", help_text)
            self.assertNotIn("ctx", help_text.split("\n")[0])


if __name__ == "__main__":
    unittest.main()
