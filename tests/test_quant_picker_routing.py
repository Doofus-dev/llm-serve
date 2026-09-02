"""Regression tests for model-card quant-picker routing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tui.app import LLMServeApp, ModelNav
from tui.data.hf import HubFile
from tui.data.presets import get_active_slot, set_preset
from tui.screens.quant_picker import QuantPickerScreen


class QuantPickerRoutingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _select_model(nav: ModelNav, model_name: str) -> None:
        for index in range(nav.option_count):
            option = nav.get_option_at_index(index)
            if nav._option_data.get(option.id or "") == ("model", model_name):
                nav.highlighted = index
                return
        raise AssertionError(f"model card not found: {model_name}")

    @staticmethod
    def _select_alias(nav: ModelNav, alias_name: str) -> None:
        for index in range(nav.option_count):
            option = nav.get_option_at_index(index)
            if nav._option_data.get(option.id or "") == ("alias", alias_name):
                nav.highlighted = index
                return
        raise AssertionError(f"alias not found: {alias_name}")

    async def test_quant_action_on_model_card_opens_quant_picker(self) -> None:
        app = LLMServeApp()
        with patch.object(QuantPickerScreen, "_fetch_files", lambda self, load_id: None):
            async with app.run_test(size=(120, 45)) as pilot:
                nav = app.query_one(ModelNav)
                self._select_model(nav, "q5-14b-bartowski")

                await pilot.press("p")
                await pilot.pause()

                self.assertIsInstance(app.screen, QuantPickerScreen)
                self.assertEqual(app.screen.context_options, [32_768, 40_960])

    async def test_duplicate_quant_ids_have_unique_file_rows(self) -> None:
        app = LLMServeApp()
        with patch.object(QuantPickerScreen, "_fetch_files", lambda self, load_id: None):
            async with app.run_test(size=(120, 45)) as pilot:
                nav = app.query_one(ModelNav)
                self._select_model(nav, "q5-14b-bartowski")
                await pilot.press("p")
                await pilot.pause()

                screen = app.screen
                self.assertIsInstance(screen, QuantPickerScreen)
                screen._files = [
                    HubFile(path="Model-Q8_0.gguf", size=100),
                    HubFile(path="mtp-Model-Q8_0.gguf", size=50),
                ]
                screen._refresh_table()

                self.assertEqual(screen.query_one("#quant-table").row_count, 2)

    async def test_number_key_activates_matching_preset_on_model_card(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 45)) as pilot:
            nav = app.query_one(ModelNav)
            model_name = "q5-14b-bartowski"
            self._select_model(nav, model_name)
            quant = app._model_active_quant(model_name)
            set_preset(
                app.preset_store,
                model_name,
                quant,
                5,
                "Keyboard preset",
                {"ctx": 32_768},
            )

            with (
                patch("tui.app.save_presets"),
                patch.object(app, "_reload_registry"),
            ):
                await pilot.press("5")
                await pilot.pause()

            self.assertEqual(get_active_slot(app.preset_store, model_name, quant), 5)

    async def test_arrow_key_immediately_cycles_alias_target(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 45)) as pilot:
            nav = app.query_one(ModelNav)
            alias_name = next(iter(app.registry.aliases))
            self._select_alias(nav, alias_name)
            nav.focus()
            model_names = list(app.registry.models)
            current = app.registry.aliases[alias_name]
            expected = model_names[(model_names.index(current) + 1) % len(model_names)]

            with (
                patch("tui.app.save_registry") as save,
                patch.object(app, "_reload_registry"),
            ):
                await pilot.press("right")
                await pilot.pause()

            self.assertEqual(app.registry.aliases[alias_name], expected)
            save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
