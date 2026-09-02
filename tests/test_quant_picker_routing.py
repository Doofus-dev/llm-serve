"""Regression tests for model-card quant-picker routing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tui.app import AliasNav, LLMServeApp, ModelNav
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
    def _select_alias(nav: AliasNav, alias_name: str) -> None:
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
                model_name = next(iter(app.registry.models))
                self._select_model(nav, model_name)

                await pilot.press("p")
                await pilot.pause()

                self.assertIsInstance(app.screen, QuantPickerScreen)
                self.assertTrue(app.screen.context_options)

    async def test_duplicate_quant_ids_have_unique_file_rows(self) -> None:
        app = LLMServeApp()
        with patch.object(QuantPickerScreen, "_fetch_files", lambda self, load_id: None):
            async with app.run_test(size=(120, 45)) as pilot:
                nav = app.query_one(ModelNav)
                self._select_model(nav, next(iter(app.registry.models)))
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
            model_name = next(iter(app.registry.models))
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
            nav = app.query_one(AliasNav)
            alias_name = next(iter(app.registry.aliases))
            self._select_alias(nav, alias_name)
            nav.focus()
            model_names = list(app.registry.models)
            current = app.registry.aliases[alias_name].model
            expected = model_names[(model_names.index(current) + 1) % len(model_names)]

            with (
                patch("tui.app.save_registry") as save,
                patch.object(app, "_reload_registry"),
            ):
                await pilot.press("right")
                await pilot.pause()

            self.assertEqual(app.registry.aliases[alias_name].model, expected)
            self.assertIsNone(app.registry.aliases[alias_name].preset_slot)
            save.assert_called_once()

    async def test_number_key_pins_and_unpins_alias_preset(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 45)) as pilot:
            nav = app.query_one(AliasNav)
            alias_name = next(iter(app.registry.aliases))
            self._select_alias(nav, alias_name)
            nav.focus()
            target = app.registry.aliases[alias_name]
            quant = app._model_active_quant(target.model)
            set_preset(
                app.preset_store,
                target.model,
                quant,
                5,
                "Alias preset",
                {"ctx": 32_768},
            )

            with (
                patch("tui.app.save_registry"),
                patch.object(app, "_reload_registry"),
            ):
                await pilot.press("5")
                self.assertEqual(target.quant, quant)
                self.assertEqual(target.preset_slot, 5)
                nav.refresh_aliases()
                selected_option = nav.get_option_at_index(nav.highlighted)
                self.assertIn("  5  ←/→", str(selected_option.prompt))
                self.assertNotIn("Alias preset", str(selected_option.prompt))

                await pilot.press("5")
                self.assertIsNone(target.quant)
                self.assertIsNone(target.preset_slot)
                nav.refresh_aliases()
                selected_option = nav.get_option_at_index(nav.highlighted)
                self.assertNotIn("  5  ←/→", str(selected_option.prompt))

    async def test_tab_moves_focus_from_models_to_aliases(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 45)) as pilot:
            self.assertIsInstance(app.focused, ModelNav)

            await pilot.press("tab")

            self.assertIsInstance(app.focused, AliasNav)

    async def test_launch_from_pinned_alias_preserves_alias_name(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 45)) as pilot:
            nav = app.query_one(AliasNav)
            alias_name = next(iter(app.registry.aliases))
            self._select_alias(nav, alias_name)
            nav.focus()
            target = app.registry.aliases[alias_name]
            target.quant = app._model_active_quant(target.model)
            target.preset_slot = get_active_slot(
                app.preset_store,
                target.model,
                target.quant,
            )

            with (
                patch(
                    "tui.app.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ) as run,
                patch.object(app, "_reload_registry"),
                patch.object(app, "_refresh_pid"),
            ):
                await pilot.press("l")
                await pilot.pause()

            self.assertEqual(run.call_args.args[0][1], alias_name)

    async def test_remote_hotkey_adds_flag_to_next_launch(self) -> None:
        app = LLMServeApp()
        async with app.run_test(size=(120, 45)) as pilot:
            nav = app.query_one(ModelNav)
            self._select_model(nav, next(iter(app.registry.models)))
            nav.focus()

            await pilot.press("r")
            self.assertTrue(app.remote_launch)

            with (
                patch(
                    "tui.app.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ) as run,
                patch.object(app, "_reload_registry"),
                patch.object(app, "_refresh_pid"),
            ):
                await pilot.press("l")
                await pilot.pause()

            command = run.call_args.args[0]
            self.assertIn("--remote", command)


if __name__ == "__main__":
    unittest.main()
