"""Regression tests for quant-row keyboard routing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tui.app import LLMServeApp, ModelTree
from tui.data.hf import HubFile
from tui.screens.quant_picker import QuantPickerScreen


class QuantPickerRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_edit_on_quant_row_opens_quant_picker(self) -> None:
        app = LLMServeApp()
        with patch.object(QuantPickerScreen, "_fetch_files", lambda self, load_id: None):
            async with app.run_test(size=(120, 45)) as pilot:
                tree = app.query_one(ModelTree)
                model = next(
                    node
                    for node in tree.root.children
                    if node.data == ("model", "q5-14b-bartowski")
                )
                quant = next(
                    node
                    for node in model.children
                    if node.data and node.data[0] == "quant"
                )
                tree.select_node(quant)

                await pilot.press("e")
                await pilot.pause()

                self.assertIsInstance(app.screen, QuantPickerScreen)
                self.assertEqual(app.screen.context_options, [32_768, 40_960])

    async def test_duplicate_quant_ids_have_unique_file_rows(self) -> None:
        app = LLMServeApp()
        with patch.object(QuantPickerScreen, "_fetch_files", lambda self, load_id: None):
            async with app.run_test(size=(120, 45)) as pilot:
                tree = app.query_one(ModelTree)
                model = next(
                    node
                    for node in tree.root.children
                    if node.data == ("model", "q5-14b-bartowski")
                )
                quant = next(
                    node
                    for node in model.children
                    if node.data and node.data[0] == "quant"
                )
                tree.select_node(quant)
                await pilot.press("e")
                await pilot.pause()

                screen = app.screen
                self.assertIsInstance(screen, QuantPickerScreen)
                screen._files = [
                    HubFile(path="Model-Q8_0.gguf", size=100),
                    HubFile(path="mtp-Model-Q8_0.gguf", size=50),
                ]
                screen._refresh_table()

                self.assertEqual(screen.query_one("#quant-table").row_count, 2)


if __name__ == "__main__":
    unittest.main()
