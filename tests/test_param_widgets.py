"""Tests for editor param widget value mapping."""

from __future__ import annotations

import unittest

from tui.data.param_widgets import (
    HIDDEN_EDITOR_PARAMS,
    coerce_text_param_value,
    is_select_param,
    read_field_value,
    select_initial_value,
    select_to_stored,
)


class ParamWidgetsTests(unittest.TestCase):
    def test_file_hidden_from_editor(self) -> None:
        self.assertIn("file", HIDDEN_EDITOR_PARAMS)

    def test_select_params_registered(self) -> None:
        for param in (
            "cache_k",
            "cache_v",
            "jinja",
            "flash_attn",
            "thinking",
            "reasoning_format",
            "mtp",
        ):
            self.assertTrue(is_select_param(param))

    def test_cache_type_round_trip(self) -> None:
        self.assertEqual(select_initial_value("cache_k", "q8_0"), "q8_0")
        self.assertEqual(select_to_stored("cache_k", "q4_0"), "q4_0")

    def test_optional_blank_params(self) -> None:
        self.assertIsNone(select_initial_value("thinking", ""))
        self.assertIsNone(select_initial_value("reasoning_format", ""))
        self.assertIsNone(select_initial_value("reasoning_format", False))
        from textual.widgets import Select

        self.assertEqual(select_to_stored("thinking", Select.NULL), "")
        self.assertEqual(select_to_stored("reasoning_format", Select.NULL), "")

    def test_mtp_labels_store_zero_one(self) -> None:
        self.assertEqual(select_initial_value("mtp", 0), "0")
        self.assertEqual(select_initial_value("mtp", "1"), "1")
        self.assertEqual(select_to_stored("mtp", "1"), "1")

    def test_coerce_numeric_text_fields(self) -> None:
        self.assertEqual(coerce_text_param_value("32768"), 32768)
        self.assertEqual(coerce_text_param_value("0.7"), 0.7)
        self.assertEqual(coerce_text_param_value("on"), "on")

    def test_read_field_value_from_select(self) -> None:
        class FakeSelect:
            value = "off"

        self.assertEqual(read_field_value("jinja", FakeSelect()), "off")


if __name__ == "__main__":
    unittest.main()
