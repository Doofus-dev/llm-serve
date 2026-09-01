"""Tests for preset slot allocation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.presets import (
    MAX_PRESETS_PER_MODEL,
    PresetStore,
    next_free_slot,
    set_preset,
)


class NextFreeSlotTests(unittest.TestCase):
    def test_first_slot_on_empty_model(self) -> None:
        store = PresetStore()
        self.assertEqual(next_free_slot(store, "qwen36", "Q4_K_M"), 1)

    def test_skips_used_slots(self) -> None:
        store = PresetStore()
        set_preset(store, "qwen36", "Q4_K_M", 1, "fast", {"ctx": 8192})
        set_preset(store, "qwen36", "Q4_K_M", 2, "long", {"ctx": 65536})
        self.assertEqual(next_free_slot(store, "qwen36", "Q4_K_M"), 3)

    def test_fills_gap(self) -> None:
        store = PresetStore()
        set_preset(store, "qwen36", "IQ2_S", 1, "a", {})
        set_preset(store, "qwen36", "IQ2_S", 3, "c", {})
        self.assertEqual(next_free_slot(store, "qwen36", "IQ2_S"), 2)

    def test_full_model_returns_none(self) -> None:
        store = PresetStore()
        for slot in range(1, MAX_PRESETS_PER_MODEL + 1):
            set_preset(store, "qwen36", "Q8_0", slot, f"slot-{slot}", {})
        self.assertIsNone(next_free_slot(store, "qwen36", "Q8_0"))

    def test_quants_are_independent(self) -> None:
        store = PresetStore()
        set_preset(store, "qwen36", "IQ2_S", 1, "iq", {})
        self.assertEqual(next_free_slot(store, "qwen36", "Q4_K_M"), 1)


if __name__ == "__main__":
    unittest.main()
