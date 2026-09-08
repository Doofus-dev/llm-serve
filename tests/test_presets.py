"""Tests for preset slot allocation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tui.data.models_json import AliasTarget, load_registry, remap_alias_preset_slots
from tui.data.presets import (
    MAX_PRESETS_PER_MODEL,
    PresetStore,
    compact_preset_slots,
    delete_preset,
    get_active_slot,
    get_preset,
    next_free_slot,
    set_active_preset,
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


class CompactPresetSlotTests(unittest.TestCase):
    def test_renumbers_gaps_and_remaps_active(self) -> None:
        store = PresetStore()
        set_preset(store, "qwen36", "Q2_K", 1, "default", {"ctx": 1})
        set_preset(store, "qwen36", "Q2_K", 3, "deep", {"ctx": 2})
        set_preset(store, "qwen36", "Q2_K", 5, "think", {"ctx": 3})
        set_active_preset(store, "qwen36", "Q2_K", 5)

        mapping = compact_preset_slots(store, "qwen36", "Q2_K")

        self.assertEqual(mapping, {1: 1, 3: 2, 5: 3})
        self.assertEqual(sorted(store.presets["qwen36"]["Q2_K"]), [1, 2, 3])
        self.assertEqual(get_preset(store, "qwen36", "Q2_K", 2).name, "deep")
        self.assertEqual(get_preset(store, "qwen36", "Q2_K", 3).name, "think")
        self.assertEqual(get_preset(store, "qwen36", "Q2_K", 3).slot, 3)
        self.assertEqual(get_active_slot(store, "qwen36", "Q2_K"), 3)
        self.assertEqual(next_free_slot(store, "qwen36", "Q2_K"), 4)

    def test_already_compact_is_identity(self) -> None:
        store = PresetStore()
        set_preset(store, "qwen36", "Q4_K_M", 1, "a", {})
        set_preset(store, "qwen36", "Q4_K_M", 2, "b", {})
        mapping = compact_preset_slots(store, "qwen36", "Q4_K_M")
        self.assertEqual(mapping, {1: 1, 2: 2})
        self.assertEqual(sorted(store.presets["qwen36"]["Q4_K_M"]), [1, 2])

    def test_delete_compacts_remaining_slots(self) -> None:
        store = PresetStore()
        set_preset(store, "qwen36", "Q2_K", 1, "default", {})
        set_preset(store, "qwen36", "Q2_K", 2, "mid", {})
        set_preset(store, "qwen36", "Q2_K", 3, "deep", {})
        set_active_preset(store, "qwen36", "Q2_K", 3)

        mapping = delete_preset(store, "qwen36", "Q2_K", 2)

        self.assertEqual(mapping, {1: 1, 3: 2})
        self.assertEqual(sorted(store.presets["qwen36"]["Q2_K"]), [1, 2])
        self.assertEqual(get_preset(store, "qwen36", "Q2_K", 2).name, "deep")
        self.assertEqual(get_active_slot(store, "qwen36", "Q2_K"), 2)

    def test_delete_active_clears_then_compacts(self) -> None:
        store = PresetStore()
        set_preset(store, "qwen36", "Q2_K", 1, "default", {})
        set_preset(store, "qwen36", "Q2_K", 3, "deep", {})
        set_active_preset(store, "qwen36", "Q2_K", 1)

        delete_preset(store, "qwen36", "Q2_K", 1)

        self.assertEqual(sorted(store.presets["qwen36"]["Q2_K"]), [1])
        self.assertEqual(get_preset(store, "qwen36", "Q2_K", 1).name, "deep")
        self.assertIsNone(get_active_slot(store, "qwen36", "Q2_K"))


class RemapAliasPresetSlotTests(unittest.TestCase):
    def test_rewrites_and_unpins_deleted_slot(self) -> None:
        aliases = {
            "keep": AliasTarget(model="qwen36", quant="Q2_K", preset_slot=5),
            "gone": AliasTarget(model="qwen36", quant="Q2_K", preset_slot=3),
            "other": AliasTarget(model="qwen36", quant="Q4_K_M", preset_slot=3),
        }

        changed = remap_alias_preset_slots(
            aliases, "qwen36", "Q2_K", {1: 1, 5: 2}, deleted_slot=3
        )

        self.assertTrue(changed)
        self.assertEqual(aliases["keep"].preset_slot, 2)
        self.assertIsNone(aliases["gone"].preset_slot)
        self.assertIsNone(aliases["gone"].quant)
        self.assertEqual(aliases["other"].preset_slot, 3)

    def test_load_registry_compacts_gapped_presets_and_alias_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_path = root / "models.json"
            presets_path = root / "presets.json"
            models_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen36": {
                                "display": "Qwen",
                                "file": "model.gguf",
                                "active_quant": "Q2_K",
                                "quants": {"Q2_K": {"filename": "model.gguf"}},
                            }
                        },
                        "aliases": {
                            "think": {
                                "model": "qwen36",
                                "quant": "Q2_K",
                                "preset": 5,
                            }
                        },
                    }
                )
            )
            presets_path.write_text(
                json.dumps(
                    {
                        "_active": {"qwen36": {"Q2_K": 5}},
                        "qwen36": {
                            "Q2_K": {
                                "1": {"name": "default", "params": {"ctx": 1}},
                                "3": {"name": "deep", "params": {"ctx": 2}},
                                "5": {"name": "think", "params": {"ctx": 3}},
                            }
                        },
                    }
                )
            )

            reg = load_registry(models_path)
            saved = json.loads(presets_path.read_text())

            self.assertEqual(sorted(saved["qwen36"]["Q2_K"]), ["1", "2", "3"])
            self.assertEqual(saved["qwen36"]["Q2_K"]["2"]["name"], "deep")
            self.assertEqual(saved["qwen36"]["Q2_K"]["3"]["name"], "think")
            self.assertEqual(saved["_active"]["qwen36"]["Q2_K"], 3)
            self.assertEqual(
                reg.aliases["think"],
                AliasTarget(model="qwen36", quant="Q2_K", preset_slot=3),
            )
            self.assertEqual(
                json.loads(models_path.read_text())["aliases"]["think"]["preset"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
