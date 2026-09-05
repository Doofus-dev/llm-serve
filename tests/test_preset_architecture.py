"""Tests for preset-only architecture (identity in models.json, runtime in presets.json)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.models_json import load_registry, validate_display_name
from tui.data.preset_template import DEFAULT_PRESET_PARAMS, PRESET_PARAM_KEYS
from tui.data.presets import (
    PresetStore,
    get_active_slot,
    get_preset,
    load_presets,
    merge_identity_and_preset,
    seed_default_preset,
)


class PresetArchitectureTests(unittest.TestCase):
    def test_load_registry_migrates_runtime_to_presets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_path = root / "models.json"
            presets_path = root / "presets.json"
            models_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen38": {
                                "display": "Qwen 3.8",
                                "file": "a/model.gguf",
                                "port": 8081,
                                "host": "127.0.0.1",
                                "gpu_layers": 80,
                                "ctx": 16384,
                                "active_quant": "Q4_K_M",
                                "quants": {
                                    "Q4_K_M": {"filename": "model.gguf", "file": "a/model.gguf"}
                                },
                            }
                        }
                    }
                )
            )

            reg = load_registry(models_path)
            cfg = reg.models["qwen38"]
            self.assertNotIn("gpu_layers", cfg.params)
            self.assertNotIn("ctx", cfg.params)
            self.assertEqual(cfg.display, "Qwen 3.8")

            self.assertTrue(presets_path.exists())
            store = load_presets(presets_path)
            self.assertEqual(get_active_slot(store, "qwen38", "Q4_K_M"), 1)
            preset = get_preset(store, "qwen38", "Q4_K_M", 1)
            assert preset is not None
            self.assertEqual(preset.params.get("gpu_layers"), 80)
            self.assertEqual(preset.params.get("ctx"), 16384)

            merged = merge_identity_and_preset(cfg.params, preset)
            self.assertEqual(merged["gpu_layers"], 80)
            self.assertEqual(merged["port"], 8081)

    def test_seed_default_preset_uses_fixed_template(self) -> None:
        store = PresetStore()
        created = seed_default_preset(store, "m", "Q4_K_M", runtime_seed={"ctx": 99999})
        self.assertTrue(created)
        preset = get_preset(store, "m", "Q4_K_M", 1)
        assert preset is not None
        self.assertEqual(preset.params.get("ctx"), 99999)
        self.assertEqual(preset.params.get("gpu_layers"), DEFAULT_PRESET_PARAMS["gpu_layers"])
        for key in PRESET_PARAM_KEYS:
            self.assertIn(key, preset.params)

    def test_validate_display_name_unique(self) -> None:
        reg = load_registry(Path(__file__).parent.parent / "models.json")
        self.assertIsNone(validate_display_name(reg, "Unique Name XYZ"))
        first = next(iter(reg.models.values()))
        self.assertIsNotNone(validate_display_name(reg, first.display))


if __name__ == "__main__":
    unittest.main()
