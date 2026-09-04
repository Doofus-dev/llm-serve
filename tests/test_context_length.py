"""Tests for model context length helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.context_length import (
    cap_context_value,
    clamp_preset_params,
    context_length_options,
    fmt_ctx_range,
    hub_min_context_options,
)
from tui.data.gguf import UINT32, apply_architecture_from_gguf
from tui.data.hf import HubRepo
from tui.data.models_json import clamp_preset_contexts, load_registry, sync_gguf_architecture
from tui.data.presets import PresetStore, get_preset, load_presets, seed_default_preset
from tests.test_gguf import _pack_kv_string, _pack_kv_u32, write_gguf


class ContextLengthTests(unittest.TestCase):
    def test_hub_repo_parses_context_length(self) -> None:
        repo = HubRepo.from_json(
            {
                "id": "bartowski/Hermes-4-14B-GGUF",
                "downloads": 1,
                "likes": 0,
                "gguf": {"context_length": 40960, "total": 9_000_000_000},
            }
        )
        self.assertEqual(repo.context_length, 40960)

    def test_context_length_options_caps_at_model_max(self) -> None:
        self.assertEqual(context_length_options(40_960), [32_768, 40_960])
        self.assertEqual(context_length_options(32_768), [32_768])

    def test_hub_min_context_options_are_or_above(self) -> None:
        options = hub_min_context_options()
        self.assertIn(("32k+", 32_768), options)
        self.assertIn(("256k+", 262_144), options)

    def test_clamp_preset_params(self) -> None:
        capped = clamp_preset_params({"ctx": 65_000, "gpu_layers": 99}, 40_960)
        self.assertEqual(capped["ctx"], 40_960)
        self.assertEqual(capped["gpu_layers"], 99)

    def test_cap_context_value(self) -> None:
        self.assertEqual(cap_context_value(65_000, 40_960), 40_960)
        self.assertEqual(cap_context_value(32_768, None), 32_768)

    def test_fmt_ctx_range(self) -> None:
        self.assertEqual(fmt_ctx_range(32_768, 40_960), "32k / max 40k")
        self.assertEqual(fmt_ctx_range(32_768, None), "32k")

    def test_apply_architecture_stores_context_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            write_gguf(
                path,
                [
                    _pack_kv_string("general.architecture", "qwen3"),
                    _pack_kv_u32("qwen3.context_length", 40960),
                    _pack_kv_u32("qwen3.block_count", 40),
                ],
            )
            params: dict = {}
            apply_architecture_from_gguf(params, path, replace_cloned=True)
            self.assertEqual(params["total_layers"], 40)
            self.assertEqual(params["context_length"], 40960)

    def test_seed_default_preset_respects_max_ctx(self) -> None:
        store = PresetStore()
        created = seed_default_preset(
            store,
            "m",
            "Q5_K_M",
            runtime_seed={"ctx": 65_000},
            max_ctx=40_960,
        )
        self.assertTrue(created)
        preset = get_preset(store, "m", "Q5_K_M", 1)
        assert preset is not None
        self.assertEqual(preset.params["ctx"], 40_960)

    def test_sync_gguf_architecture_updates_context_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir()
            gguf = models_dir / "demo.gguf"
            write_gguf(
                gguf,
                [
                    _pack_kv_string("general.architecture", "qwen3"),
                    _pack_kv_u32("qwen3.context_length", 40960),
                    _pack_kv_u32("qwen3.block_count", 40),
                ],
            )
            path = root / "models.json"
            path.write_text(
                '{"models": {"demo": {"file": "demo.gguf", "total_layers": 28}}, "aliases": {}}'
            )
            changes = sync_gguf_architecture(path, models_dir)
            self.assertIn(("demo", "total_layers", 28, 40), changes)
            self.assertIn(("demo", "context_length", None, 40960), changes)
            reg = load_registry(path)
            self.assertEqual(reg.models["demo"].params["context_length"], 40960)

    def test_clamp_preset_contexts_on_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_path = root / "models.json"
            presets_path = root / "presets.json"
            models_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "hermes": {
                                "file": "bartowski/model.gguf",
                                "context_length": 40960,
                                "active_quant": "Q5_K_M",
                                "quants": {
                                    "Q5_K_M": {
                                        "filename": "model.gguf",
                                        "file": "bartowski/model.gguf",
                                    }
                                },
                            }
                        },
                        "aliases": {},
                    }
                )
            )
            presets_path.write_text(
                json.dumps(
                    {
                        "hermes": {
                            "Q5_K_M": {
                                "1": {"name": "big", "params": {"ctx": 65000, "gpu_layers": 99}},
                                "2": {"name": "ok", "params": {"ctx": 32768, "gpu_layers": 99}},
                            }
                        },
                        "_active": {"hermes": {"Q5_K_M": 1}},
                    }
                )
            )
            clamped = clamp_preset_contexts(models_path, presets_path)
            self.assertEqual(clamped, [("hermes", "Q5_K_M", 1, 65000, 40960)])
            store = load_presets(presets_path)
            preset = get_preset(store, "hermes", "Q5_K_M", 1)
            assert preset is not None
            self.assertEqual(preset.params["ctx"], 40960)


if __name__ == "__main__":
    unittest.main()
