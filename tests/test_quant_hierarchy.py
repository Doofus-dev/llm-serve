"""Tests for quant parsing and models.json migration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.models_json import (
    add_or_update_quant,
    load_registry,
    migrate_models_json,
    set_active_quant,
)
from tui.data.presets import get_active_slot, get_preset, load_presets, set_preset
from tui.data.quant import family_display, quant_from_filename


class QuantParseTests(unittest.TestCase):
    def test_quant_from_filename(self) -> None:
        self.assertEqual(quant_from_filename("Qwen3.8-27B-IQ2_S.gguf"), "IQ2_S")
        self.assertEqual(quant_from_filename("Qwen3.6-27B-Q4_K_M.gguf"), "Q4_K_M")

    def test_family_display(self) -> None:
        self.assertEqual(
            family_display("bartowski/Qwen3.8-27B-GGUF", "Qwen3.8-27B-IQ2_S.gguf"),
            "Qwen 3.8",
        )


class ModelsMigrationTests(unittest.TestCase):
    def test_merge_same_repo_profiles(self) -> None:
        data = {
            "models": {
                "bartowski-qwen38-iq2": {
                    "file": "bartowski/Qwen3.8-27B-IQ2_S.gguf",
                    "gpu_layers": 99,
                    "ctx": 32768,
                    "source": {
                        "hub": "huggingface",
                        "repo": "bartowski/Qwen3.8-27B-GGUF",
                        "filename": "Qwen3.8-27B-IQ2_S.gguf",
                        "author": "bartowski",
                    },
                },
                "bartowski-qwen38-q4": {
                    "file": "bartowski/Qwen3.8-27B-Q4_K_M.gguf",
                    "gpu_layers": 80,
                    "ctx": 16384,
                    "source": {
                        "hub": "huggingface",
                        "repo": "bartowski/Qwen3.8-27B-GGUF",
                        "filename": "Qwen3.8-27B-Q4_K_M.gguf",
                        "author": "bartowski",
                    },
                },
            },
            "aliases": {"heavy": "bartowski-qwen38-q4"},
        }
        changed, remap = migrate_models_json(data)
        self.assertTrue(changed)
        self.assertEqual(len(data["models"]), 1)
        slug = next(iter(data["models"]))
        model = data["models"][slug]
        self.assertEqual(model["display"], "Qwen 3.8")
        self.assertIn("IQ2_S", model["quants"])
        self.assertIn("Q4_K_M", model["quants"])
        self.assertEqual(data["aliases"]["heavy"], slug)
        self.assertEqual(remap["bartowski-qwen38-iq2"], (slug, "IQ2_S"))

    def test_add_or_update_quant_appends_second_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir()
            json_path = root / "models.json"
            json_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen38-27b-bartowski": {
                                "display": "Qwen 3.8",
                                "active_quant": "IQ2_S",
                                "file": "bartowski/Qwen3.8-27B-IQ2_S.gguf",
                                "gpu_layers": 99,
                                "ctx": 32768,
                                "quants": {
                                    "IQ2_S": {
                                        "filename": "Qwen3.8-27B-IQ2_S.gguf",
                                        "file": "bartowski/Qwen3.8-27B-IQ2_S.gguf",
                                    }
                                },
                                "source": {
                                    "repo": "bartowski/Qwen3.8-27B-GGUF",
                                    "author": "bartowski",
                                    "filename": "Qwen3.8-27B-IQ2_S.gguf",
                                },
                            }
                        }
                    }
                )
            )
            (models_dir / "bartowski").mkdir()
            (models_dir / "bartowski/Qwen3.8-27B-Q4_K_M.gguf").write_bytes(b"x")

            slug, qid = add_or_update_quant(
                json_path,
                repo_id="bartowski/Qwen3.8-27B-GGUF",
                filename="Qwen3.8-27B-Q4_K_M.gguf",
                file_rel="bartowski/Qwen3.8-27B-Q4_K_M.gguf",
                source={
                    "hub": "huggingface",
                    "repo": "bartowski/Qwen3.8-27B-GGUF",
                    "filename": "Qwen3.8-27B-Q4_K_M.gguf",
                    "author": "bartowski",
                },
                base_params={"gpu_layers": 99, "ctx": 32768},
                models_dir=models_dir,
            )
            self.assertEqual(slug, "qwen38-27b-bartowski")
            self.assertEqual(qid, "Q4_K_M")
            reg = load_registry(json_path, models_dir=models_dir)
            cfg = reg.models[slug]
            self.assertEqual(cfg.active_quant, "Q4_K_M")
            self.assertEqual(cfg.file, "bartowski/Qwen3.8-27B-Q4_K_M.gguf")

    def test_set_active_quant_switches_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            json_path = root / "models.json"
            json_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "m": {
                                "display": "Test",
                                "active_quant": "IQ2_S",
                                "file": "a/IQ2.gguf",
                                "quants": {
                                    "IQ2_S": {"filename": "IQ2.gguf", "file": "a/IQ2.gguf"},
                                    "Q4_K_M": {"filename": "Q4.gguf", "file": "a/Q4.gguf"},
                                },
                                "source": {"repo": "a/r", "author": "a", "filename": "IQ2.gguf"},
                            }
                        }
                    }
                )
            )
            self.assertTrue(set_active_quant(json_path, "m", "Q4_K_M", models_dir=models_dir))
            reg = load_registry(json_path, models_dir=models_dir)
            self.assertEqual(reg.models["m"].file, "a/Q4.gguf")
            self.assertEqual(reg.models["m"].active_quant, "Q4_K_M")


class PresetQuantTests(unittest.TestCase):
    def test_presets_nested_by_quant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.json"
            path.write_text(
                json.dumps(
                    {
                        "_active": {"m": {"IQ2_S": 1}},
                        "m": {
                            "IQ2_S": {
                                "1": {"name": "fast", "overrides": {"ctx": 8192}}
                            }
                        },
                    }
                )
            )
            store = load_presets(path)
            self.assertIsNotNone(get_preset(store, "m", "IQ2_S", 1))
            self.assertEqual(get_active_slot(store, "m", "IQ2_S"), 1)
            set_preset(store, "m", "Q4_K_M", 1, "other", {"ctx": 4096})
            self.assertIsNone(get_preset(store, "m", "IQ2_S", 2))
            self.assertIsNotNone(get_preset(store, "m", "Q4_K_M", 1))


if __name__ == "__main__":
    unittest.main()
