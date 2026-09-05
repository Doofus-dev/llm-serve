"""Tests for quant parsing and models.json migration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.models_json import (
    AliasTarget,
    add_or_update_quant,
    delete_model,
    load_registry,
    migrate_models_json,
    resolve_model_key,
    set_active_quant,
    unshared_model_file_paths,
)
from tui.data.presets import get_active_slot, get_preset, load_presets, set_preset
from tui.data.quant import family_display, quant_from_filename


class QuantParseTests(unittest.TestCase):
    def test_quant_from_filename(self) -> None:
        self.assertEqual(quant_from_filename("Qwen3.8-27B-IQ2_S.gguf"), "IQ2_S")
        self.assertEqual(quant_from_filename("Qwen3.6-27B-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(quant_from_filename("Qwen_Qwen3.6-27B-Q2_K.gguf"), "Q2_K")
        self.assertEqual(quant_from_filename("Qwen_Qwen3.6-27B-Q6_K.gguf"), "Q6_K")

    def test_family_display(self) -> None:
        self.assertEqual(
            family_display("bartowski/Qwen3.8-27B-GGUF", "Qwen3.8-27B-IQ2_S.gguf"),
            "Qwen 3.8",
        )


class ModelDeletionTests(unittest.TestCase):
    def test_unshared_model_files_exclude_files_used_by_other_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir()
            shared = models_dir / "shared.gguf"
            unique = models_dir / "unique.gguf"
            shared.write_bytes(b"shared")
            unique.write_bytes(b"unique")
            registry_path = root / "models.json"
            registry_path.write_text(json.dumps({
                "models": {
                    "first": {
                        "file": "unique.gguf",
                        "quants": {
                            "Q4": {"file": "unique.gguf"},
                            "Q5": {"file": "shared.gguf"},
                        },
                    },
                    "second": {
                        "file": "shared.gguf",
                        "quants": {"Q5": {"file": "shared.gguf"}},
                    },
                },
                "aliases": {},
            }))

            reg = load_registry(registry_path, models_dir=models_dir)

            self.assertEqual(
                unshared_model_file_paths(reg, "first", models_dir),
                {unique.resolve()},
            )

    def test_delete_model_removes_aliases_targeting_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(json.dumps({
                "models": {
                    "first": {"file": "first.gguf"},
                    "second": {"file": "second.gguf"},
                },
                "aliases": {"default": "first", "other": "second"},
            }))

            delete_model(path, "first")
            reg = load_registry(path)

            self.assertNotIn("first", reg.models)
            self.assertNotIn("default", reg.aliases)
            self.assertEqual(reg.aliases["other"], AliasTarget(model="second"))

    def test_load_registry_migrates_legacy_alias_and_preserves_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(json.dumps({
                "models": {"model": {"file": "model.gguf"}},
                "aliases": {
                    "legacy": "model",
                    "pinned": {"model": "model", "quant": "Q4_K_M", "preset": 2},
                },
            }))

            reg = load_registry(path)

            self.assertEqual(reg.aliases["legacy"], AliasTarget(model="model"))
            self.assertEqual(
                reg.aliases["pinned"],
                AliasTarget(model="model", quant="Q4_K_M", preset_slot=2),
            )
            saved = json.loads(path.read_text())
            self.assertEqual(saved["aliases"]["legacy"], {"model": "model"})
            self.assertEqual(
                saved["aliases"]["pinned"],
                {"model": "model", "quant": "Q4_K_M", "preset": 2},
            )
            self.assertEqual(resolve_model_key(saved, "pinned"), "model")


class ModelsMigrationTests(unittest.TestCase):
    def test_load_registry_repairs_filename_fallback_quant_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_path = root / "models.json"
            presets_path = root / "presets.json"
            old_quant = "Qwen_Qwen3.6-27B-Q2_K"
            filename = f"{old_quant}.gguf"
            models_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen36": {
                                "display": "Qwen 3.6",
                                "file": f"bartowski/{filename}",
                                "active_quant": old_quant,
                                "quants": {
                                    old_quant: {
                                        "filename": filename,
                                        "file": f"bartowski/{filename}",
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
                        "_active": {"qwen36": {old_quant: 1}},
                        "qwen36": {
                            old_quant: {
                                "1": {"name": "default", "params": {"ctx": 32768}}
                            }
                        },
                    }
                )
            )

            registry = load_registry(models_path)
            model = registry.models["qwen36"]
            self.assertEqual(model.active_quant, "Q2_K")
            self.assertIn("Q2_K", model.params["quants"])
            self.assertNotIn(old_quant, model.params["quants"])

            presets = load_presets(presets_path)
            self.assertEqual(get_active_slot(presets, "qwen36", "Q2_K"), 1)
            self.assertIsNotNone(get_preset(presets, "qwen36", "Q2_K", 1))

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
        self.assertEqual(data["aliases"]["heavy"], {"model": slug})
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
