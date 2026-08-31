"""Tests for Hugging Face integration helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.hf import (
    build_download_plan,
    build_source_metadata,
    shard_filenames,
)
from tui.data.models_json import create_downloaded_model, load_registry, merge_editor_params, update_model
from tui.data.settings import TUISettings, load_settings, remember_hf_author, save_settings


class HFPathTests(unittest.TestCase):
    def test_build_download_plan_flat_filename(self) -> None:
        models_dir = Path("/repo/models")
        plan = build_download_plan("bartowski/Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q3_K_S.gguf", models_dir)
        self.assertEqual(plan.author, "bartowski")
        self.assertEqual(plan.local_dir, models_dir / "bartowski")
        self.assertEqual(plan.relative_file, "bartowski/Qwen3.6-27B-Q3_K_S.gguf")
        self.assertEqual(plan.filenames, ["Qwen3.6-27B-Q3_K_S.gguf"])

    def test_shard_filenames(self) -> None:
        all_files = [
            "Model-00001-of-00003.gguf",
            "Model-00002-of-00003.gguf",
            "Model-00003-of-00003.gguf",
            "Model-Q4_K_M.gguf",
        ]
        shards = shard_filenames("Model-00001-of-00003.gguf", all_files)
        self.assertEqual(
            shards,
            ["Model-00001-of-00003.gguf", "Model-00002-of-00003.gguf", "Model-00003-of-00003.gguf"],
        )
        self.assertEqual(shard_filenames("Model-Q4_K_M.gguf", all_files), ["Model-Q4_K_M.gguf"])

    def test_build_source_metadata(self) -> None:
        models_dir = Path("/repo/models")
        plan = build_download_plan("unsloth/Qwen3.6-27B-GGUF", "Qwen3.6-27B-UD-Q3_K_XL.gguf", models_dir)
        source = build_source_metadata(plan, "Qwen3.6-27B-UD-Q3_K_XL.gguf")
        self.assertEqual(source["author"], "unsloth")
        self.assertEqual(source["repo"], "unsloth/Qwen3.6-27B-GGUF")


class SettingsTests(unittest.TestCase):
    def test_hf_authors_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tui-settings.json"
            settings = TUISettings(theme="gruvbox", hf_authors=["bartowski"])
            save_settings(path, settings)
            loaded = load_settings(path)
            self.assertEqual(loaded.theme, "gruvbox")
            self.assertEqual(loaded.hf_authors, ["bartowski"])

            self.assertTrue(remember_hf_author(loaded, "unsloth"))
            self.assertFalse(remember_hf_author(loaded, "unsloth"))
            self.assertEqual(loaded.hf_authors, ["bartowski", "unsloth"])


class ModelJsonTests(unittest.TestCase):
    def test_merge_editor_params_preserves_source(self) -> None:
        existing = {
            "file": "bartowski/model.gguf",
            "ctx": 65536,
            "source": {"hub": "huggingface", "author": "bartowski"},
        }
        edited = {"file": "bartowski/model.gguf", "ctx": 32768, "port": 8081}
        merged = merge_editor_params(existing, edited, {"file", "ctx", "port"})
        self.assertEqual(merged["ctx"], 32768)
        self.assertEqual(merged["source"], existing["source"])

    def test_create_downloaded_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(json.dumps({
                "models": {
                    "base": {"file": "old.gguf", "ctx": 4096, "port": 8081},
                },
                "aliases": {},
            }))
            create_downloaded_model(
                path,
                "qwen36-bart",
                {"file": "old.gguf", "ctx": 4096, "port": 8081},
                file_rel="bartowski/Qwen3.6-27B-Q3_K_S.gguf",
                source={
                    "hub": "huggingface",
                    "repo": "bartowski/Qwen3.6-27B-GGUF",
                    "filename": "Qwen3.6-27B-Q3_K_S.gguf",
                    "author": "bartowski",
                    "revision": "main",
                },
            )
            reg = load_registry(path)
            self.assertIn("qwen36-bart", reg.models)
            params = reg.models["qwen36-bart"].params
            self.assertEqual(params["file"], "bartowski/Qwen3.6-27B-Q3_K_S.gguf")
            self.assertEqual(params["source"]["author"], "bartowski")

            update_model(path, "qwen36-bart", merge_editor_params(params, {"ctx": 8192}, {"ctx"}))
            reg2 = load_registry(path)
            self.assertEqual(reg2.models["qwen36-bart"].params["source"]["author"], "bartowski")


class HFCliTests(unittest.TestCase):
    @patch("tui.data.hf._run_hf")
    def test_list_gguf_repos(self, mock_run) -> None:
        from tui.data.hf import list_gguf_repos

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps([
            {"id": "bartowski/Qwen3.6-27B-GGUF", "downloads": 1, "likes": 2, "trending_score": 3},
        ])
        with patch("tui.data.hf.hf_available", return_value=True):
            repos, error = list_gguf_repos(author="bartowski")
        self.assertIsNone(error)
        self.assertEqual(len(repos), 1)
        self.assertEqual(repos[0].author, "bartowski")


if __name__ == "__main__":
    unittest.main()
