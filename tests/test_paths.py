"""Machine-wide data dir vs per-checkout llama.cpp."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tui.paths import default_paths, migrate_shared_data, shared_data_home


class SharedDataHomeTests(unittest.TestCase):
    def test_llm_serve_data_wins(self) -> None:
        env = {"LLM_SERVE_DATA": "/tmp/explicit-data", "XDG_DATA_HOME": "/tmp/xdg", "HOME": "/tmp/home"}
        self.assertEqual(shared_data_home(env=env), Path("/tmp/explicit-data").resolve())

    def test_xdg_data_home(self) -> None:
        env = {"XDG_DATA_HOME": "/tmp/xdg", "HOME": "/tmp/home"}
        self.assertEqual(shared_data_home(env=env), Path("/tmp/xdg/llm-serve").resolve())

    def test_default_under_home(self) -> None:
        env = {"HOME": "/tmp/home"}
        self.assertEqual(shared_data_home(env=env), Path("/tmp/home/.local/share/llm-serve").resolve())


class DefaultPathsIsolationTests(unittest.TestCase):
    def test_root_keeps_json_inside_the_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = default_paths(root=root, env={"HOME": str(root / "not-used")})
            self.assertEqual(paths.data, root)
            self.assertEqual(paths.models_json, root / "models.json")
            self.assertEqual(paths.models_dir, root / "models")
            self.assertEqual(paths.llama_dir, root / "llama.cpp")


class MigrateSharedDataTests(unittest.TestCase):
    def test_copies_json_and_moves_ggufs_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            data = Path(tmp) / "data"
            models = repo / "models" / "bartowski"
            models.mkdir(parents=True)
            gguf = models / "Qwen.gguf"
            gguf.write_bytes(b"GGUF")
            (repo / "models.json").write_text(json.dumps({"models": {"qwen": {"display": "Qwen"}}, "aliases": {}}) + "\n")
            (repo / "presets.json").write_text("{}\n")
            migrate_shared_data(repo, data, env={})
            self.assertTrue((data / "models.json").is_file())
            self.assertEqual(json.loads((data / "models.json").read_text())["models"]["qwen"]["display"], "Qwen")
            self.assertTrue((data / "presets.json").is_file())
            self.assertTrue((data / "models" / "bartowski" / "Qwen.gguf").is_file())
            self.assertFalse(gguf.exists())
            self.assertTrue((repo / "models" / "README.md").is_file())
            # second run does not overwrite
            (data / "models.json").write_text(json.dumps({"models": {"keep": {}}, "aliases": {}}) + "\n")
            migrate_shared_data(repo, data, env={})
            self.assertIn("keep", json.loads((data / "models.json").read_text())["models"])

    def test_model_dir_env_skips_weight_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            data = Path(tmp) / "data"
            gguf = repo / "models" / "x.gguf"
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b"GGUF")
            migrate_shared_data(repo, data, env={"MODEL_DIR": "/mnt/models"})
            self.assertTrue(gguf.is_file())
            self.assertFalse(_has_gguf(data / "models"))

    def test_production_default_paths_use_data_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            data = Path(tmp) / "share"
            repo.mkdir()
            (repo / "llama.cpp").mkdir()
            env = {"LLM_SERVE_DATA": str(data), "HOME": str(Path(tmp) / "home")}
            # Force code root via LLM_SERVE_HOME so default_paths without root= still
            # would need REPO_ROOT. We pass nothing and patch by calling with env
            # after migrate using default_paths(root=) is isolated; instead call
            # shared paths through migrate + explicit construction via env+monkey.
            from tui import paths as paths_mod

            old = paths_mod.REPO_ROOT
            try:
                paths_mod.REPO_ROOT = repo
                result = paths_mod.default_paths(env=env)
            finally:
                paths_mod.REPO_ROOT = old
            self.assertEqual(result.data, data.resolve())
            self.assertEqual(result.models_json, data.resolve() / "models.json")
            self.assertEqual(result.llama_dir, repo / "llama.cpp")
            self.assertTrue((data / "models.json").is_file())


def _has_gguf(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.gguf"))


if __name__ == "__main__":
    unittest.main()
