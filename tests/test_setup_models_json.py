"""First setup seeds an empty models.json; tests never read the repo-root file."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tui.data.models_json import load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
EMPTY_BODY = '{"models": {}, "aliases": {}}'
EMPTY_REGISTRY = {"models": {}, "aliases": {}}


def _run_setup_phase5(script_dir: Path) -> subprocess.CompletedProcess[str]:
    """Execute setup.sh phase 5 against an isolated SCRIPT_DIR."""
    setup = (REPO_ROOT / "setup.sh").read_text()
    start = setup.index("# Phase 5:")
    end = setup.index("# Make launcher executable")
    phase5 = setup[start:end]
    script = f"""
set -euo pipefail
SCRIPT_DIR={json.dumps(str(script_dir))}
ok() {{ echo "[ok] $*"; }}
warn() {{ echo "[warn] $*"; }}
{phase5}
"""
    return subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


class GitignoreModelsJsonTests(unittest.TestCase):
    def test_gitignore_lists_models_json_next_to_presets(self) -> None:
        lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("models.json", lines)
        self.assertIn("presets.json", lines)
        self.assertEqual(abs(lines.index("models.json") - lines.index("presets.json")), 1)


class SetupSeedsEmptyRegistryTests(unittest.TestCase):
    def test_setup_sh_does_not_copy_example(self) -> None:
        src = (REPO_ROOT / "setup.sh").read_text()
        self.assertIn(EMPTY_BODY, src)
        self.assertNotIn("models.json.example", src)
        self.assertNotIn("cp models.json.example", src)

    def test_missing_file_gets_empty_registry_not_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            example = root / "models.json.example"
            example.write_text(json.dumps({
                "models": {"qwen-should-not-be-copied": {"display": "Qwen"}},
                "aliases": {"default": {"model": "qwen-should-not-be-copied"}},
            }) + "\n")
            result = _run_setup_phase5(root)
            seeded = root / "models.json"
            self.assertTrue(seeded.is_file(), result.stdout)
            self.assertEqual(json.loads(seeded.read_text()), EMPTY_REGISTRY)
            self.assertEqual(seeded.read_text().strip(), EMPTY_BODY)
            self.assertNotIn("qwen", seeded.read_text())
            self.assertIn("qwen-should-not-be-copied", example.read_text())

    def test_existing_file_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = {"models": {"keep-me": {"display": "Keep"}}, "aliases": {}}
            (root / "models.json").write_text(json.dumps(existing) + "\n")
            _run_setup_phase5(root)
            self.assertEqual(json.loads((root / "models.json").read_text()), existing)


class EmptyRegistryLoadTests(unittest.TestCase):
    def test_load_registry_empty_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(EMPTY_BODY + "\n")
            reg = load_registry(path, migrate=False)
            self.assertEqual(reg.models, {})
            self.assertEqual(reg.aliases, {})

    def test_load_registry_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            self.assertFalse(path.exists())
            reg = load_registry(path, migrate=False)
            self.assertEqual(reg.models, {})
            self.assertEqual(reg.aliases, {})
