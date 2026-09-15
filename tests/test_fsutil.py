"""Tests for the atomic file-write helper."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tui.data.fsutil import atomic_write


class AtomicWriteTests(unittest.TestCase):
    def test_writes_exact_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cfg.json"
            content = '{"a": 1}\n'
            atomic_write(target, content)
            self.assertEqual(target.read_text(), content)

    def test_no_leftover_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cfg.json"
            atomic_write(target, "{}\n")
            leftovers = [p for p in Path(tmp).iterdir() if p.name != "cfg.json"]
            self.assertEqual(leftovers, [])

    def test_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cfg.json"
            target.write_text('{"old": true}\n')
            atomic_write(target, '{"new": true}\n')
            self.assertEqual(json.loads(target.read_text()), {"new": True})

    def test_creates_parent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "dir" / "cfg.json"
            atomic_write(target, "x\n")
            self.assertEqual(target.read_text(), "x\n")

    def test_target_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "models.json"
            atomic_write(target, json.dumps({"models": {}}) + "\n")
            self.assertIsInstance(json.loads(target.read_text()), dict)


if __name__ == "__main__":
    unittest.main()
