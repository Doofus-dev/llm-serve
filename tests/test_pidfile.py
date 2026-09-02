"""Tests for backward-compatible server PID metadata."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tui.data.pidfile import read_pid_file


class PidFileTests(unittest.TestCase):
    def _read(self, contents: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.pid"
            path.write_text(contents)
            return read_pid_file(path)

    def test_reads_running_preset_metadata(self) -> None:
        info = self._read("123 qwen36-27b-bartowski 8080 1000 Q2_K 1 1\n")

        self.assertIsNotNone(info)
        self.assertEqual(info.quant, "Q2_K")
        self.assertEqual(info.preset_slot, 1)
        self.assertTrue(info.remote)

    def test_old_pid_format_remains_supported(self) -> None:
        info = self._read("123 qwen36-27b-bartowski 8080 1000\n")

        self.assertIsNotNone(info)
        self.assertIsNone(info.quant)
        self.assertIsNone(info.preset_slot)
        self.assertFalse(info.remote)


if __name__ == "__main__":
    unittest.main()
