"""Tests for backward-compatible server PID metadata."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tui.data.pidfile import read_pid_file, remap_pid_preset_slots, write_pid_file


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

    def test_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.pid"
            write_pid_file(
                path,
                pid=99,
                model="demo-model",
                port=8081,
                quant="Q4_K_M",
                preset_slot=2,
                remote=True,
                started_at=1000,
            )
            info = read_pid_file(path)
            self.assertIsNotNone(info)
            self.assertEqual(info.pid, 99)
            self.assertEqual(info.model, "demo-model")
            self.assertEqual(info.port, 8081)
            self.assertEqual(info.ts, "1000")
            self.assertEqual(info.quant, "Q4_K_M")
            self.assertEqual(info.preset_slot, 2)
            self.assertTrue(info.remote)

    def test_remap_updates_running_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.pid"
            write_pid_file(
                path,
                pid=os.getpid(),
                model="demo-model",
                port=8081,
                quant="Q2_K",
                preset_slot=5,
                remote=False,
                started_at=1000,
            )
            changed = remap_pid_preset_slots(
                path, {("demo-model", "Q2_K"): {1: 1, 3: 2, 5: 3}}
            )
            self.assertTrue(changed)
            info = read_pid_file(path)
            self.assertEqual(info.preset_slot, 3)
            self.assertEqual(info.ts, "1000")


if __name__ == "__main__":
    unittest.main()
