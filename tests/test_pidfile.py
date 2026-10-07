"""Tests for backward-compatible server PID metadata."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tui.data.pidfile import (
    forget_instance,
    instance_file,
    list_instances,
    read_pid_file,
    record_instance,
    remap_all_preset_slots,
    remap_pid_preset_slots,
    write_pid_file,
)


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


class MultiInstancePidTests(unittest.TestCase):
    def test_record_instance_fills_vacant_tracked_pidfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pid_file = log_dir / "server.pid"
            record_instance(
                pid_file=pid_file,
                log_dir=log_dir,
                pid=os.getpid(),
                model="demo-model",
                port=8081,
                quant="Q4_K_M",
                preset_slot=1,
                remote=False,
                started_at=1000,
            )
            tracked = read_pid_file(pid_file)
            extra = read_pid_file(instance_file(log_dir, os.getpid()))
            self.assertIsNotNone(tracked)
            self.assertEqual(tracked.pid, os.getpid())
            self.assertIsNotNone(extra)
            self.assertEqual(extra.port, 8081)

    def test_record_instance_does_not_overwrite_live_tracked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pid_file = log_dir / "server.pid"
            write_pid_file(
                pid_file,
                pid=os.getpid(),
                model="first",
                port=8081,
                quant="Q4_K_M",
                preset_slot=1,
                remote=False,
                started_at=1,
            )
            record_instance(
                pid_file=pid_file,
                log_dir=log_dir,
                pid=os.getpid(),
                model="second",
                port=8082,
                quant="Q4_K_M",
                preset_slot=1,
                remote=False,
                started_at=2,
            )
            tracked = read_pid_file(pid_file)
            self.assertEqual(tracked.model, "first")
            self.assertEqual(tracked.port, 8081)

    def test_list_instances_unions_pidfile_and_dir_and_prunes_dead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pid_file = log_dir / "server.pid"
            write_pid_file(
                pid_file,
                pid=os.getpid(),
                model="live",
                port=8081,
                quant="Q4_K_M",
                preset_slot=1,
                remote=False,
                started_at=1,
            )
            write_pid_file(
                instance_file(log_dir, 999_999_999),
                pid=999_999_999,
                model="dead",
                port=8082,
                quant="Q4_K_M",
                preset_slot=1,
                remote=False,
                started_at=2,
            )
            found = list_instances(pid_file=pid_file, log_dir=log_dir)
            self.assertEqual([info.model for info in found], ["live"])
            self.assertFalse(instance_file(log_dir, 999_999_999).exists())

    def test_forget_instance_clears_tracked_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pid_file = log_dir / "server.pid"
            record_instance(
                pid_file=pid_file,
                log_dir=log_dir,
                pid=os.getpid(),
                model="demo-model",
                port=8081,
                quant="Q4_K_M",
                preset_slot=1,
                remote=False,
            )
            forget_instance(pid_file=pid_file, log_dir=log_dir, pid=os.getpid())
            self.assertFalse(pid_file.exists())
            self.assertFalse(instance_file(log_dir, os.getpid()).exists())

    def test_remap_all_preset_slots_updates_every_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            pid_file = log_dir / "server.pid"
            write_pid_file(
                pid_file,
                pid=os.getpid(),
                model="demo-model",
                port=8081,
                quant="Q2_K",
                preset_slot=5,
                remote=False,
                started_at=1,
            )
            record_instance(
                pid_file=pid_file,
                log_dir=log_dir,
                pid=os.getpid(),
                model="demo-model",
                port=8082,
                quant="Q2_K",
                preset_slot=5,
                remote=False,
                started_at=2,
            )
            changed = remap_all_preset_slots(
                pid_file, log_dir, {("demo-model", "Q2_K"): {5: 3}}
            )
            self.assertTrue(changed)
            self.assertEqual(read_pid_file(pid_file).preset_slot, 3)
            self.assertEqual(
                read_pid_file(instance_file(log_dir, os.getpid())).preset_slot, 3
            )


if __name__ == "__main__":
    unittest.main()
