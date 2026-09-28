"""Tests for GPU totals and per-process VRAM/RAM parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tui.data.gpu import (
    collect_process_stats,
    kfd_client_pids,
    parse_drm_fdinfo,
    parse_nvidia_compute_apps,
    parse_pidof,
)


class GpuProcessParseTests(unittest.TestCase):
    def test_parse_nvidia_compute_apps_rows(self) -> None:
        text = (
            "1234, /usr/bin/llama-server, 8192\n"
            "5678, Xorg, 256\n"
            "No running processes found\n"
        )
        procs = parse_nvidia_compute_apps(text)
        self.assertEqual(len(procs), 2)
        self.assertEqual(procs[0].pid, 1234)
        self.assertEqual(procs[0].name, "llama-server")
        self.assertEqual(procs[0].vram_mb, 8192.0)
        self.assertEqual(procs[1].pid, 5678)
        self.assertEqual(procs[1].name, "Xorg")
        self.assertEqual(procs[1].vram_mb, 256.0)

    def test_parse_nvidia_compute_apps_skips_na(self) -> None:
        procs = parse_nvidia_compute_apps("42, helper, [N/A]\n")
        self.assertEqual(len(procs), 1)
        self.assertEqual(procs[0].vram_mb, 0.0)

    def test_parse_drm_fdinfo_vram_kib(self) -> None:
        text = (
            "pos:\t0\n"
            "drm-driver:       amdgpu\n"
            "drm-memory-vram:  2048 KiB\n"
            "drm-memory-gtt:   512 KiB\n"
        )
        self.assertAlmostEqual(parse_drm_fdinfo(text), 2.0)

    def test_parse_drm_fdinfo_falls_back_to_gtt(self) -> None:
        text = "drm-driver: amdgpu\ndrm-memory-gtt: 1024 KiB\n"
        self.assertAlmostEqual(parse_drm_fdinfo(text), 1.0)

    def test_parse_pidof_multiple_llama_servers(self) -> None:
        self.assertEqual(parse_pidof("1001 2001\n"), [1001, 2001])

    def test_kfd_client_pids_lists_compute_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1001").mkdir()
            (root / "2001").mkdir()
            (root / "card0").mkdir()
            self.assertEqual(set(kfd_client_pids(root)), {1001, 2001})


class GpuCollectProcessStatsTests(unittest.TestCase):
    def _collect_amd(self, *, kfd: list[int], llama: list[int], tracked: list[int]):
        def comm(pid: int) -> str:
            return "llama-server"

        def drm(pid: int) -> float:
            return 8192.0 if pid == 1001 else 4096.0

        def rss(pid: int) -> float | None:
            return 256.0

        with (
            patch("tui.data.gpu._query_nvidia_compute_apps", return_value=[]),
            patch("tui.data.gpu.kfd_client_pids", return_value=kfd),
            patch("tui.data.gpu._query_llama_server_pids", return_value=llama),
            patch("tui.data.gpu._comm", side_effect=comm),
            patch("tui.data.gpu._drm_vram_mb", side_effect=drm),
            patch("tui.data.gpu.process_rss_mb", side_effect=rss),
            patch("tui.data.gpu._child_pids", return_value=set()),
        ):
            return collect_process_stats(tracked)

    def test_collect_process_stats_amd_kfd_lists_second_server(self) -> None:
        procs = self._collect_amd(kfd=[1001, 2001], llama=[], tracked=[1001])
        self.assertEqual([proc.pid for proc in procs], [1001, 2001])
        self.assertTrue(procs[0].tracked)
        self.assertFalse(procs[1].tracked)
        self.assertEqual(procs[0].vram_mb, 8192.0)
        self.assertEqual(procs[1].vram_mb, 4096.0)
        self.assertEqual(procs[1].name, "llama-server")

    def test_collect_process_stats_amd_pidof_lists_leftover_server(self) -> None:
        procs = self._collect_amd(kfd=[], llama=[2001], tracked=[1001])
        self.assertEqual([proc.pid for proc in procs], [1001, 2001])
        self.assertTrue(procs[0].tracked)
        self.assertFalse(procs[1].tracked)
        self.assertEqual(procs[1].vram_mb, 4096.0)


if __name__ == "__main__":
    unittest.main()
