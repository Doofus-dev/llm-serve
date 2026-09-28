"""Tests for GPU totals and per-process VRAM/RAM parsing."""

from __future__ import annotations

import unittest

from tui.data.gpu import parse_drm_fdinfo, parse_nvidia_compute_apps


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


if __name__ == "__main__":
    unittest.main()
