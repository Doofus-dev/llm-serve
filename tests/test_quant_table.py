"""Tests for shared Hub/quant-picker file table rows."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tui.data.baselines import RunBaseline, record_baseline
from tui.data.gpu import GPUStats
from tui.data.hf import HubFile
from tui.data.quant_table import (
    build_quant_file_rows,
    fmt_downloaded,
    fmt_downloaded_cell,
    quant_file_row_cells,
)


class QuantTableDownloadTests(unittest.TestCase):
    def test_fmt_downloaded_marks(self) -> None:
        self.assertEqual(fmt_downloaded(True), "●")
        self.assertEqual(fmt_downloaded(False), "—")
        self.assertEqual(fmt_downloaded_cell(True), "[green]●[/]")
        self.assertEqual(fmt_downloaded_cell(False), "—")

    def test_local_file_is_marked_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            author = "bartowski"
            (root / author).mkdir()
            (root / author / "Model-Q4_K_M.gguf").write_bytes(b"gguf")

            rows = build_quant_file_rows(
                [
                    HubFile(path="Model-Q8_0.gguf", size=2000),
                    HubFile(path="Model-Q4_K_M.gguf", size=1000),
                ],
                gpu=GPUStats(),
                context_tokens=65_536,
                offload_ratio=1.0,
                baselines_path=None,
                models_dir=root,
                author=author,
            )
            by_path = {row.path: row for row in rows}

            self.assertTrue(by_path["Model-Q4_K_M.gguf"].downloaded)
            self.assertFalse(by_path["Model-Q8_0.gguf"].downloaded)
            self.assertEqual(
                quant_file_row_cells(by_path["Model-Q4_K_M.gguf"])[0],
                "[green]●[/]",
            )
            self.assertEqual(quant_file_row_cells(by_path["Model-Q8_0.gguf"])[0], "—")

    def test_actual_columns_show_128k_run_at_64k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(
                path,
                RunBaseline(
                    model="qwen35",
                    file="bartowski/Qwen_Qwen3.5-9B-Q5_K_L.gguf",
                    file_size=7_739_240_480,
                    gpu_name="NVIDIA GeForce RTX 5080",
                    ctx=131_072,
                    gpu_layers=99,
                    total_layers=33,
                    cache_k="q4_0",
                    cache_v="q4_0",
                    vram_used_mb=9_631.0,
                    gen_tps=47.8,
                ),
            )
            rows = build_quant_file_rows(
                [HubFile(path="Qwen_Qwen3.5-9B-Q5_K_L.gguf", size=7_739_240_480)],
                gpu=GPUStats(name="NVIDIA GeForce RTX 5080", available=True),
                context_tokens=65_536,
                offload_ratio=1.0,
                baselines_path=path,
            )
            self.assertEqual(len(rows), 1)
            self.assertNotEqual(rows[0].act_vram, "—")
            self.assertNotEqual(rows[0].act_tps, "—")

    def test_actual_columns_use_nearby_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(
                path,
                RunBaseline(
                    model="qwen",
                    file="bartowski/Model-Q4_K_M.gguf",
                    file_size=1000,
                    gpu_name="NVIDIA GeForce RTX 5080",
                    ctx=65_000,
                    gpu_layers=99,
                    total_layers=40,
                    cache_k="q4_0",
                    cache_v="q4_0",
                    vram_used_mb=4_096.0,
                    gen_tps=48.2,
                ),
            )
            rows = build_quant_file_rows(
                [HubFile(path="Model-Q4_K_M.gguf", size=1000)],
                gpu=GPUStats(name="NVIDIA GeForce RTX 5080", available=True),
                context_tokens=65_536,
                offload_ratio=1.0,
                baselines_path=path,
            )
            self.assertEqual(len(rows), 1)
            self.assertNotEqual(rows[0].act_vram, "—")
            self.assertNotEqual(rows[0].act_tps, "—")


if __name__ == "__main__":
    unittest.main()
