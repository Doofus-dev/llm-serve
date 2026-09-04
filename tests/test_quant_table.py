"""Tests for shared Hub/quant-picker file table rows."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
