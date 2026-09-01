"""Tests for background download progress state."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.downloads import DownloadState
from tui.data.hf import DownloadPlan


class DownloadStateTests(unittest.TestCase):
    def test_progress_pct_when_total_known(self) -> None:
        st = DownloadState(running=True, used_bytes=500, expected_bytes=1000)
        self.assertAlmostEqual(st.progress_pct or 0, 50.0)

    def test_progress_pct_none_without_total(self) -> None:
        st = DownloadState(running=True, used_bytes=500, expected_bytes=0)
        self.assertIsNone(st.progress_pct)

    def test_status_line_available_immediately(self) -> None:
        from tui.data.downloads import DownloadManager

        mgr = DownloadManager()
        plan = DownloadPlan(
            repo_id="author/repo",
            author="author",
            filenames=["model.gguf"],
            local_dir=Path("/tmp"),
            relative_file="author/model.gguf",
        )
        mgr.state = DownloadState(
            running=True,
            filename="model.gguf",
            expected_bytes=1_000_000,
        )
        mgr.state.status_line = mgr.format_status()
        self.assertIn("Downloading model.gguf", mgr.state.status_line)
        self.assertIn("0%", mgr.state.status_line)


if __name__ == "__main__":
    unittest.main()
