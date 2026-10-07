"""Tests for background download progress state."""

from __future__ import annotations

import unittest
from pathlib import Path

from tui.data.downloads import DownloadJob, DownloadManager, DownloadState
from tui.data.hf import DownloadPlan


def _job(filename: str, author: str = "author") -> DownloadJob:
    plan = DownloadPlan(
        repo_id=f"{author}/repo",
        author=author,
        filenames=[filename],
        local_dir=Path("/tmp") / author,
        relative_file=f"{author}/{filename}",
    )
    return DownloadJob(
        plan=plan,
        filename=filename,
        expected_bytes=1_000,
        display="Qwen 3.8",
        model_slug="qwen38",
    )


class DownloadStateTests(unittest.TestCase):
    def test_progress_pct_when_total_known(self) -> None:
        st = DownloadState(running=True, used_bytes=500, expected_bytes=1000)
        self.assertAlmostEqual(st.progress_pct or 0, 50.0)

    def test_progress_pct_none_without_total(self) -> None:
        st = DownloadState(running=True, used_bytes=500, expected_bytes=0)
        self.assertIsNone(st.progress_pct)

    def test_status_line_available_immediately(self) -> None:
        mgr = DownloadManager()
        mgr.enqueue(_job("model.gguf"))
        line = mgr.state.status_line
        self.assertIn("Downloading", line)
        self.assertIn("model.gguf", line)
        self.assertIn("Qwen 3.8", line)

    def test_each_enqueue_starts_immediately(self) -> None:
        mgr = DownloadManager()
        first = _job("a.gguf")
        second = _job("b.gguf")
        self.assertEqual(mgr.enqueue(first), "started")
        self.assertEqual(mgr.enqueue(second), "started")
        self.assertEqual(mgr.active_count, 2)
        self.assertTrue(mgr.busy)
        self.assertTrue(mgr.has_filename("a.gguf"))
        self.assertTrue(mgr.has_filename("b.gguf"))

    def test_duplicate_file_is_rejected(self) -> None:
        mgr = DownloadManager()
        self.assertEqual(mgr.enqueue(_job("a.gguf")), "started")
        self.assertEqual(mgr.enqueue(_job("a.gguf")), "duplicate")
        self.assertEqual(mgr.active_count, 1)

    def test_status_lists_every_active_file(self) -> None:
        mgr = DownloadManager()
        mgr.enqueue(_job("a.gguf"))
        mgr.enqueue(_job("b.gguf"))
        line = mgr.format_status()
        self.assertIn("2 files", line)
        self.assertIn("a.gguf", line)
        self.assertIn("b.gguf", line)

    def test_progress_for_matches_filename(self) -> None:
        mgr = DownloadManager()
        mgr.enqueue(_job("Qwen3.8-27B-Q3_K_S.gguf"))
        st = mgr.state
        self.assertTrue(st.is_transferring("Qwen3.8-27B-Q3_K_S.gguf"))
        self.assertFalse(st.is_transferring("other.gguf"))
        self.assertIsNotNone(st.progress_for("Qwen3.8-27B-Q3_K_S.gguf"))

    def test_unsubscribe_stops_notifications(self) -> None:
        mgr = DownloadManager()
        seen: list[int] = []
        listener = lambda state: seen.append(state.active_count)
        mgr.subscribe(listener)
        mgr.enqueue(_job("a.gguf"))
        mgr.unsubscribe(listener)
        mgr.enqueue(_job("b.gguf"))
        self.assertEqual(seen, [1])


if __name__ == "__main__":
    unittest.main()
