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
    return DownloadJob(plan=plan, filename=filename, expected_bytes=1_000)


class DownloadStateTests(unittest.TestCase):
    def test_progress_pct_when_total_known(self) -> None:
        st = DownloadState(running=True, used_bytes=500, expected_bytes=1000)
        self.assertAlmostEqual(st.progress_pct or 0, 50.0)

    def test_progress_pct_none_without_total(self) -> None:
        st = DownloadState(running=True, used_bytes=500, expected_bytes=0)
        self.assertIsNone(st.progress_pct)

    def test_status_line_available_immediately(self) -> None:
        mgr = DownloadManager()
        mgr.state = DownloadState(
            running=True,
            filename="model.gguf",
            expected_bytes=1_000_000,
        )
        mgr.state.status_line = mgr.format_status()
        self.assertIn("Downloading model.gguf", mgr.state.status_line)
        self.assertIn("0%", mgr.state.status_line)

    def test_first_enqueue_starts_and_second_queues(self) -> None:
        mgr = DownloadManager()
        first = _job("a.gguf")
        second = _job("b.gguf")
        self.assertEqual(mgr.enqueue(first), "started")
        self.assertEqual(mgr.enqueue(second), "queued")
        self.assertEqual(mgr.queue_size, 2)
        self.assertTrue(mgr.busy)
        popped = mgr.pop_next()
        self.assertIs(popped, first)
        self.assertEqual(mgr.queue_size, 1)
        self.assertEqual(mgr.state.queued, ("b.gguf",))

    def test_duplicate_file_is_rejected(self) -> None:
        mgr = DownloadManager()
        self.assertEqual(mgr.enqueue(_job("a.gguf")), "started")
        self.assertEqual(mgr.enqueue(_job("a.gguf")), "duplicate")
        self.assertEqual(mgr.queue_size, 1)

    def test_status_mentions_queued_files(self) -> None:
        mgr = DownloadManager()
        mgr.enqueue(_job("a.gguf"))
        mgr.pop_next()
        mgr.state.running = True
        mgr.state.filename = "a.gguf"
        mgr.state.expected_bytes = 1_000_000
        mgr.enqueue(_job("b.gguf"))
        mgr.enqueue(_job("c.gguf"))
        line = mgr.format_status()
        self.assertIn("Downloading a.gguf", line)
        self.assertIn("2 queued", line)


if __name__ == "__main__":
    unittest.main()

