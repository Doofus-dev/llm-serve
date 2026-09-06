"""Tests for llama.cpp log line family classification."""

from __future__ import annotations

import unittest

from tui.data.log_collapse import classify_family, slice_to_session

IDLE = "0.01.469.096 I srv  update_slots: all slots are idle"
REQUEST = (
    "0.42.109.598 I slot get_availabl: id  0 | task -1 | "
    "selected slot by LRU, t_last = -1"
)
CONT_1 = "\trepeat_last_n = 64, repeat_penalty = 1.000"
CONT_2 = "You are a helpful assistant<|im_end|>"
LAUNCH = "── 2026-09-04 17:31:50 launch: qwen36-27b-bartowski (PID 220184) ──"


class ClassifyTests(unittest.TestCase):
    def test_idle_family(self) -> None:
        self.assertEqual(classify_family(IDLE), "idle")

    def test_continuation_has_no_family(self) -> None:
        self.assertIsNone(classify_family(CONT_1))
        self.assertIsNone(classify_family(CONT_2))

    def test_launch_family(self) -> None:
        self.assertEqual(classify_family(LAUNCH), "launch")


class SessionSliceTests(unittest.TestCase):
    def test_keeps_from_last_launch(self) -> None:
        earlier = (
            "── 2026-09-04 13:40:10 launch: older-model (PID 1) ──\n"
            f"{IDLE}\n"
        )
        later = f"{LAUNCH}\n{REQUEST}\n"
        sliced = slice_to_session(earlier + later)
        self.assertIn("qwen36-27b-bartowski", sliced)
        self.assertNotIn("older-model", sliced)


if __name__ == "__main__":
    unittest.main()
