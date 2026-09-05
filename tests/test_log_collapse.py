"""Tests for consecutive llama.cpp log collapsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tui.data.log_collapse import (
    LogCollapser,
    LogTailer,
    classify_family,
    slice_to_session,
)

IDLE = "0.01.469.096 I srv  update_slots: all slots are idle"
IDLE_2 = "0.01.654.189 I srv  update_slots: all slots are idle"
IDLE_3 = "0.02.153.810 I srv  update_slots: all slots are idle"
REQUEST = (
    "0.42.109.598 I slot get_availabl: id  0 | task -1 | "
    "selected slot by LRU, t_last = -1"
)
CACHE_1 = (
    "0.42.109.971 I slot   operator(): id  0 | task 162 | "
    "cached n_tokens = 0, memory_seq_rm [0, end)"
)
CACHE_2 = (
    "0.43.376.384 I slot   operator(): id  0 | task 162 | "
    "cached n_tokens = 2048, memory_seq_rm [2048, end)"
)
CACHE_3 = (
    "0.44.688.215 I slot   operator(): id  0 | task 162 | "
    "cached n_tokens = 4096, memory_seq_rm [4096, end)"
)
PRINT_1 = "0.00.339.417 I print_info: file type   = Q2_K - Medium"
PRINT_2 = "0.00.339.418 I print_info: file size   = 11.21 GiB (3.53 BPW)"
PRINT_3 = "0.00.456.228 I print_info: model params          = 27.32 B"
SAMPLER = "0.42.109.962 I slot launch_slot_: id  0 | task -1 | sampler params: "
CONT_1 = "\trepeat_last_n = 64, repeat_penalty = 1.000"
CONT_2 = "You are a helpful assistant<|im_end|>"
LAUNCH = "── 2026-09-04 17:31:50 launch: qwen36-27b-bartowski (PID 220184) ──"


def rows_from(text: str):
    collapser = LogCollapser()
    collapser.feed_text(text)
    return collapser.rows


class ClassifyTests(unittest.TestCase):
    def test_idle_family(self) -> None:
        self.assertEqual(classify_family(IDLE), "idle")

    def test_continuation_has_no_family(self) -> None:
        self.assertIsNone(classify_family(CONT_1))
        self.assertIsNone(classify_family(CONT_2))

    def test_launch_family(self) -> None:
        self.assertEqual(classify_family(LAUNCH), "launch")


class CollapseTests(unittest.TestCase):
    def test_idle_run_collapses_to_one_row(self) -> None:
        rows = rows_from("\n".join((IDLE, IDLE_2, IDLE_3)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].family, "idle")
        self.assertEqual(rows[0].count, 3)
        self.assertEqual(rows[0].text, IDLE_3)
        self.assertIn("×3", rows[0].render())

    def test_idle_request_idle_are_two_idle_rows(self) -> None:
        rows = rows_from("\n".join((IDLE, IDLE_2, REQUEST, IDLE_3)))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].family, "idle")
        self.assertEqual(rows[0].count, 2)
        self.assertEqual(rows[0].text, IDLE_2)
        self.assertNotEqual(rows[1].family, "idle")
        self.assertIn("selected slot by LRU", rows[1].text)
        self.assertEqual(rows[2].family, "idle")
        self.assertEqual(rows[2].count, 1)
        self.assertEqual(rows[2].text, IDLE_3)

    def test_prefill_ticks_keep_latest_n_tokens(self) -> None:
        rows = rows_from("\n".join((CACHE_1, CACHE_2, CACHE_3)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].family, "cached_tokens")
        self.assertEqual(rows[0].count, 3)
        self.assertIn("cached n_tokens = 4096", rows[0].text)

    def test_print_info_fields_stay_separate(self) -> None:
        rows = rows_from("\n".join((PRINT_1, PRINT_2, PRINT_3)))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row.count == 1 for row in rows))
        self.assertIn("file type", rows[0].text)
        self.assertIn("file size", rows[1].text)
        self.assertIn("model params", rows[2].text)

    def test_continuation_lines_stay_visible(self) -> None:
        rows = rows_from("\n".join((SAMPLER, CONT_1, CONT_2, REQUEST)))
        self.assertEqual(len(rows), 4)
        self.assertIn("sampler params", rows[0].text)
        self.assertIn("repeat_last_n", rows[1].text)
        self.assertIn("helpful assistant", rows[2].text)
        self.assertIn("selected slot by LRU", rows[3].text)

    def test_single_line_has_no_count_suffix(self) -> None:
        rows = rows_from(REQUEST)
        self.assertEqual(rows[0].count, 1)
        self.assertNotIn("×", rows[0].render())

    def test_kv_dump_fields_stay_separate(self) -> None:
        rows = rows_from(
            "\n".join(
                (
                    "0.00.339.609 I llama_model_loader: loaded meta data with 46 key-value pairs",
                    "0.00.339.624 I llama_model_loader: - kv   0: general.architecture str = qwen35",
                    "0.00.339.624 I llama_model_loader: - kv   1: general.type str = model",
                    "0.00.364.372 I llama_model_loader: - type  f32:  456 tensors",
                )
            )
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.count == 1 for row in rows))

    def test_distinct_unused_tensors_stay_separate(self) -> None:
        rows = rows_from(
            "\n".join(
                (
                    "0.00.457.459 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring",
                    "0.00.457.461 W model has unused tensor blk.64.post_attention_norm.weight (size = 20480 bytes) -- ignoring",
                )
            )
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.count == 1 for row in rows))

    def test_cors_separators_use_normalized_fallback(self) -> None:
        rows = rows_from(
            "\n".join(
                (
                    "0.00.077.046 W srv  llama_server: -----------------",
                    "0.00.077.047 W srv  llama_server: CORS is set to allow all origins ('*') and no API key is set",
                    "0.00.077.047 W srv  llama_server: -----------------",
                )
            )
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].count, 1)
        self.assertEqual(rows[2].family, rows[0].family)


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


class TailerTests(unittest.TestCase):
    def test_incremental_idle_updates_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-serve.log"
            path.write_text(f"{LAUNCH}\n{IDLE}\n")
            tailer = LogTailer()
            self.assertTrue(tailer.poll(path))
            self.assertEqual(tailer.rows[-1].count, 1)

            with path.open("a") as handle:
                handle.write(f"{IDLE_2}\n")
            self.assertTrue(tailer.poll(path))
            idle_rows = [row for row in tailer.rows if row.family == "idle"]
            self.assertEqual(len(idle_rows), 1)
            self.assertEqual(idle_rows[0].count, 2)

    def test_truncation_resets_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-serve.log"
            path.write_text(f"{LAUNCH}\n{IDLE}\n")
            tailer = LogTailer()
            tailer.poll(path)
            path.write_text("── 2026-09-04 18:00:00 launch: other-model (PID 99) ──\n")
            self.assertTrue(tailer.poll(path))
            self.assertTrue(any("other-model" in row.text for row in tailer.rows))
            self.assertFalse(any("qwen36-27b-bartowski" in row.text for row in tailer.rows))


if __name__ == "__main__":
    unittest.main()
