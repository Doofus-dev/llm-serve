"""Tests for throughput history, live rate computation, and sparkline rendering."""

from __future__ import annotations

import io
import unittest

from rich.console import Console

from tui.data.stats import Metrics
from tui.data.throughput_history import (
    LastRequest,
    LiveThroughput,
    SPARKLINE_WIDTH,
    ThroughputHistory,
    ThroughputReader,
    compute_live_tps,
    format_avg_line,
    format_last_request,
    live_tps_from_metrics,
    render_prefill_progress,
    render_stage_strip,
    render_tps_sparkline,
    sample_tps_for_history,
    summarize_slots,
)


def render_text(renderable) -> str:
    output = io.StringIO()
    Console(file=output, width=100, color_system=None).print(renderable)
    return output.getvalue()


class ThroughputHistoryTests(unittest.TestCase):
    def test_metrics_delta_preferred_when_processing(self) -> None:
        metrics = Metrics(
            requests_processing=1,
            predicted_tokens_seconds=10.0,
            gen_tps_derived=48.0,
        )
        live = live_tps_from_metrics(metrics)
        self.assertAlmostEqual(live.gen_tps, 48.0)
        self.assertEqual(live.source, "metrics_delta")

    def test_metrics_gauge_used_when_idle(self) -> None:
        metrics = Metrics(
            requests_processing=0,
            predicted_tokens_seconds=42.0,
            gen_tps_derived=0.0,
        )
        live = live_tps_from_metrics(metrics)
        self.assertAlmostEqual(live.gen_tps, 42.0)
        self.assertEqual(live.source, "metrics_gauge")

    def test_slot_n_decoded_delta_produces_tps(self) -> None:
        metrics = Metrics(requests_processing=1)
        slots_before = [
            {
                "is_processing": True,
                "n_prompt_tokens_processed": 10,
                "next_token": [{"n_decoded": 5}],
            }
        ]
        slots_after = [
            {
                "is_processing": True,
                "n_prompt_tokens_processed": 10,
                "next_token": [{"n_decoded": 15}],
            }
        ]

        _, snap, t0 = compute_live_tps(metrics, slots_before, now=0.0)
        live, _, _ = compute_live_tps(
            metrics,
            slots_after,
            last_slot_snap=snap,
            last_slot_t=t0,
            now=0.5,
        )

        self.assertAlmostEqual(live.gen_tps, 20.0)
        self.assertEqual(live.source, "slots")
        self.assertEqual(live.stage, "generating")

    def test_slot_prompt_delta_sets_prefill_phase(self) -> None:
        metrics = Metrics(requests_processing=1)
        slots_before = [
            {
                "is_processing": True,
                "n_prompt_tokens_processed": 0,
                "next_token": [{"n_decoded": 0}],
            }
        ]
        slots_after = [
            {
                "is_processing": True,
                "n_prompt_tokens_processed": 50,
                "next_token": [{"n_decoded": 0}],
            }
        ]

        _, snap, t0 = compute_live_tps(metrics, slots_before, now=1.0)
        live, _, _ = compute_live_tps(
            metrics,
            slots_after,
            last_slot_snap=snap,
            last_slot_t=t0,
            now=1.5,
        )

        self.assertAlmostEqual(live.prompt_tps, 100.0)
        self.assertEqual(live.stage, "prefill")

    def test_direct_tg_tps_from_slots(self) -> None:
        metrics = Metrics(requests_processing=1)
        slots = [{"is_processing": True, "tg_tps": 51.2, "pp_tps": 120.0}]
        live, _, _ = compute_live_tps(metrics, slots)
        self.assertAlmostEqual(live.gen_tps, 51.2)
        self.assertAlmostEqual(live.prompt_tps, 120.0)
        self.assertEqual(live.source, "slots")

    def test_slot_prefill_detected_without_rate_delta(self) -> None:
        metrics = Metrics(requests_processing=1)
        slots = [
            {
                "is_processing": True,
                "n_prompt_tokens": 1000,
                "n_prompt_tokens_processed": 200,
                "next_token": [{"n_decoded": 0}],
            }
        ]
        live, _, _ = compute_live_tps(metrics, slots)
        self.assertEqual(live.stage, "prefill")
        self.assertEqual(live.gen_tps, 0.0)
        self.assertEqual(live.n_prompt_processed, 200)
        self.assertEqual(live.n_prompt_total, 1000)

    def test_metrics_gauge_not_used_while_processing(self) -> None:
        metrics = Metrics(
            requests_processing=1,
            predicted_tokens_seconds=293.9,
            gen_tps_derived=0.0,
        )
        live = live_tps_from_metrics(metrics)
        self.assertEqual(live.gen_tps, 0.0)
        self.assertEqual(live.source, "idle")

    def test_sample_tps_ignores_stale_gauge(self) -> None:
        live = LiveThroughput(
            gen_tps=293.9,
            prompt_tps=0.0,
            stage="generating",
            source="metrics_gauge",
        )
        self.assertEqual(sample_tps_for_history(live), 0.0)

    def test_sample_tps_ignores_prefill_rate(self) -> None:
        live = LiveThroughput(
            gen_tps=0.0,
            prompt_tps=3800.0,
            stage="prefill",
            source="slots",
        )
        self.assertEqual(sample_tps_for_history(live), 0.0)

    def test_sample_tps_uses_gen_rate_during_generation(self) -> None:
        live = LiveThroughput(gen_tps=42.0, prompt_tps=0.0, stage="generating", source="slots")
        self.assertAlmostEqual(sample_tps_for_history(live), 42.0)

    def test_phase_hold_prevents_idle_flicker(self) -> None:
        reader = ThroughputReader()
        reader.PHASE_HOLD_S = 2.0
        metrics_busy = Metrics(requests_processing=1)
        metrics_idle = Metrics(requests_processing=0)
        slots_prefill = [
            {
                "is_processing": True,
                "n_prompt_tokens": 500,
                "n_prompt_tokens_processed": 50,
                "next_token": [{"n_decoded": 0}],
            }
        ]

        live_busy = reader.update(metrics_busy, slots_prefill)
        self.assertEqual(live_busy.stage, "prefill")

        live_idle = reader.update(metrics_idle, [])
        self.assertEqual(live_idle.stage, "prefill")

    def test_format_avg_line_ignores_idle_samples(self) -> None:
        self.assertIsNone(format_avg_line([0.0, 0.0], 0.5))
        line = format_avg_line([0.0, 20.0, 0.0, 40.0], 0.5)
        assert line is not None
        self.assertIn("avg 15.0 tok/s", line.plain)
        self.assertIn("2s rolling", line.plain)

    def test_history_average_includes_zeros(self) -> None:
        history = ThroughputHistory(max_samples=10)
        history.push(0.0)
        history.push(20.0)
        history.push(0.0)
        history.push(40.0)
        self.assertAlmostEqual(history.average, 15.0)

    def test_summarize_slots_aggregates_processing_slots(self) -> None:
        slots = [
            {
                "is_processing": True,
                "n_prompt_tokens_processed": 10,
                "next_token": [{"n_decoded": 3}],
            },
            {
                "is_processing": False,
                "n_prompt_tokens_processed": 99,
                "next_token": [{"n_decoded": 99}],
            },
            {
                "is_processing": True,
                "n_prompt_tokens_processed": 5,
                "next_token": [{"n_decoded": 2}],
            },
        ]
        snap = summarize_slots(slots)
        assert snap is not None
        self.assertTrue(snap.is_processing)
        self.assertEqual(snap.n_prompt_processed, 15)
        self.assertEqual(snap.n_decoded, 5)
        self.assertEqual(snap.n_prompt_cache, 0)

    def test_summarize_slots_reads_cache_tokens(self) -> None:
        slots = [
            {
                "is_processing": True,
                "n_prompt_tokens": 4096,
                "n_prompt_tokens_processed": 200,
                "n_prompt_tokens_cache": 1200,
                "next_token": [{"n_decoded": 0}],
            }
        ]
        snap = summarize_slots(slots)
        assert snap is not None
        self.assertEqual(snap.n_prompt_cache, 1200)
        self.assertEqual(snap.n_prompt_total, 4096)
        self.assertTrue(snap.in_prefill)

    def test_deferred_without_processing_is_queued(self) -> None:
        metrics = Metrics(requests_processing=0, requests_deferred=2)
        live, _, _ = compute_live_tps(metrics, [])
        self.assertEqual(live.stage, "queued")
        self.assertEqual(live.requests_deferred, 2)

    def test_prefill_with_cache_keeps_prefill_stage(self) -> None:
        metrics = Metrics(requests_processing=1)
        slots = [
            {
                "is_processing": True,
                "n_prompt_tokens": 4096,
                "n_prompt_tokens_processed": 2431,
                "n_prompt_tokens_cache": 1200,
                "next_token": [{"n_decoded": 0}],
            }
        ]
        live, _, _ = compute_live_tps(metrics, slots)
        self.assertEqual(live.stage, "prefill")
        self.assertEqual(live.n_prompt_cache, 1200)
        self.assertEqual(live.n_prompt_processed, 2431)
        self.assertEqual(live.n_prompt_total, 4096)

    def test_stage_strip_highlights_prefill_and_cache(self) -> None:
        live = LiveThroughput(
            stage="prefill",
            n_prompt_cache=1200,
            n_prompt_processed=2431,
            n_prompt_total=4096,
            source="slots",
        )
        rendered = render_text(render_stage_strip(live))
        self.assertIn("PREFILL", rendered)
        self.assertIn("CACHE", rendered)
        self.assertIn("idle", rendered)
        self.assertNotIn("IDLE", rendered)
        self.assertNotIn("GENERATE", rendered)

    def test_stage_strip_lights_queue_while_generating(self) -> None:
        live = LiveThroughput(stage="generating", requests_deferred=1, source="slots")
        rendered = render_text(render_stage_strip(live))
        self.assertIn("QUEUE", rendered)
        self.assertIn("GENERATE", rendered)
        self.assertNotIn("PREFILL", rendered)

    def test_prefill_progress_includes_cache(self) -> None:
        live = LiveThroughput(
            stage="prefill",
            prompt_tps=3800.0,
            n_prompt_processed=2431,
            n_prompt_total=4096,
            n_prompt_cache=1200,
            source="slots",
        )
        rendered = render_text(render_prefill_progress(live))
        self.assertIn("3,631/4,096", rendered)
        self.assertIn("cache 1,200", rendered)
        self.assertIn("3.8k t/s", rendered)
        self.assertTrue(any(ch in rendered for ch in "█░"))

    def test_format_last_request_recap(self) -> None:
        line = format_last_request(
            LastRequest(prompt_tokens=4096, cache_tokens=1200, gen_tokens=142, gen_tps=41.2)
        )
        assert line is not None
        self.assertIn("last: 4,096 prompt (1,200 cache) · 142 gen @ 41 t/s", line.plain)

    def test_throughput_reader_freezes_last_request(self) -> None:
        reader = ThroughputReader()
        reader.PHASE_HOLD_S = 0.0
        busy = Metrics(requests_processing=1)
        idle = Metrics(requests_processing=0)
        slots = [
            {
                "is_processing": True,
                "n_prompt_tokens": 4096,
                "n_prompt_tokens_processed": 2896,
                "n_prompt_tokens_cache": 1200,
                "next_token": [{"n_decoded": 142}],
                "tg_tps": 41.0,
            }
        ]
        live_busy = reader.update(busy, slots)
        self.assertEqual(live_busy.stage, "generating")
        self.assertIsNone(live_busy.last_request)

        live_idle = reader.update(idle, [])
        self.assertEqual(live_idle.stage, "idle")
        assert live_idle.last_request is not None
        self.assertEqual(live_idle.last_request.prompt_tokens, 4096)
        self.assertEqual(live_idle.last_request.cache_tokens, 1200)
        self.assertEqual(live_idle.last_request.gen_tokens, 142)
        self.assertAlmostEqual(live_idle.last_request.gen_tps, 41.0)

    def test_throughput_reader_tracks_state(self) -> None:
        reader = ThroughputReader()
        metrics = Metrics(requests_processing=1)
        slots_a = [{"is_processing": True, "n_prompt_tokens_processed": 0, "next_token": [{"n_decoded": 0}]}]
        slots_b = [{"is_processing": True, "n_prompt_tokens_processed": 0, "next_token": [{"n_decoded": 10}]}]

        reader.update(metrics, slots_a)
        live = reader.update(metrics, slots_b)
        self.assertGreater(live.gen_tps, 0.0)
        self.assertEqual(live.source, "slots")

        reader.clear()
        idle = reader.update(metrics, slots_b)
        self.assertEqual(idle.gen_tps, 0.0)

    def test_history_tracks_average(self) -> None:
        history = ThroughputHistory(max_samples=10)
        history.push(10.0)
        history.push(20.0)
        history.push(30.0)
        self.assertAlmostEqual(history.average, 20.0)
        self.assertEqual(history.count, 3)

    def test_history_respects_max_samples(self) -> None:
        history = ThroughputHistory(max_samples=3)
        for value in (1.0, 2.0, 3.0, 4.0):
            history.push(value)
        self.assertEqual(history.samples, [2.0, 3.0, 4.0])

    def test_sparkline_renders_block_chars(self) -> None:
        samples = [1.0, 5.0, 10.0, 20.0, 40.0]
        line = render_tps_sparkline(samples, width=5)
        rendered = render_text(line)
        self.assertEqual(len(line.plain), 5)
        self.assertIn("█", rendered)
        self.assertTrue(any(ch in rendered for ch in "▁▂▃▄▅▆▇"))

    def test_sparkline_is_always_fixed_width(self) -> None:
        short = render_tps_sparkline([10.0, 20.0], width=8)
        full = render_tps_sparkline([float(x) for x in range(20)], width=8)
        empty = render_tps_sparkline([], width=8)
        self.assertEqual(len(short.plain), 8)
        self.assertEqual(len(full.plain), 8)
        self.assertEqual(len(empty.plain), 8)

    def test_sparkline_right_aligns_partial_history(self) -> None:
        line = render_tps_sparkline([5.0, 10.0], width=6)
        self.assertTrue(line.plain.startswith("    "))
        self.assertFalse(line.plain.endswith(" "))

    def test_sparkline_scrolls_oldest_off_left(self) -> None:
        samples = [float(x) for x in range(1, 9)]
        line = render_tps_sparkline(samples, width=4)
        # Last four samples are 5..8 — line should end with a block, not whitespace.
        self.assertEqual(len(line.plain), 4)
        self.assertNotEqual(line.plain[-1], " ")

    def test_sparkline_empty_is_blank_fixed_width(self) -> None:
        line = render_tps_sparkline([], width=SPARKLINE_WIDTH)
        self.assertEqual(len(line.plain), SPARKLINE_WIDTH)
        self.assertEqual(line.plain.strip(), "")


if __name__ == "__main__":
    unittest.main()
