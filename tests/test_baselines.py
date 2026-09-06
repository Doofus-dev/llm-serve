"""Tests for on-machine run baselines."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tui.data.baselines import (
    RunBaseline,
    latest_baseline_ctx,
    load_baselines,
    lookup_baseline,
    record_baseline,
)
from tui.data.vram import fmt_tps


def _run(**overrides) -> RunBaseline:
    values = dict(
        model="qwen36",
        file="bartowski/Qwen3.6-27B-Q3_K_S.gguf",
        file_size=12_000_000_000,
        gpu_name="Radeon RX 7900 XTX",
        ctx=65_536,
        gpu_layers=99,
        total_layers=64,
        cache_k="q4_0",
        cache_v="q4_0",
        vram_used_mb=18_400.0,
        gen_tps=31.2,
        prompt_tps=140.0,
        tokens_predicted=400.0,
    )
    values.update(overrides)
    return RunBaseline(**values)


class BaselineStoreTests(unittest.TestCase):
    def test_record_then_lookup_by_filename_and_ctx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(path, _run())
            match = lookup_baseline(
                load_baselines(path),
                filename="Qwen3.6-27B-Q3_K_S.gguf",
                file_size=12_000_000_000,
                ctx=65_536,
                offload_ratio=1.0,
                gpu_name="Radeon RX 7900 XTX",
            )
            self.assertIsNotNone(match)
            self.assertAlmostEqual(match.vram_used_mb, 18_400.0)
            self.assertAlmostEqual(match.gen_tps, 31.2)

    def test_idle_zero_tps_does_not_erase_measured_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(path, _run(gen_tps=31.2, tokens_predicted=400))
            record_baseline(
                path,
                _run(gen_tps=None, tokens_predicted=400, vram_used_mb=18_410.0),
            )
            stored = load_baselines(path)[0]
            self.assertAlmostEqual(stored.gen_tps, 31.2)

    def test_lookup_still_shows_actuals_at_other_context(self) -> None:
        run = _run(ctx=131_072)
        match = lookup_baseline(
            [run],
            filename="Qwen3.6-27B-Q3_K_S.gguf",
            file_size=12_000_000_000,
            ctx=65_536,
            offload_ratio=1.0,
            gpu_name="Radeon RX 7900 XTX",
        )
        self.assertIs(match, run)

    def test_lookup_prefers_closer_context(self) -> None:
        at_64k = _run(ctx=65_536, vram_used_mb=16_000.0)
        at_128k = _run(ctx=131_072, vram_used_mb=18_400.0)
        match = lookup_baseline(
            [at_64k, at_128k],
            filename="Qwen3.6-27B-Q3_K_S.gguf",
            file_size=12_000_000_000,
            ctx=65_536,
            offload_ratio=1.0,
            gpu_name="Radeon RX 7900 XTX",
        )
        self.assertIs(match, at_64k)

    def test_latest_baseline_ctx_uses_newest_file_on_gpu(self) -> None:
        older = _run(ctx=65_536, updated_at="2026-09-01T00:00:00Z")
        newer = _run(ctx=131_072, updated_at="2026-09-06T00:00:00Z")
        self.assertEqual(
            latest_baseline_ctx(
                [older, newer],
                filenames=["Qwen3.6-27B-Q3_K_S.gguf"],
                gpu_name="Radeon RX 7900 XTX",
            ),
            131_072,
        )

    def test_full_offload_matches_ngl_99(self) -> None:
        run = _run(gpu_layers=99, total_layers=64)
        self.assertEqual(run.offload_ratio, 1.0)
        match = lookup_baseline(
            [run],
            filename="Qwen3.6-27B-Q3_K_S.gguf",
            file_size=12_000_000_000,
            ctx=65_536,
            offload_ratio=1.0,
            gpu_name="Radeon RX 7900 XTX",
        )
        self.assertIs(match, run)

    def test_lookup_does_not_match_similar_sized_other_file(self) -> None:
        hermes = _run(
            file="bartowski/NousResearch_Hermes-4-14B-Q5_K_M.gguf",
            file_size=10_514_570_176,
            ctx=65_000,
        )
        self.assertIsNone(
            lookup_baseline(
                [hermes],
                filename="Qwen3.8-27B-IQ2_XS.gguf",
                file_size=9_986_799_200,
                ctx=65_536,
                offload_ratio=1.0,
                gpu_name="Radeon RX 7900 XTX",
            )
        )

    def test_fmt_measured_tps_has_no_tilde(self) -> None:
        self.assertEqual(fmt_tps(31.2, estimated=False), "31 t/s")
        self.assertEqual(fmt_tps(None, estimated=False), "—")
        self.assertEqual(fmt_tps(31.2), "~31 t/s")

    def test_lookup_matches_nearby_preset_context(self) -> None:
        run = _run(ctx=65_000)
        match = lookup_baseline(
            [run],
            filename="Qwen3.6-27B-Q3_K_S.gguf",
            file_size=12_000_000_000,
            ctx=65_536,
            offload_ratio=1.0,
            gpu_name="Radeon RX 7900 XTX",
        )
        self.assertIs(match, run)

    def test_lookup_matches_nvidia_smi_and_pci_gpu_names(self) -> None:
        run = _run(gpu_name="NVIDIA GeForce RTX 5080")
        match = lookup_baseline(
            [run],
            filename="Qwen3.6-27B-Q3_K_S.gguf",
            file_size=12_000_000_000,
            ctx=65_536,
            offload_ratio=1.0,
            gpu_name="NVIDIA Corporation GB203 [GeForce RTX 5080] (rev a1) (unified)",
        )
        self.assertIs(match, run)

    def test_lookup_prefers_measured_run_over_idle(self) -> None:
        idle = _run(
            gpu_name="NVIDIA Corporation GB203 [GeForce RTX 5080] (rev a1)",
            vram_used_mb=14.0,
            gen_tps=None,
            prompt_tps=None,
        )
        measured = _run(
            gpu_name="NVIDIA GeForce RTX 5080",
            vram_used_mb=18_400.0,
            gen_tps=31.2,
        )
        match = lookup_baseline(
            [idle, measured],
            filename="Qwen3.6-27B-Q3_K_S.gguf",
            file_size=12_000_000_000,
            ctx=65_536,
            offload_ratio=1.0,
            gpu_name="NVIDIA GeForce RTX 5080",
        )
        self.assertIs(match, measured)

    def test_record_skips_idle_vram_without_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            self.assertFalse(
                record_baseline(path, _run(vram_used_mb=14.0, gen_tps=None, prompt_tps=None))
            )
            self.assertEqual(load_baselines(path), [])

    def test_recent_live_speed_replaces_stale_lifetime_average(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(path, _run(gen_tps=47.8, tokens_predicted=179_359))
            record_baseline(path, _run(gen_tps=87.4, tokens_predicted=80))
            stored = load_baselines(path)[0]
            self.assertAlmostEqual(stored.gen_tps, 87.4)

    def test_slower_later_sample_does_not_lower_peak_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(path, _run(gen_tps=92.0, tokens_predicted=80))
            record_baseline(path, _run(gen_tps=61.0, tokens_predicted=400))
            stored = load_baselines(path)[0]
            self.assertAlmostEqual(stored.gen_tps, 92.0)

    def test_record_does_not_replace_vram_with_idle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(path, _run(vram_used_mb=18_400.0, gen_tps=31.2))
            record_baseline(
                path,
                _run(vram_used_mb=14.0, gen_tps=None, prompt_tps=None, tokens_predicted=400),
            )
            stored = load_baselines(path)[0]
            self.assertAlmostEqual(stored.vram_used_mb, 18_400.0)
            self.assertAlmostEqual(stored.gen_tps, 31.2)


if __name__ == "__main__":
    unittest.main()
