"""Tests for on-machine run baselines."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.baselines import RunBaseline, load_baselines, lookup_baseline, record_baseline
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

    def test_lookup_requires_matching_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baselines.json"
            record_baseline(path, _run(ctx=65_536))
            runs = load_baselines(path)
            self.assertIsNone(
                lookup_baseline(
                    runs,
                    filename="Qwen3.6-27B-Q3_K_S.gguf",
                    file_size=12_000_000_000,
                    ctx=32_768,
                    offload_ratio=1.0,
                    gpu_name="Radeon RX 7900 XTX",
                )
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

    def test_fmt_measured_tps_has_no_tilde(self) -> None:
        self.assertEqual(fmt_tps(31.2, estimated=False), "31 t/s")
        self.assertEqual(fmt_tps(None, estimated=False), "—")
        self.assertEqual(fmt_tps(31.2), "~31 t/s")


if __name__ == "__main__":
    unittest.main()
