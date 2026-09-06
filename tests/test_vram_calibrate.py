"""Calibrate VRAM / tok/s estimates from on-machine run baselines."""

from __future__ import annotations

import unittest

from tui.data.baselines import RunBaseline
from tui.data.gpu import GPUStats
from tui.data.vram import (
    estimate_gen_tps,
    estimate_gen_tps_calibrated,
    estimate_vram_calibrated,
    estimate_vram_mb,
)


def _run(**overrides) -> RunBaseline:
    values = dict(
        model="qwen35-9b-bartowski",
        file="bartowski/Qwen_Qwen3.5-9B-Q5_K_L.gguf",
        file_size=7_739_240_480,
        gpu_name="NVIDIA GeForce RTX 5080",
        ctx=131_072,
        gpu_layers=99,
        total_layers=33,
        cache_k="q4_0",
        cache_v="q4_0",
        vram_used_mb=9_609.0,
        gen_tps=84.0,
    )
    values.update(overrides)
    return RunBaseline(**values)


class VRAMCalibrateTests(unittest.TestCase):
    def test_uncalibrated_qwen35_is_far_above_actual(self) -> None:
        naive = estimate_vram_mb(7_739_240_480, 131_072)
        self.assertGreater(naive, 12_000)

    def test_sibling_run_reconstructs_measured_quant(self) -> None:
        calibrated = estimate_vram_calibrated(
            7_739_240_480,
            131_072,
            1.0,
            runs=[_run()],
            gpu_name="NVIDIA GeForce RTX 5080",
            filename="Qwen_Qwen3.5-9B-Q5_K_L.gguf",
        )
        self.assertAlmostEqual(calibrated, 9_609.0, delta=80.0)

    def test_sibling_run_scales_other_quants_by_file_size(self) -> None:
        q4_size = 6_200_000_000
        calibrated = estimate_vram_calibrated(
            q4_size,
            131_072,
            1.0,
            runs=[_run()],
            gpu_name="NVIDIA GeForce RTX 5080",
            filename="Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        )
        naive = estimate_vram_mb(q4_size, 131_072)
        expected = 9_609.0 - (7_739.240480 - 6_200.0)
        self.assertAlmostEqual(calibrated, expected, delta=80.0)
        self.assertLess(calibrated, naive)
        self.assertLess(abs(calibrated - expected), abs(naive - expected))

    def test_two_contexts_fit_kv_growth(self) -> None:
        runs = [
            _run(
                file="bartowski/NousResearch_Hermes-4-14B-Q5_K_M.gguf",
                file_size=10_514_570_176,
                ctx=32_768,
                vram_used_mb=12_696.0,
                total_layers=40,
            ),
            _run(
                file="bartowski/NousResearch_Hermes-4-14B-Q5_K_M.gguf",
                file_size=10_514_570_176,
                ctx=65_000,
                vram_used_mb=14_269.0,
                total_layers=40,
            ),
        ]
        at_65k = estimate_vram_calibrated(
            10_514_570_176,
            65_000,
            1.0,
            runs=runs,
            gpu_name="NVIDIA GeForce RTX 5080",
            filename="NousResearch_Hermes-4-14B-Q5_K_M.gguf",
        )
        self.assertAlmostEqual(at_65k, 14_269.0, delta=80.0)

    def test_other_family_does_not_use_qwen_kv(self) -> None:
        hermes = estimate_vram_calibrated(
            10_514_570_176,
            65_536,
            1.0,
            runs=[_run()],
            gpu_name="NVIDIA GeForce RTX 5080",
            filename="NousResearch_Hermes-4-14B-Q5_K_M.gguf",
        )
        naive = estimate_vram_mb(10_514_570_176, 65_536)
        qwen = estimate_vram_calibrated(
            7_739_240_480,
            131_072,
            1.0,
            runs=[_run()],
            gpu_name="NVIDIA GeForce RTX 5080",
            filename="Qwen_Qwen3.5-9B-Q5_K_L.gguf",
        )
        # GPU-wide scale of the generic heuristic, not the Qwen residual.
        self.assertNotAlmostEqual(hermes, qwen, delta=500)
        self.assertGreater(hermes, 0)
        self.assertGreater(naive, 0)

    def test_tps_scales_from_measured_sibling(self) -> None:
        gpu = GPUStats(name="NVIDIA GeForce RTX 5080", vram_total_mb=16_000)
        naive = estimate_gen_tps(6_200_000_000, 131_072, gpu, 1.0)
        calibrated = estimate_gen_tps_calibrated(
            6_200_000_000,
            131_072,
            gpu,
            1.0,
            runs=[_run()],
            filename="Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        )
        self.assertIsNotNone(naive)
        self.assertIsNotNone(calibrated)
        self.assertGreater(calibrated, naive)


if __name__ == "__main__":
    unittest.main()
