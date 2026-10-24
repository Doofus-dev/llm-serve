"""llama-bench parser, host-isolated JSON store, and Benchmark screen smoke."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import DataTable, Input, Static

from tests.support import DISPLAY, MODEL_SLUG, OTHER_DISPLAY, QUANT, Harness
from tui.data.benchmarks import (
    BenchConfig,
    BenchRow,
    BenchTarget,
    HardwareFingerprint,
    LlamaBenchError,
    LlamaBenchMissing,
    build_bench_command,
    catalogued_ggufs,
    format_speed,
    load_benchmarks,
    parse_llama_bench_output,
    record_benchmark,
    require_llama_bench,
    row_from_run,
    rows_for_table,
    run_llama_bench,
)
from tui.data.gpu import GPUStats
from tui.screens.benchmark import PICKER_COLUMNS, RESULT_COLUMNS, BenchmarkScreen

SAMPLE_JSON = """
[
  {
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "model_type": "qwen2 7B Q4_K - Medium",
    "n_threads": 8,
    "n_gpu_layers": -1,
    "n_prompt": 512,
    "n_gen": 0,
    "n_depth": 0,
    "test_time": "2025-04-24T11:58:50Z",
    "avg_ts": 7100.002165,
    "stddev_ts": 140.341520
  },
  {
    "cpu_info": "AMD Ryzen 7 7800X3D 8-Core Processor",
    "gpu_info": "NVIDIA GeForce RTX 4080",
    "backends": "CUDA",
    "model_filename": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "model_type": "qwen2 7B Q4_K - Medium",
    "n_threads": 8,
    "n_gpu_layers": -1,
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 0,
    "test_time": "2025-04-24T11:58:51Z",
    "avg_ts": 118.881588,
    "stddev_ts": 1.041811
  }
]
"""

SAMPLE_MARKDOWN = """
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  99 |           pp512 |      7340.20 ± 23.45 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | CUDA       |  99 |           tg128 |        120.60 ± 0.59 |
"""

FINGERPRINT = HardwareFingerprint(
    hostname="box-a",
    gpu_name="NVIDIA GeForce RTX 5080",
    vram_total_mb=15872,
    cpu_model="AMD Ryzen 7 7800X3D 8-Core Processor",
)


def _row(**overrides) -> BenchRow:
    values = dict(
        hostname="box-a",
        model=MODEL_SLUG,
        quant=QUANT,
        file="demo/Demo-Q4_K_M.gguf",
        gpu_name="NVIDIA GeForce RTX 5080",
        vram_total_mb=15872,
        cpu_model="AMD Ryzen 7 7800X3D",
        n_prompt=512,
        n_gen=128,
        repetitions=5,
        pp_tps=7100.0,
        pp_stddev=140.0,
        tg_tps=118.8,
        tg_stddev=1.0,
        n_gpu_layers=-1,
        n_threads=8,
        backend="CUDA",
        updated_at="2025-04-24T11:58:51Z",
        display=DISPLAY,
    )
    values.update(overrides)
    return BenchRow(**values)


class ParserTests(unittest.TestCase):
    def test_json_pairs_pp_and_tg(self) -> None:
        measurement = parse_llama_bench_output(SAMPLE_JSON)
        self.assertAlmostEqual(measurement.pp_tps, 7100.002165)
        self.assertAlmostEqual(measurement.pp_stddev, 140.341520)
        self.assertAlmostEqual(measurement.tg_tps, 118.881588)
        self.assertAlmostEqual(measurement.tg_stddev, 1.041811)
        self.assertEqual(measurement.backend, "CUDA")
        self.assertEqual(measurement.n_gpu_layers, -1)
        self.assertEqual(measurement.n_threads, 8)
        self.assertEqual(measurement.n_prompt, 512)
        self.assertEqual(measurement.n_gen, 128)
        self.assertEqual(measurement.timestamp, "2025-04-24T11:58:51Z")

    def test_json_ignores_cuda_preamble(self) -> None:
        noisy = "ggml_cuda_init: found 1 CUDA devices\n" + SAMPLE_JSON
        measurement = parse_llama_bench_output(noisy, stderr="device warning")
        self.assertAlmostEqual(measurement.pp_tps, 7100.002165)
        self.assertAlmostEqual(measurement.tg_tps, 118.881588)

    def test_markdown_pairs_pp_and_tg(self) -> None:
        measurement = parse_llama_bench_output(SAMPLE_MARKDOWN)
        self.assertAlmostEqual(measurement.pp_tps, 7340.20)
        self.assertAlmostEqual(measurement.pp_stddev, 23.45)
        self.assertAlmostEqual(measurement.tg_tps, 120.60)
        self.assertAlmostEqual(measurement.tg_stddev, 0.59)
        self.assertEqual(measurement.backend, "CUDA")
        self.assertEqual(measurement.n_gpu_layers, 99)
        self.assertEqual(measurement.n_prompt, 512)
        self.assertEqual(measurement.n_gen, 128)

    def test_empty_output_fails_loud(self) -> None:
        with self.assertRaises(LlamaBenchError) as ctx:
            parse_llama_bench_output("", stderr="loader exploded")
        self.assertIn("loader exploded", str(ctx.exception))


class CommandAndRunnerTests(unittest.TestCase):
    def test_command_uses_fixed_defaults_and_local_gguf(self) -> None:
        cmd = build_bench_command(
            Path("/opt/llama-bench"),
            [Path("/models/a.gguf"), Path("/models/a-00002-of-00002.gguf")],
            BenchConfig(),
        )
        self.assertEqual(cmd[0], "/opt/llama-bench")
        self.assertEqual(
            cmd[1:],
            [
                "-m",
                "/models/a.gguf",
                "-m",
                "/models/a-00002-of-00002.gguf",
                "-p",
                "512",
                "-n",
                "128",
                "-r",
                "5",
                "-o",
                "json",
            ],
        )

    def test_missing_binary_fails_loud(self) -> None:
        missing = Path("/no/such/llama-bench")
        with self.assertRaises(LlamaBenchMissing) as ctx:
            require_llama_bench(missing)
        self.assertIn(str(missing), str(ctx.exception))
        with self.assertRaises(LlamaBenchMissing):
            run_llama_bench(
                binary=missing,
                model_paths=[Path("demo.gguf")],
                runner=lambda *a, **k: self.fail("subprocess must not start"),
            )

    def test_run_parses_mocked_subprocess(self) -> None:
        seen = {}

        def runner(cmd, **kwargs):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout=SAMPLE_JSON, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-bench"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
            gguf = Path(tmp) / "demo.gguf"
            gguf.write_bytes(b"GGUF")
            measurement = run_llama_bench(
                binary=binary,
                model_paths=[gguf],
                runner=runner,
            )
        self.assertEqual(seen["cmd"][0], str(binary))
        self.assertIn(str(gguf), seen["cmd"])
        self.assertAlmostEqual(measurement.tg_tps, 118.881588)

    def test_nonzero_exit_fails_loud(self) -> None:
        def runner(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="CUDA out of memory")

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-bench"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
            with self.assertRaises(LlamaBenchError) as ctx:
                run_llama_bench(
                    binary=binary,
                    model_paths=[Path(tmp) / "demo.gguf"],
                    runner=runner,
                )
        self.assertIn("CUDA out of memory", str(ctx.exception))


class StoreTests(unittest.TestCase):
    def test_other_host_row_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tui-benchmarks.json"
            record_benchmark(path, _row(hostname="box-a", pp_tps=100.0, tg_tps=10.0))
            record_benchmark(path, _row(hostname="box-b", pp_tps=50.0, tg_tps=5.0, gpu_name="RX 7900 XTX"))
            stored = load_benchmarks(path)
            by_host = {row.hostname: row for row in stored}
            self.assertEqual(set(by_host), {"box-a", "box-b"})
            self.assertAlmostEqual(by_host["box-a"].pp_tps, 100.0)
            self.assertAlmostEqual(by_host["box-b"].pp_tps, 50.0)
            self.assertEqual(by_host["box-a"].gpu_name, "NVIDIA GeForce RTX 5080")
            self.assertEqual(by_host["box-b"].gpu_name, "RX 7900 XTX")

    def test_rerun_replaces_same_host_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tui-benchmarks.json"
            record_benchmark(path, _row(hostname="box-a", pp_tps=100.0))
            record_benchmark(path, _row(hostname="box-b", pp_tps=50.0))
            record_benchmark(path, _row(hostname="box-a", pp_tps=200.0, tg_tps=20.0))
            stored = load_benchmarks(path)
            by_host = {row.hostname: row for row in stored}
            self.assertEqual(len(stored), 2)
            self.assertAlmostEqual(by_host["box-a"].pp_tps, 200.0)
            self.assertAlmostEqual(by_host["box-a"].tg_tps, 20.0)
            self.assertAlmostEqual(by_host["box-b"].pp_tps, 50.0)

    def test_different_config_does_not_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tui-benchmarks.json"
            record_benchmark(path, _row(n_prompt=512, pp_tps=100.0))
            record_benchmark(path, _row(n_prompt=256, pp_tps=80.0))
            stored = load_benchmarks(path)
            self.assertEqual(len(stored), 2)
            by_pp = {row.n_prompt: row.pp_tps for row in stored}
            self.assertAlmostEqual(by_pp[512], 100.0)
            self.assertAlmostEqual(by_pp[256], 80.0)

    def test_table_puts_this_host_first_in_a_model_quant_group(self) -> None:
        ordered = rows_for_table(
            [
                _row(hostname="box-b", model="demo-model", quant="Q4_K_M"),
                _row(hostname="box-a", model="demo-model", quant="Q4_K_M"),
                _row(hostname="box-a", model="other-model", quant="Q8_0", display="Other"),
            ],
            this_hostname="box-a",
        )
        self.assertEqual(
            [(row.hostname, row.model, row.quant) for row in ordered],
            [
                ("box-a", "demo-model", "Q4_K_M"),
                ("box-b", "demo-model", "Q4_K_M"),
                ("box-a", "other-model", "Q8_0"),
            ],
        )

    def test_format_speed_includes_stddev(self) -> None:
        self.assertEqual(format_speed(7100.002, 140.34), "7100.00 ± 140.34")
        self.assertEqual(format_speed(None), "—")

    def test_store_does_not_touch_live_baselines_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "tui-benchmarks.json"
            baselines = Path(tmp) / "tui-baselines.json"
            record_benchmark(bench, _row())
            self.assertTrue(bench.exists())
            self.assertFalse(baselines.exists())
            payload = json.loads(bench.read_text())
            self.assertIn("runs", payload)
            self.assertEqual(payload["runs"][0]["hostname"], "box-a")


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def test_lists_on_disk_catalog_entries_only(self) -> None:
        app = self.harness.app()
        targets = catalogued_ggufs(app.registry, self.harness.paths.models_dir)
        keys = {target.key for target in targets}
        self.assertIn(f"{MODEL_SLUG}/{QUANT}", keys)
        self.assertTrue(all(path.is_file() for target in targets for path in target.paths))


class BenchmarkScreenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def _screen(self, **kwargs) -> BenchmarkScreen:
        defaults = dict(
            registry=self.harness.app().registry,
            paths=self.harness.paths,
            gpu=GPUStats(
                name="NVIDIA GeForce RTX 5080",
                vram_total_mb=15872,
                available=True,
            ),
            fingerprint=FINGERPRINT,
        )
        defaults.update(kwargs)
        return BenchmarkScreen(**defaults)

    async def test_screen_mounts_with_table_columns(self) -> None:
        app = self.harness.app()
        screen = self._screen(registry=app.registry)
        async with app.run_test(size=(140, 50)) as pilot:
            app.push_screen(screen)
            await pilot.pause()
            picker = screen.query_one("#bench-picker", DataTable)
            results = screen.query_one("#bench-results", DataTable)
            self.assertEqual(
                [str(col.label) for col in picker.columns.values()],
                list(PICKER_COLUMNS),
            )
            self.assertEqual(
                [str(col.label) for col in results.columns.values()],
                list(RESULT_COLUMNS),
            )
            picker_text = " ".join(
                str(cell) for row in range(picker.row_count) for cell in picker.get_row_at(row)
            )
            self.assertIn(DISPLAY, picker_text)
            self.assertIn(OTHER_DISPLAY, picker_text)
            self.assertIn(QUANT, picker_text)
            status = str(screen.query_one("#bench-status", Static).render())
            self.assertIn("llama-bench not found", status)
            knobs = (
                screen.query_one("#bench-pp", Input).value,
                screen.query_one("#bench-n", Input).value,
                screen.query_one("#bench-r", Input).value,
            )
            self.assertEqual(knobs, ("512", "128", "5"))
            hardware = str(screen.query_one("#bench-hardware", Static).render())
            self.assertIn("box-a", hardware)
            self.assertIn("RTX 5080", hardware)

    async def test_open_from_dashboard_binding(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause(0.3)
            with patch("tui.app.query_gpu", return_value=GPUStats(name="RTX 5080", available=True)):
                app.action_open_benchmark()
                await pilot.pause()
            self.assertIsInstance(app.screen, BenchmarkScreen)
            results = app.screen.query_one("#bench-results", DataTable)
            self.assertEqual(
                [str(col.label) for col in results.columns.values()],
                list(RESULT_COLUMNS),
            )

    async def test_run_writes_this_host_row_and_keeps_other_host(self) -> None:
        record_benchmark(
            self.harness.paths.benchmarks_json,
            _row(hostname="box-b", pp_tps=50.0, tg_tps=5.0, gpu_name="RX 7900 XTX"),
        )
        binary = self.harness.paths.llama_bench
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        seen = {}

        def runner(cmd, **kwargs):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout=SAMPLE_JSON, stderr="")

        app = self.harness.app()
        screen = self._screen(registry=app.registry, runner=runner)
        async with app.run_test(size=(140, 50)) as pilot:
            app.push_screen(screen)
            await pilot.pause()
            screen.picked = {f"{MODEL_SLUG}/{QUANT}"}
            screen._render_picker()
            screen.action_run_bench()
            workers = [worker for worker in screen.workers if worker.state.name != "CANCELLED"]
            if workers:
                await workers[0].wait()
            await pilot.pause()
            self.assertIn(str(binary), seen.get("cmd", []))
            self.assertIn("-p", seen["cmd"])
            self.assertIn("512", seen["cmd"])
            stored = load_benchmarks(self.harness.paths.benchmarks_json)
            by_host = {row.hostname: row for row in stored}
            self.assertEqual(set(by_host), {"box-a", "box-b"})
            self.assertAlmostEqual(by_host["box-a"].pp_tps, 7100.002165)
            self.assertAlmostEqual(by_host["box-b"].pp_tps, 50.0)
            self.assertEqual(screen.status_by_key[f"{MODEL_SLUG}/{QUANT}"], "done")
            results = screen.query_one("#bench-results", DataTable)
            hosts = [str(results.get_row_at(i)[0]) for i in range(results.row_count)]
            self.assertTrue(any("box-a" in host for host in hosts), hosts)
            self.assertTrue(any("box-b" in host for host in hosts), hosts)
            picker = screen.query_one("#bench-picker", DataTable)
            statuses = [str(picker.get_row_at(i)[4]) for i in range(picker.row_count)]
            self.assertIn("done", statuses)

    async def test_run_without_binary_marks_failed(self) -> None:
        app = self.harness.app()
        screen = self._screen(registry=app.registry)
        async with app.run_test(size=(140, 50)) as pilot:
            app.push_screen(screen)
            await pilot.pause()
            screen.picked = {f"{MODEL_SLUG}/{QUANT}"}
            screen.action_run_bench()
            await pilot.pause()
            self.assertEqual(screen.status_by_key[f"{MODEL_SLUG}/{QUANT}"], "failed")
            status = str(screen.query_one("#bench-status", Static).render())
            self.assertIn("llama-bench not found", status)


class RowFromRunTests(unittest.TestCase):
    def test_row_carries_fingerprint_and_catalog_identity(self) -> None:
        measurement = parse_llama_bench_output(SAMPLE_JSON)
        target = BenchTarget(
            model=MODEL_SLUG,
            display=DISPLAY,
            quant=QUANT,
            file_rel="demo/Demo-Q4_K_M.gguf",
            paths=(Path("demo/Demo-Q4_K_M.gguf"),),
        )
        row = row_from_run(
            target=target,
            measurement=measurement,
            fingerprint=FINGERPRINT,
            config=BenchConfig(),
        )
        self.assertEqual(row.hostname, "box-a")
        self.assertEqual(row.gpu_name, "NVIDIA GeForce RTX 5080")
        self.assertEqual(row.cpu_model, FINGERPRINT.cpu_model)
        self.assertEqual(row.model, MODEL_SLUG)
        self.assertEqual(row.quant, QUANT)
        payload = json.dumps(row.__dict__)
        self.assertIn("box-a", payload)
        self.assertIn("15872", payload)
