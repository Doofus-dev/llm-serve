"""Tests for Hugging Face integration helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.hf import (
    HF_INSTALL_HINT,
    HUB_FILTER_FETCH_LIMIT,
    HUB_LIST_LIMIT,
    HUB_REPO_EXPAND,
    AuthStatus,
    DownloadPlan,
    HubFile,
    HubRepo,
    take_hf_progress_lines,
    local_download_bytes,
    build_download_plan,
    build_source_metadata,
    download_files,
    filter_hub_repos,
    fmt_size,
    fmt_count,
    hf_available,
    list_gguf_repos,
    list_repo_ggufs,
    shard_filenames,
)
from tui.data.models_json import create_downloaded_model, load_registry, merge_editor_params, update_model
from tui.data.settings import TUISettings, load_settings, remember_hf_author, save_settings
from tui.data.gpu import GPUStats, effective_gpu_memory, parse_rocm_csv
from tui.data.vram import (
    classify_vram,
    estimate_gen_tps,
    estimate_vram_mb,
    fmt_tps,
    gpu_bandwidth_gb_s,
)


class HFPathTests(unittest.TestCase):
    def test_fmt_count(self) -> None:
        self.assertEqual(fmt_count(9354057), "9,354,057")
        self.assertEqual(fmt_count(0), "0")

    def test_build_download_plan_flat_filename(self) -> None:
        models_dir = Path("/repo/models")
        plan = build_download_plan("bartowski/Qwen3.6-27B-GGUF", "Qwen3.6-27B-Q3_K_S.gguf", models_dir)
        self.assertEqual(plan.author, "bartowski")
        self.assertEqual(plan.local_dir, models_dir / "bartowski")
        self.assertEqual(plan.relative_file, "bartowski/Qwen3.6-27B-Q3_K_S.gguf")
        self.assertEqual(plan.filenames, ["Qwen3.6-27B-Q3_K_S.gguf"])

    def test_shard_filenames(self) -> None:
        all_files = [
            "Model-00001-of-00003.gguf",
            "Model-00002-of-00003.gguf",
            "Model-00003-of-00003.gguf",
            "Model-Q4_K_M.gguf",
        ]
        shards = shard_filenames("Model-00001-of-00003.gguf", all_files)
        self.assertEqual(
            shards,
            ["Model-00001-of-00003.gguf", "Model-00002-of-00003.gguf", "Model-00003-of-00003.gguf"],
        )
        self.assertEqual(shard_filenames("Model-Q4_K_M.gguf", all_files), ["Model-Q4_K_M.gguf"])

    def test_build_source_metadata(self) -> None:
        models_dir = Path("/repo/models")
        plan = build_download_plan("unsloth/Qwen3.6-27B-GGUF", "Qwen3.6-27B-UD-Q3_K_XL.gguf", models_dir)
        source = build_source_metadata(plan, "Qwen3.6-27B-UD-Q3_K_XL.gguf")
        self.assertEqual(source["author"], "unsloth")
        self.assertEqual(source["repo"], "unsloth/Qwen3.6-27B-GGUF")

    def test_take_hf_progress_lines_splits_on_cr_and_lf(self) -> None:
        lines, leftover = take_hf_progress_lines(b"12%|xxx\r45%|yyy\nDone")
        self.assertEqual(lines, ["12%|xxx", "45%|yyy"])
        self.assertEqual(leftover, b"Done")

    def test_take_hf_progress_lines_skips_empty_crlf(self) -> None:
        lines, leftover = take_hf_progress_lines(b"ok\r\n")
        self.assertEqual(lines, ["ok"])
        self.assertEqual(leftover, b"")

    def test_local_download_bytes_counts_file_and_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            plan = build_download_plan("bartowski/Demo-GGUF", "Demo-Q4.gguf", models_dir)
            plan.local_dir.mkdir(parents=True)
            self.assertEqual(local_download_bytes(plan), 0)
            (plan.local_dir / "Demo-Q4.gguf.incomplete").write_bytes(b"x" * 50)
            self.assertEqual(local_download_bytes(plan), 50)
            (plan.local_dir / "Demo-Q4.gguf").write_bytes(b"y" * 80)
            self.assertEqual(local_download_bytes(plan), 80)

    def test_local_download_bytes_counts_hf_cache_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            plan = build_download_plan("bartowski/Demo-GGUF", "Demo-Q4.gguf", models_dir)
            cache = plan.local_dir / ".cache" / "huggingface" / "download"
            cache.mkdir(parents=True)
            (cache / "Demo-Q4.gguf.lock").write_bytes(b"")
            (cache / "abc.etag.incomplete").write_bytes(b"z" * 120)
            self.assertEqual(local_download_bytes(plan), 120)

    def test_local_download_bytes_ignores_stale_lock_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            plan = build_download_plan("bartowski/Demo-GGUF", "Demo-Q4.gguf", models_dir)
            cache = plan.local_dir / ".cache" / "huggingface" / "download"
            cache.mkdir(parents=True)
            (cache / "Old-A.gguf.lock").write_bytes(b"")
            (cache / "Old-B.gguf.lock").write_bytes(b"")
            (cache / "Demo-Q4.gguf.lock").write_bytes(b"")
            (cache / "current.etag.incomplete").write_bytes(b"z" * 120)

            self.assertEqual(local_download_bytes(plan), 120)

    def test_local_download_bytes_matches_metadata_etag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            plan = build_download_plan("bartowski/Demo-GGUF", "Demo-Q4.gguf", models_dir)
            cache = plan.local_dir / ".cache" / "huggingface" / "download"
            cache.mkdir(parents=True)
            (cache / "Demo-Q4.gguf.lock").write_bytes(b"")
            (cache / "Demo-Q4.gguf.metadata").write_text(
                "revision\nexpected-etag\n0\n"
            )
            (cache / "expected-etag.incomplete").write_bytes(b"x" * 80)

            self.assertEqual(local_download_bytes(plan), 80)


class SettingsTests(unittest.TestCase):
    def test_hf_authors_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tui-settings.json"
            settings = TUISettings(theme="gruvbox", hf_authors=["bartowski"])
            save_settings(path, settings)
            loaded = load_settings(path)
            self.assertEqual(loaded.theme, "gruvbox")
            self.assertEqual(loaded.hf_authors, ["bartowski"])
            self.assertEqual(loaded.log_verbosity, 4)
            self.assertFalse(loaded.remote_launch)

            self.assertTrue(remember_hf_author(loaded, "unsloth"))
            self.assertFalse(remember_hf_author(loaded, "unsloth"))
            self.assertEqual(loaded.hf_authors, ["bartowski", "unsloth"])

    def test_log_verbosity_round_trip_and_cycle(self) -> None:
        from tui.data.settings import cycle_log_verbosity, DEFAULT_LOG_VERBOSITY

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tui-settings.json"
            settings = TUISettings(log_verbosity=5)
            save_settings(path, settings)
            loaded = load_settings(path)
            self.assertEqual(loaded.log_verbosity, 5)

        self.assertEqual(DEFAULT_LOG_VERBOSITY, 4)
        self.assertEqual(cycle_log_verbosity(3), 4)
        self.assertEqual(cycle_log_verbosity(4), 5)
        self.assertEqual(cycle_log_verbosity(5), 3)
        self.assertEqual(cycle_log_verbosity(99), 4)

    def test_remote_launch_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tui-settings.json"
            settings = TUISettings(remote_launch=True)
            save_settings(path, settings)
            loaded = load_settings(path)
            self.assertTrue(loaded.remote_launch)

            loaded.remote_launch = False
            save_settings(path, loaded)
            self.assertFalse(load_settings(path).remote_launch)


class ModelJsonTests(unittest.TestCase):
    def test_merge_editor_params_preserves_source(self) -> None:
        existing = {
            "file": "bartowski/model.gguf",
            "ctx": 65536,
            "source": {"hub": "huggingface", "author": "bartowski"},
        }
        edited = {"file": "bartowski/model.gguf", "ctx": 32768, "port": 8081}
        merged = merge_editor_params(existing, edited, {"file", "ctx", "port"})
        self.assertEqual(merged["ctx"], 32768)
        self.assertEqual(merged["source"], existing["source"])

    def test_create_downloaded_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(json.dumps({
                "models": {
                    "base": {"file": "old.gguf", "ctx": 4096, "port": 8081},
                },
                "aliases": {},
            }))
            create_downloaded_model(
                path,
                "qwen36-bart",
                {"file": "old.gguf", "ctx": 4096, "port": 8081, "total_layers": 28},
                file_rel="bartowski/Qwen3.6-27B-Q3_K_S.gguf",
                source={
                    "hub": "huggingface",
                    "repo": "bartowski/Qwen3.6-27B-GGUF",
                    "filename": "Qwen3.6-27B-Q3_K_S.gguf",
                    "author": "bartowski",
                    "revision": "main",
                },
            )
            reg = load_registry(path)
            self.assertIn("qwen36-bart", reg.models)
            params = reg.models["qwen36-bart"].params
            self.assertEqual(params["file"], "bartowski/Qwen3.6-27B-Q3_K_S.gguf")
            self.assertEqual(params["source"]["author"], "bartowski")
            self.assertNotIn("total_layers", params)

            update_model(path, "qwen36-bart", merge_editor_params(params, {"ctx": 8192}, {"ctx"}))
            reg2 = load_registry(path)
            self.assertEqual(reg2.models["qwen36-bart"].params["source"]["author"], "bartowski")


class HFCliTests(unittest.TestCase):
    @patch("tui.data.hf._run_hf")
    def test_list_gguf_repos(self, mock_run) -> None:
        from tui.data.hf import list_gguf_repos

        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps([
            {"id": "bartowski/Qwen3.6-27B-GGUF", "downloads": 1, "likes": 2, "trending_score": 3},
        ])
        with patch("tui.data.hf.hf_available", return_value=True):
            repos, error = list_gguf_repos(author="bartowski")
        self.assertIsNone(error)
        self.assertEqual(len(repos), 1)
        self.assertEqual(repos[0].author, "bartowski")
        self.assertEqual(repos[0].downloads, 1)
        self.assertEqual(repos[0].likes, 2)
        expand_idx = mock_run.call_args.args[0].index("--expand")
        self.assertEqual(mock_run.call_args.args[0][expand_idx + 1], HUB_REPO_EXPAND)
        limit_idx = mock_run.call_args.args[0].index("--limit")
        self.assertEqual(mock_run.call_args.args[0][limit_idx + 1], str(HUB_LIST_LIMIT))
        self.assertEqual(HUB_LIST_LIMIT, 50)

    @patch("tui.data.hf._run_hf")
    def test_list_gguf_repos_filters_min_context(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = json.dumps([
            {
                "id": "bartowski/Small-GGUF",
                "downloads": 1,
                "likes": 0,
                "gguf": {"context_length": 32768},
            },
            {
                "id": "bartowski/Long-GGUF",
                "downloads": 2,
                "likes": 0,
                "gguf": {"context_length": 131072},
            },
            {
                "id": "bartowski/Unknown-GGUF",
                "downloads": 3,
                "likes": 0,
            },
        ])
        with patch("tui.data.hf.hf_available", return_value=True):
            repos, error = list_gguf_repos(min_context=65_536)
        self.assertIsNone(error)
        self.assertEqual([repo.id for repo in repos], ["bartowski/Long-GGUF"])
        limit_idx = mock_run.call_args.args[0].index("--limit")
        self.assertEqual(mock_run.call_args.args[0][limit_idx + 1], str(HUB_FILTER_FETCH_LIMIT))

    def test_filter_hub_repos_keeps_context_at_or_above(self) -> None:
        short = HubRepo(id="a/short", author="a", downloads=0, likes=0, context_length=32_768)
        long = HubRepo(id="a/long", author="a", downloads=0, likes=0, context_length=131_072)
        unknown = HubRepo(id="a/unknown", author="a", downloads=0, likes=0)
        self.assertEqual(
            filter_hub_repos([short, long, unknown], min_context=65_536),
            [long],
        )
        self.assertEqual(
            filter_hub_repos([short, long, unknown], min_context=None),
            [short, long, unknown],
        )


class VRAMEstimateTests(unittest.TestCase):
    def test_rocm_csv_parses_vram_columns(self) -> None:
        stats = parse_rocm_csv(
            "device,Temperature (Sensor edge) (C),GPU use (%),"
            "VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
            "card0,63.0,19,536870912,501817344\n"
        )
        assert stats is not None
        self.assertAlmostEqual(stats.vram_total_mb, 536.870912)
        self.assertAlmostEqual(stats.vram_used_mb, 501.817344)
        self.assertAlmostEqual(stats.utilization_pct, 19.0)

    def test_apu_uses_gtt_not_tiny_vram_bar(self) -> None:
        used, total, unified = effective_gpu_memory(
            vram_used_mb=505.0,
            vram_total_mb=536.0,
            gtt_used_mb=9_200.0,
            gtt_total_mb=16_456.0,
        )
        self.assertTrue(unified)
        self.assertAlmostEqual(used, 9_200.0)
        self.assertAlmostEqual(total, 16_456.0)

    def test_discrete_gpu_keeps_dedicated_vram(self) -> None:
        used, total, unified = effective_gpu_memory(
            vram_used_mb=18_000.0,
            vram_total_mb=24_576.0,
            gtt_used_mb=1_200.0,
            gtt_total_mb=16_000.0,
        )
        self.assertFalse(unified)
        self.assertAlmostEqual(used, 18_000.0)
        self.assertAlmostEqual(total, 24_576.0)

    def test_estimate_increases_with_context(self) -> None:
        at_64k = estimate_vram_mb(10_000_000_000, 65_536)
        at_128k = estimate_vram_mb(10_000_000_000, 131_072)
        self.assertGreater(at_128k, at_64k)

    def test_classification_uses_gpu_pool_total(self) -> None:
        gpu = GPUStats(vram_total_mb=24_000, vram_used_mb=4_000, available=True)
        estimate = classify_vram(10_000, gpu)
        self.assertAlmostEqual(estimate.percent_available, 10_000 / 24_000 * 100)
        self.assertEqual(estimate.status, "comfortable")

    def test_classification_ignores_current_occupancy_on_unified(self) -> None:
        gpu = GPUStats(
            vram_total_mb=16_456,
            vram_used_mb=10_380,
            available=True,
            unified=True,
        )
        estimate = classify_vram(10_700, gpu)
        self.assertLess(estimate.percent_available, 80)
        self.assertEqual(estimate.status, "comfortable")

    def test_vram_scales_down_with_partial_offload(self) -> None:
        full = estimate_vram_mb(10_000_000_000, 65_536, 1.0)
        half = estimate_vram_mb(10_000_000_000, 65_536, 0.5)
        none = estimate_vram_mb(10_000_000_000, 65_536, 0.0)
        self.assertGreater(full, half)
        self.assertEqual(none, 0.0)

    def test_bandwidth_matches_known_gpu_name(self) -> None:
        gpu = GPUStats(name="AMD Radeon RX 7900 XTX", vram_total_mb=24_000, available=True)
        self.assertEqual(gpu_bandwidth_gb_s(gpu), 960.0)

    def test_full_offload_faster_than_cpu(self) -> None:
        gpu = GPUStats(
            name="NVIDIA GeForce RTX 4090",
            vram_total_mb=24_000,
            available=True,
        )
        full = estimate_gen_tps(10_000_000_000, 65_536, gpu, 1.0)
        half = estimate_gen_tps(10_000_000_000, 65_536, gpu, 0.5)
        cpu = estimate_gen_tps(10_000_000_000, 65_536, gpu, 0.0)
        self.assertIsNotNone(full)
        self.assertIsNotNone(half)
        self.assertIsNotNone(cpu)
        self.assertGreater(full, half)
        self.assertGreater(half, cpu)

    def test_longer_context_is_slower(self) -> None:
        gpu = GPUStats(name="NVIDIA GeForce RTX 4090", vram_total_mb=24_000, available=True)
        at_64k = estimate_gen_tps(10_000_000_000, 65_536, gpu, 1.0)
        at_256k = estimate_gen_tps(10_000_000_000, 262_144, gpu, 1.0)
        self.assertGreater(at_64k, at_256k)

    def test_fmt_tps_uses_tilde(self) -> None:
        self.assertEqual(fmt_tps(42.2), "~42 t/s")
        self.assertEqual(fmt_tps(None), "?")


if __name__ == "__main__":
    unittest.main()
