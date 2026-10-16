"""Vision / tool-call capability detection and mmproj selection."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from tui.data.capabilities import (
    capabilities_from_files,
    capabilities_from_hf,
    is_vision_enabled,
    mmproj_filename_for,
    pick_mmproj_filename,
    resolve_capabilities,
    vision_projector_ready,
)
from tui.data.hf import HubFile, HubRepo, build_download_plan
from tui.data.models_json import merge_repo_catalog
from tui.data.quant import is_mmproj_gguf, is_sidecar_gguf
from tui.data.quant_table import build_quant_file_rows
from tui.data.gpu import GPUStats
from tui.launch import build_server_args


class CapabilityDetectionTests(unittest.TestCase):
    def test_hf_pipeline_marks_vision(self) -> None:
        caps = capabilities_from_hf(pipeline_tag="image-text-to-text", tags=["gguf"])
        self.assertTrue(caps.vision)
        self.assertEqual(caps.vision_source, "hf")
        self.assertFalse(caps.tools)

    def test_hf_tags_mark_tools(self) -> None:
        caps = capabilities_from_hf(tags=["gguf", "function-calling"])
        self.assertTrue(caps.tools)
        self.assertEqual(caps.tools_source, "hf")

    def test_file_fallback_when_hf_is_silent(self) -> None:
        files = [
            HubFile(path="Qwen3.8-27B-Q3_K_S.gguf", size=10),
            HubFile(path="mmproj-Qwen3.8-27B-f16.gguf", size=2),
        ]
        caps = resolve_capabilities(
            hf=capabilities_from_hf(tags=["gguf"]),
            files=files,
            repo_id="bartowski/Qwen3.8-27B-GGUF",
        )
        self.assertTrue(caps.vision)
        self.assertEqual(caps.vision_source, "files")
        self.assertTrue(caps.tools)
        self.assertEqual(caps.tools_source, "family")

    def test_hf_wins_over_fallback_source_label(self) -> None:
        files = [HubFile(path="mmproj-model-f16.gguf", size=1)]
        caps = resolve_capabilities(
            hf=capabilities_from_hf(pipeline_tag="image-text-to-text"),
            files=files,
        )
        self.assertTrue(caps.vision)
        self.assertEqual(caps.vision_source, "hf")


class MmprojPickTests(unittest.TestCase):
    def test_prefers_f16_projector_for_family(self) -> None:
        files = [
            "mmproj-Qwen3.8-27B-q8_0.gguf",
            "mmproj-Qwen3.8-27B-f16.gguf",
            "mmproj-Other-bf16.gguf",
        ]
        picked = pick_mmproj_filename(files, quant_filename="Qwen3.8-27B-Q3_K_S.gguf")
        self.assertEqual(picked, "mmproj-Qwen3.8-27B-f16.gguf")

    def test_sidecar_helpers(self) -> None:
        self.assertTrue(is_mmproj_gguf("mmproj-Qwen3.8-27B-f16.gguf"))
        self.assertTrue(is_sidecar_gguf("mmproj-Qwen3.8-27B-f16.gguf"))
        self.assertTrue(is_sidecar_gguf("Qwen3.8-27B-imatrix.gguf"))
        self.assertFalse(is_sidecar_gguf("Qwen3.8-27B-Q3_K_S.gguf"))


class MergeCatalogSidecarTests(unittest.TestCase):
    def test_sidecars_are_stripped_from_quants(self) -> None:
        params = {
            "source": {"author": "bartowski"},
            "active_quant": "BF16",
            "quants": {
                "BF16": {
                    "filename": "mmproj-Qwen3.8-27B-bf16.gguf",
                    "file": "bartowski/mmproj-Qwen3.8-27B-bf16.gguf",
                },
                "Q3_K_S": {
                    "filename": "Qwen3.8-27B-Q3_K_S.gguf",
                    "file": "bartowski/Qwen3.8-27B-Q3_K_S.gguf",
                },
            },
        }
        merge_repo_catalog(
            params,
            [
                ("Qwen3.8-27B-Q3_K_S.gguf", "10"),
                ("mmproj-Qwen3.8-27B-f16.gguf", "2"),
            ],
        )
        self.assertEqual(list(params["quants"]), ["Q3_K_S"])
        self.assertEqual(params["active_quant"], "Q3_K_S")


class QuantTableSidecarTests(unittest.TestCase):
    def test_sidecar_files_are_not_quant_rows(self) -> None:
        rows = build_quant_file_rows(
            [
                HubFile(path="Qwen3.8-27B-Q3_K_S.gguf", size=1000),
                HubFile(path="mmproj-Qwen3.8-27B-f16.gguf", size=200),
                HubFile(path="Qwen3.8-27B-imatrix.gguf", size=50),
            ],
            gpu=GPUStats(),
            context_tokens=65_536,
            offload_ratio=1.0,
            baselines_path=None,
        )
        self.assertEqual([row.path for row in rows], ["Qwen3.8-27B-Q3_K_S.gguf"])


class DownloadPlanExtraTests(unittest.TestCase):
    def test_extra_mmproj_is_included_once(self) -> None:
        plan = build_download_plan(
            "bartowski/Qwen3.8-27B-GGUF",
            "Qwen3.8-27B-Q3_K_S.gguf",
            Path("/models"),
            extra_files=["mmproj-Qwen3.8-27B-f16.gguf", "Qwen3.8-27B-Q3_K_S.gguf"],
        )
        self.assertEqual(
            plan.filenames,
            ["Qwen3.8-27B-Q3_K_S.gguf", "mmproj-Qwen3.8-27B-f16.gguf"],
        )


class VisionToggleTests(unittest.TestCase):
    def test_defaults_on(self) -> None:
        self.assertTrue(is_vision_enabled({}))
        self.assertTrue(is_vision_enabled({"vision_enabled": True}))
        self.assertFalse(is_vision_enabled({"vision_enabled": False}))
        self.assertFalse(is_vision_enabled({"vision_enabled": "off"}))

    def test_recorded_mmproj_name_wins(self) -> None:
        params = {"mmproj": "bartowski/mmproj-Qwen3.8-27B-bf16.gguf"}
        self.assertEqual(
            mmproj_filename_for(params, ["mmproj-Qwen3.8-27B-f16.gguf"]),
            "mmproj-Qwen3.8-27B-bf16.gguf",
        )

    def test_ready_when_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            path = models_dir / "bartowski" / "mmproj-Qwen3.8-27B-f16.gguf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"gguf")
            params = {"mmproj": "bartowski/mmproj-Qwen3.8-27B-f16.gguf"}
            name, ready = vision_projector_ready(params, models_dir)
            self.assertEqual(name, "mmproj-Qwen3.8-27B-f16.gguf")
            self.assertTrue(ready)

    def test_missing_projector_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            params = {"mmproj": "bartowski/mmproj-Qwen3.8-27B-f16.gguf"}
            name, ready = vision_projector_ready(params, Path(tmp))
            self.assertEqual(name, "mmproj-Qwen3.8-27B-f16.gguf")
            self.assertFalse(ready)

    def test_identity_keeps_vision_enabled(self) -> None:
        from tui.data.preset_template import identity_from_seed

        identity = identity_from_seed(
            {"display": "Qwen", "vision_enabled": False, "gpu_layers": 99}
        )
        self.assertFalse(identity["vision_enabled"])
        self.assertNotIn("gpu_layers", identity)


class LaunchMmprojTests(unittest.TestCase):
    def test_mmproj_flag_is_passed(self) -> None:
        from tui.data.preset_template import DEFAULT_PRESET_PARAMS

        args = build_server_args(
            model_path=Path("/tmp/m.gguf"),
            host="127.0.0.1",
            port=8081,
            params=DEFAULT_PRESET_PARAMS,
            log_verbosity=4,
            mmproj_path=Path("/tmp/mmproj-f16.gguf"),
        )
        self.assertEqual(args[0:8], [
            "-m",
            "/tmp/m.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
            "--mmproj",
            "/tmp/mmproj-f16.gguf",
        ])

    def test_prepare_launch_omits_mmproj_when_vision_off(self) -> None:
        from tests.support import MODEL_SLUG, Harness
        from tui.launch import prepare_launch

        harness = Harness()
        try:
            models_dir = harness.paths.models_dir
            mmproj = models_dir / "demo" / "mmproj-Demo-f16.gguf"
            mmproj.write_bytes(b"gguf")
            app = harness.app()
            cfg = app.registry.models[MODEL_SLUG]
            cfg.params["mmproj"] = "demo/mmproj-Demo-f16.gguf"
            cfg.params["vision_enabled"] = False
            from tui.data.models_json import save_registry

            save_registry(harness.paths.models_json, app.registry)
            plan = prepare_launch(MODEL_SLUG, paths=harness.paths)
            self.assertNotIn("--mmproj", plan.args)
        finally:
            harness.cleanup()

    def test_prepare_launch_passes_mmproj_when_vision_on(self) -> None:
        from tests.support import MODEL_SLUG, Harness
        from tui.launch import prepare_launch

        harness = Harness()
        try:
            models_dir = harness.paths.models_dir
            mmproj = models_dir / "demo" / "mmproj-Demo-f16.gguf"
            mmproj.write_bytes(b"gguf")
            app = harness.app()
            cfg = app.registry.models[MODEL_SLUG]
            cfg.params["mmproj"] = "demo/mmproj-Demo-f16.gguf"
            cfg.params["vision_enabled"] = True
            from tui.data.models_json import save_registry

            save_registry(harness.paths.models_json, app.registry)
            plan = prepare_launch(MODEL_SLUG, paths=harness.paths)
            self.assertIn("--mmproj", plan.args)
            self.assertEqual(plan.args[plan.args.index("--mmproj") + 1], str(mmproj))
        finally:
            harness.cleanup()


class HubRepoTagParseTests(unittest.TestCase):
    def test_from_json_keeps_tags_and_pipeline(self) -> None:
        repo = HubRepo.from_json(
            {
                "id": "bartowski/Qwen3.8-27B-GGUF",
                "downloads": 1,
                "likes": 2,
                "tags": ["gguf", "image-text-to-text"],
                "pipeline_tag": "image-text-to-text",
                "gguf": {"total": 10, "context_length": 32768},
            }
        )
        self.assertIn("image-text-to-text", repo.tags)
        self.assertEqual(repo.pipeline_tag, "image-text-to-text")
        self.assertTrue(capabilities_from_files([], repo_id=repo.id).tools)


if __name__ == "__main__":
    unittest.main()
