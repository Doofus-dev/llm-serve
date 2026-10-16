"""Compare Presets page (key x): idle/active metrics, caveat, notify timeout."""

from __future__ import annotations

import inspect
import os
import unittest

from tests.support import MODEL_SLUG, OTHER_SLUG, QUANT, Harness
from tui.app import LLMServeApp
from tui.data.baselines import RunBaseline, save_baselines
from tui.data.compare_metrics import (
    compare_presets_metrics,
    format_metric_cell,
    resolve_baseline,
    resolve_default_preset,
)
from tui.data.pidfile import write_pid_file
from tui.data.preset_template import DEFAULT_PRESET_PARAMS
from tui.data.presets import get_preset, load_presets, save_presets, set_preset
from tui.screens.compare import format_diff_summary
from tui.screens.preset_compare_page import PresetComparePage
from tui.theme import ERR, OK
from tui.widgets.nav import ModelNav


def _run(**overrides) -> RunBaseline:
    values = dict(
        model=MODEL_SLUG,
        file="demo/Demo-Q4_K_M.gguf",
        file_size=5_000_000,
        gpu_name="Radeon RX 7900 XTX",
        ctx=32_768,
        gpu_layers=99,
        total_layers=32,
        cache_k="q4_0",
        cache_v="q4_0",
        vram_used_mb=4000.0,
        gen_tps=40.0,
        prompt_tps=200.0,
        tokens_predicted=400.0,
    )
    values.update(overrides)
    return RunBaseline(**values)


class CompareMetricsUnitTests(unittest.TestCase):
    def test_winners_follow_direction(self) -> None:
        left = _run(vram_used_mb=8000, gen_tps=25, prompt_tps=100, ctx=16384, file_size=5_000_000)
        right = _run(vram_used_mb=4000, gen_tps=40, prompt_tps=200, ctx=32768, file_size=5_000_000)
        rows = {row["name"]: row for row in compare_presets_metrics(left, right)}
        self.assertEqual(rows["VRAM used (MB)"]["winner"], "right")
        self.assertEqual(rows["gen t/s"]["winner"], "right")
        self.assertEqual(rows["prompt t/s"]["winner"], "right")
        self.assertEqual(rows["file size"]["winner"], "equal")
        self.assertEqual(rows["context"]["winner"], "right")
        vram_left = format_metric_cell(
            8000, winner="right", side="left", fmt="mb"
        )
        vram_right = format_metric_cell(
            4000, winner="right", side="right", fmt="mb"
        )
        self.assertEqual(vram_left, f"[{ERR}]8000 MB[/]")
        self.assertEqual(vram_right, f"[{OK}]4000 MB[/]")

    def test_missing_values_render_no_data(self) -> None:
        rows = compare_presets_metrics(None, _run())
        vram = next(row for row in rows if row["name"] == "VRAM used (MB)")
        self.assertIsNone(vram["left_value"])
        self.assertEqual(vram["winner"], None)
        self.assertEqual(
            format_metric_cell(None, winner=None, side="left", fmt="mb"),
            "[dim]no data[/]",
        )

    def test_resolve_baseline_matches_ctx_not_sibling(self) -> None:
        runs = [
            _run(ctx=32768, vram_used_mb=4000, gen_tps=40),
            _run(ctx=16384, vram_used_mb=8000, gen_tps=25, prompt_tps=100),
        ]
        identity = {"file": "demo/Demo-Q4_K_M.gguf", "quants": {
            QUANT: {"file": "demo/Demo-Q4_K_M.gguf"}
        }}
        from tui.data.presets import Preset

        matched = resolve_baseline(
            MODEL_SLUG,
            QUANT,
            2,
            None,
            runs,
            identity=identity,
            preset=Preset(slot=2, name="max-ctx", params={"ctx": 16384, "gpu_layers": 99}),
        )
        self.assertIsNotNone(matched)
        self.assertAlmostEqual(matched.vram_used_mb, 8000.0)
        missing = resolve_baseline(
            MODEL_SLUG,
            QUANT,
            3,
            None,
            runs,
            identity=identity,
            preset=Preset(slot=3, name="odd", params={"ctx": 4096, "gpu_layers": 99}),
        )
        self.assertTrue(missing is None or missing.vram_used_mb == 0)


class PresetComparePageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.harness = Harness()
        store = load_presets(self.harness.paths.presets_json)
        custom = dict(DEFAULT_PRESET_PARAMS)
        custom["ctx"] = 16384
        custom["gpu_layers"] = 40
        set_preset(store, MODEL_SLUG, QUANT, 2, "max-ctx", custom)
        other_custom = dict(DEFAULT_PRESET_PARAMS)
        other_custom["ctx"] = 8192
        set_preset(store, OTHER_SLUG, QUANT, 2, "small", other_custom)
        save_presets(self.harness.paths.presets_json, store)
        save_baselines(
            self.harness.paths.baselines_json,
            [
                _run(ctx=32768, vram_used_mb=4000, gen_tps=40, prompt_tps=200),
                _run(ctx=16384, vram_used_mb=8000, gen_tps=25, prompt_tps=100, gpu_layers=40),
                _run(
                    model=OTHER_SLUG,
                    ctx=8192,
                    vram_used_mb=2000,
                    gen_tps=55,
                    prompt_tps=300,
                ),
            ],
        )

    def tearDown(self) -> None:
        self.harness.cleanup()

    def _select_preset(self, app: LLMServeApp, model: str, slot: int) -> None:
        nav = app.query_one(ModelNav)
        nav.focus()
        for index in range(nav.option_count):
            option = nav.get_option_at_index(index)
            data = nav._option_data.get(option.id or "")
            if data and data[0] == "preset" and data[1] == model and data[3] == slot:
                nav.highlighted = index
                return
        self.fail(f"preset {model}[{slot}] not in nav")

    def _metrics(self, screen) -> str:
        return str(screen.query_one("#pcp-metrics").content)

    async def test_idle_compares_selected_against_default(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            self._select_preset(app, MODEL_SLUG, 2)
            app.action_compare_selected()
            await pilot.pause(0.4)
            screen = app.screen
            self.assertIsInstance(screen, PresetComparePage)
            self.assertEqual(screen.left.slot, 2)
            self.assertEqual(screen.left.role, "selected")
            self.assertEqual(screen.right.role, "default")
            self.assertEqual(screen.right.slot, 1)
            metrics = self._metrics(screen)
            self.assertIn("VRAM used (MB)", metrics)
            self.assertIn(f"[{ERR}]8000 MB[/]", metrics)
            self.assertIn(f"[{OK}]4000 MB[/]", metrics)
            self.assertIn(f"[{ERR}]25 t/s[/]", metrics)
            self.assertIn(f"[{OK}]40 t/s[/]", metrics)
            self.assertNotIn("different models", str(screen.query_one("#pcp-status").content))

    async def test_active_compares_selected_against_running(self) -> None:
        write_pid_file(
            self.harness.paths.pid_file,
            pid=os.getpid(),
            model=MODEL_SLUG,
            port=8081,
            quant=QUANT,
            preset_slot=1,
            remote=False,
        )
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app._refresh_pid()
            self._select_preset(app, MODEL_SLUG, 2)
            app.action_compare_selected()
            await pilot.pause(0.4)
            screen = app.screen
            self.assertIsInstance(screen, PresetComparePage)
            self.assertEqual(screen.left.role, "selected")
            self.assertEqual(screen.right.role, "running")
            self.assertEqual(screen.right.slot, 1)
            metrics = self._metrics(screen)
            self.assertIn(f"[{ERR}]8000 MB[/]", metrics)
            self.assertIn(f"[{OK}]4000 MB[/]", metrics)

    def _config_grid(self, screen) -> str:
        return str(screen.query_one("#pcp-config-grid").content)

    async def test_column_grid_pairs_values(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            self._select_preset(app, MODEL_SLUG, 2)
            app.action_compare_selected()
            await pilot.pause(0.4)
            screen = app.screen
            self.assertIsInstance(screen, PresetComparePage)
            grid = self._config_grid(screen)
            self.assertIn("Model", grid)
            self.assertIn("Quant", grid)
            self.assertIn("Preset", grid)
            self.assertIn("Max context", grid)
            # Wide layout (width >= 100) puts identity and runtime side by side
            # without a "RUNTIME CONFIG" header; narrow layout has one.
            self.assertIn("ctx", grid)
            self.assertIn("gpu_layers", grid)

    async def test_arrow_keys_cycle_right_preset(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            self._select_preset(app, MODEL_SLUG, 2)
            app.action_compare_selected()
            await pilot.pause(0.4)
            screen = app.screen
            self.assertIsInstance(screen, PresetComparePage)
            self.assertEqual(screen.right.slot, 1)
            await pilot.press("down")
            await pilot.pause(0.2)
            self.assertEqual(screen.right.slot, 2)
            await pilot.press("down")
            await pilot.pause(0.2)
            self.assertEqual(screen.right.slot, 1)
            await pilot.press("up")
            await pilot.pause(0.2)
            self.assertEqual(screen.right.slot, 2)
            grid = self._config_grid(screen)
            self.assertIn("max-ctx", grid)

    def test_diff_summary_colors_changed_values_by_direction(self) -> None:
        summary = format_diff_summary(
            "a",
            "b",
            {"ctx": 8192, "gpu_layers": 99},
            {"ctx": 16384, "gpu_layers": 99},
        )
        self.assertIn(f"ctx  [{ERR}]8192[/] → [{OK}]16384[/]", summary)
        self.assertNotIn("gpu_layers", summary)

    async def test_cross_model_shows_caveat(self) -> None:
        write_pid_file(
            self.harness.paths.pid_file,
            pid=os.getpid(),
            model=MODEL_SLUG,
            port=8081,
            quant=QUANT,
            preset_slot=1,
            remote=False,
        )
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            app._refresh_pid()
            self._select_preset(app, OTHER_SLUG, 2)
            app.action_compare_selected()
            await pilot.pause(0.4)
            screen = app.screen
            self.assertIsInstance(screen, PresetComparePage)
            self.assertNotEqual(screen.left.model, screen.right.model)
            status = str(screen.query_one("#pcp-status").content)
            self.assertIn("different models — metrics not directly comparable", status)

    async def test_missing_baseline_renders_no_data(self) -> None:
        store = load_presets(self.harness.paths.presets_json)
        params = dict(DEFAULT_PRESET_PARAMS)
        params["ctx"] = 4096
        set_preset(store, MODEL_SLUG, QUANT, 3, "unmeasured", params)
        save_presets(self.harness.paths.presets_json, store)
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            self._select_preset(app, MODEL_SLUG, 3)
            app.action_compare_selected()
            await pilot.pause(0.4)
            screen = app.screen
            self.assertIsInstance(screen, PresetComparePage)
            metrics = self._metrics(screen)
            self.assertIn("[dim]no data[/]", metrics)

    async def test_editor_mode_blocks_compare(self) -> None:
        app = self.harness.app()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.4)
            self._select_preset(app, MODEL_SLUG, 2)
            app._editor_mode = True
            app.action_compare_selected()
            await pilot.pause(0.2)
            self.assertNotIsInstance(app.screen, PresetComparePage)

    def test_notify_default_timeout_is_four_seconds(self) -> None:
        timeout = inspect.signature(LLMServeApp.notify).parameters["timeout"].default
        self.assertEqual(timeout, 4.0)

    def test_default_preset_is_seeded_name(self) -> None:
        store = load_presets(self.harness.paths.presets_json)
        default = resolve_default_preset(store, MODEL_SLUG, QUANT)
        self.assertIsNotNone(default)
        self.assertEqual(default.name, "default")
        self.assertEqual(default.slot, 1)
        self.assertEqual(
            get_preset(store, MODEL_SLUG, QUANT, 1).name, default.name
        )


if __name__ == "__main__":
    unittest.main()
