"""Tests for the compact main-page dashboard renderers."""

from __future__ import annotations

import io
import os
import subprocess
import unittest

from rich.console import Console

from tui.widgets.config import ConfigPanel
from tui.widgets.health import generation_health, temperature_health, vram_health
from tui.widgets.status import (
    StatusHeader,
    StatusPanel,
    format_generation_speed,
    format_instance_preset_column,
    format_preset_display,
)
from tui.data.throughput_history import LastRequest, LiveThroughput, SPARKLINE_WIDTH
from tui.data.gpu import GPUStats, ProcessMem
from tui.data.pidfile import PidInfo, write_pid_file
from tui.data.stats import Metrics


def render_text(renderable) -> str:
    output = io.StringIO()
    Console(file=output, width=100, color_system=None).print(renderable)
    return output.getvalue()


class DashboardTests(unittest.TestCase):
    def test_config_labels_and_values_use_distinct_styles(self) -> None:
        self.assertEqual(str(ConfigPanel._label("ctx").style), "cyan")
        self.assertEqual(str(ConfigPanel._value(40960).style), "bold yellow")

    def test_status_panel_renders_populated_metrics(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics(
            prompt_tokens_total=59_916,
            tokens_predicted_total=3_732,
            prompt_tokens_seconds=120.5,
            predicted_tokens_seconds=49.5,
            requests_processing=1,
        )
        panel.live_throughput = LiveThroughput(
            gen_tps=49.5,
            prompt_tps=120.5,
            source="metrics_gauge",
            stage="generating",
        )

        rendered = render_text(panel.render())

        self.assertIn("GENERATE", rendered)
        self.assertIn("49.5 t/s FAST", rendered)
        self.assertIn("20.2 ms/token", rendered)
        self.assertNotIn("t/s generation", rendered)
        self.assertNotIn("59,916 prompt tokens", rendered)

    def test_generation_speed_fits_sparkline_width(self) -> None:
        cases = (
            (10.0, "MODERATE", "100.0"),
            (100.0, "FAST", "10.0"),
            (49.5, "FAST", "20.2"),
        )
        for tps, label, ms_token in cases:
            with self.subTest(tps=tps):
                line = format_generation_speed(tps)
                self.assertLessEqual(line.cell_len, SPARKLINE_WIDTH)
                self.assertTrue(line.no_wrap)
                output = io.StringIO()
                Console(file=output, width=SPARKLINE_WIDTH, color_system=None).print(line)
                rendered = output.getvalue()
                self.assertEqual(rendered.count("\n"), 1)
                self.assertIn(f"{tps:.1f} t/s {label}", rendered)
                self.assertIn(f"{ms_token} ms/token", rendered)
                self.assertNotIn("generation", rendered)

    def test_status_panel_shows_prefill_phase(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics(requests_processing=1, prompt_tokens_seconds=80.0)
        panel.live_throughput = LiveThroughput(
            gen_tps=0.0,
            prompt_tps=3800.0,
            source="slots",
            stage="prefill",
            n_prompt_processed=2431,
            n_prompt_total=4096,
            n_prompt_cache=1200,
        )

        rendered = render_text(panel.render())

        self.assertIn("PREFILL", rendered)
        self.assertIn("CACHE", rendered)
        self.assertIn("3,631/4,096", rendered)
        self.assertIn("waiting on generate", rendered)
        self.assertNotIn("3800", rendered)

    def test_status_panel_shows_last_request_when_idle(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics()
        panel.live_throughput = LiveThroughput(
            stage="idle",
            last_request=LastRequest(
                prompt_tokens=4096,
                cache_tokens=1200,
                gen_tokens=142,
                gen_tps=41.0,
            ),
        )

        rendered = render_text(panel.render())
        collapsed = " ".join(rendered.split())

        self.assertIn("IDLE queue cache prefill generate", collapsed)
        self.assertIn("last: 4,096 prompt (1,200 cache) · 142 gen @ 41 t/s", collapsed)

    def test_status_panel_prefers_friendly_model_name(self) -> None:
        panel = StatusPanel()
        panel.pid_info = PidInfo(
            pid=os.getpid(),
            model="qwen36-27b-bartowski",
            port=8080,
            ts="",
            remote=True,
        )
        panel.model_display = "Qwen 3.6"
        panel.quant_display = "Q8_0"
        panel.preset_display = "[1] default"

        rendered = render_text(panel.render())

        self.assertIn("RUNNING  Qwen 3.6  Q8_0  [1] default  REMOTE", rendered)
        self.assertNotIn("qwen36-27b-bartowski", rendered)

    def test_status_panel_hot_bar_shows_family_quant_and_preset_name(self) -> None:
        panel = StatusPanel()
        panel.pid_info = PidInfo(
            pid=os.getpid(),
            model="qwen38-27b-bartowski",
            port=8080,
            ts="",
        )
        panel.model_display = "Qwen 3.8"
        panel.quant_display = "Q3_K_S"
        panel.preset_display = "[5] 120K-HighPower"
        panel.uptime = 65
        panel.props = {
            "model_alias": "coding",
            "model_path": "/home/doofus/models/Qwen3.8-27B-Q3_K_S.gguf",
        }

        rendered = render_text(panel.render())

        self.assertIn("RUNNING  Qwen 3.8  Q3_K_S  [5] 120K-HighPower", rendered)
        self.assertIn("Qwen 3.8  •  Q3_K_S  •  [5] 120K-HighPower", rendered)
        self.assertNotIn("slot 5", rendered)
        self.assertNotIn("alias coding", rendered)
        self.assertNotIn("Qwen3.8-27B-Q3_K_S.gguf", rendered)
        self.assertNotIn("port 8080", rendered)
        self.assertNotIn(f"PID {os.getpid()}", rendered)
        self.assertNotIn("up 0:01:05", rendered)

    def test_format_preset_display_includes_name(self) -> None:
        self.assertEqual(format_preset_display(5, "120K-HighPower"), "[5] 120K-HighPower")
        self.assertEqual(format_preset_display(1, "  "), "[1]")
        self.assertIsNone(format_preset_display(None, "default"))
        self.assertEqual(
            format_instance_preset_column("Q3_K_S", 5, "120K-HighPower"),
            "Q3_K_S [5] 120K-HighPower",
        )
        self.assertEqual(format_instance_preset_column("Q8_0", 2, None), "Q8_0 [2]")

    def test_status_panel_multi_instance_preset_column_includes_name(self) -> None:
        panel = StatusPanel()
        self_pid = os.getpid()
        child = subprocess.Popen(["sleep", "30"])
        try:
            primary = PidInfo(
                pid=self_pid,
                model="qwen38-27b-bartowski",
                port=8080,
                ts="",
                quant="Q3_K_S",
                preset_slot=5,
            )
            extra = PidInfo(
                pid=child.pid,
                model="other-model",
                port=8081,
                ts="",
                quant="Q8_0",
                preset_slot=1,
            )
            panel.pid_info = primary
            panel.instances = [primary, extra]
            panel.instance_labels = {
                primary.pid: "Qwen 3.8",
                extra.pid: "Other",
            }
            panel.instance_preset_names = {
                primary.pid: "120K-HighPower",
                extra.pid: "default",
            }
            panel.model_display = "Qwen 3.8"
            panel.quant_display = "Q3_K_S"
            panel.preset_display = "[5] 120K-HighPower"

            rendered = render_text(panel.render())
            collapsed = " ".join(rendered.split())

            self.assertIn("Q3_K_S [5] 120K-HighPower", collapsed)
            self.assertIn("Q8_0 [1] default", collapsed)
            self.assertIn("PRESET", collapsed)
        finally:
            child.terminate()
            child.wait(timeout=5)

    def test_status_header_shows_live_status_or_not_running(self) -> None:
        header = StatusHeader()
        self.assertIn("not running", render_text(header.render()))

        header.pid_info = PidInfo(
            pid=os.getpid(),
            model="qwen38-27b-bartowski",
            port=8080,
            ts="",
        )
        header.uptime = 65
        rendered = render_text(header.render())
        self.assertIn(f"port 8080  •  PID {os.getpid()}  •  up 0:01:05", rendered)

    def test_status_health_thresholds(self) -> None:
        self.assertEqual(vram_health(74.9), ("OK", "bold green"))
        self.assertEqual(vram_health(75), ("HIGH", "bold yellow"))
        self.assertEqual(vram_health(90), ("CRITICAL", "bold red"))
        self.assertEqual(temperature_health(69.9), ("COOL", "bold green"))
        self.assertEqual(temperature_health(70), ("WARM", "bold yellow"))
        self.assertEqual(temperature_health(85), ("HOT", "bold red"))
        self.assertEqual(generation_health(20), ("FAST", "bold green"))
        self.assertEqual(generation_health(5), ("MODERATE", "bold yellow"))
        self.assertEqual(generation_health(1), ("SLOW", "bold red"))

    def test_status_panel_renders_gpu_pressure_indicators(self) -> None:
        panel = StatusPanel()
        panel.gpu = GPUStats(
            name="Test GPU",
            vram_used_mb=15_360,
            vram_total_mb=16_384,
            utilization_pct=96,
            temp_c=87,
            available=True,
        )

        rendered = render_text(panel.render())

        self.assertIn("CRITICAL", rendered)
        self.assertIn("87°C HOT", rendered)

    def test_status_panel_renders_throughput_graph(self) -> None:
        panel = StatusPanel()
        panel.metrics = Metrics(predicted_tokens_seconds=25.0)
        panel.live_throughput = LiveThroughput(gen_tps=25.0, source="metrics_gauge")
        panel.gen_tps_history = [10.0, 20.0, 25.0, 30.0]

        rendered = render_text(panel.render())

        self.assertIn("avg 21.2 tok/s", rendered)
        self.assertTrue(any(ch in rendered for ch in "▁▂▃▄▅▆▇█"))


    def test_status_panel_renders_multiple_gpu_processes(self) -> None:
        panel = StatusPanel()
        panel.gpu = GPUStats(
            name="Test GPU",
            vram_used_mb=12_288,
            vram_total_mb=16_384,
            utilization_pct=40,
            temp_c=55,
            available=True,
            processes=[
                ProcessMem(pid=100, name="llama-server", vram_mb=8192, ram_mb=1500, tracked=True),
                ProcessMem(pid=200, name="other-llama", vram_mb=4096, ram_mb=900, tracked=False),
            ],
        )

        rendered = render_text(panel.render())
        collapsed = " ".join(rendered.split())

        self.assertIn("12.0G / 16.0G total", collapsed)
        self.assertIn("llama-server", collapsed)
        self.assertIn("other-llama", collapsed)
        self.assertIn("100", collapsed)
        self.assertIn("200", collapsed)
        self.assertIn("8.0G", collapsed)
        self.assertIn("4.0G", collapsed)
        self.assertIn("VRAM", collapsed)
        self.assertIn("RAM", collapsed)

    def test_status_panel_shows_full_seven_digit_pids(self) -> None:
        panel = StatusPanel()
        panel.gpu = GPUStats(
            name="Test GPU",
            vram_used_mb=12_288,
            vram_total_mb=16_384,
            utilization_pct=40,
            temp_c=55,
            available=True,
            processes=[
                ProcessMem(
                    pid=1120647,
                    name="llama-server",
                    vram_mb=13902,
                    ram_mb=1251,
                    tracked=True,
                ),
                ProcessMem(
                    pid=4155563,
                    name="llama-server",
                    vram_mb=0,
                    ram_mb=2.5,
                    tracked=False,
                ),
            ],
        )

        rendered = render_text(panel.render())

        self.assertIn("1120647", rendered)
        self.assertIn("4155563", rendered)
        self.assertNotIn("11206…", rendered)
        self.assertNotIn("41555…", rendered)

    def test_status_panel_shows_next_launch_remote_toggle(self) -> None:
        panel = StatusPanel()
        self.assertIn("NEXT LAUNCH [LOCAL]  [LOG TRACE]", render_text(panel.render()))

        panel.next_remote = True
        panel.next_log_verbosity = 3

        rendered = render_text(panel.render())
        self.assertIn("NEXT LAUNCH [REMOTE]", rendered)
        self.assertIn("[LOG INFO]", rendered)


class StatusHeaderComposeTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_top_bar_is_live_status_not_title(self) -> None:
        from textual.widgets import Header

        from tests.support import Harness

        harness = Harness()
        try:
            app = harness.app()
            async with app.run_test(size=(120, 40)):
                self.assertEqual(len(list(app.query(Header))), 0)
                header = app.query_one(StatusHeader)
                self.assertIn("not running", render_text(header.render()))
        finally:
            harness.cleanup()

    async def test_header_stays_live_in_editor_mode(self) -> None:
        from tests.support import MODEL_SLUG, QUANT, Harness

        harness = Harness()
        try:
            app = harness.app()
            async with app.run_test(size=(120, 40)):
                write_pid_file(
                    harness.paths.pid_file,
                    pid=os.getpid(),
                    model=MODEL_SLUG,
                    port=8080,
                    quant=QUANT,
                    preset_slot=1,
                    remote=False,
                )
                app._editor_mode = True
                app._refresh_pid()
                header = app.query_one(StatusHeader)
                self.assertIn(
                    f"port 8080  •  PID {os.getpid()}",
                    render_text(header.render()),
                )

                harness.paths.pid_file.unlink()
                app._refresh_pid()
                self.assertIn("not running", render_text(header.render()))
        finally:
            harness.cleanup()

    async def test_refresh_pid_shows_quant_and_preset_name(self) -> None:
        from tests.support import DISPLAY, MODEL_SLUG, QUANT, Harness
        from tui.data.presets import get_preset, save_presets

        harness = Harness()
        try:
            app = harness.app()
            async with app.run_test(size=(120, 40)):
                preset = get_preset(app.preset_store, MODEL_SLUG, QUANT, 1)
                self.assertIsNotNone(preset)
                preset.name = "120K-HighPower"
                save_presets(harness.paths.presets_json, app.preset_store)
                write_pid_file(
                    harness.paths.pid_file,
                    pid=os.getpid(),
                    model=MODEL_SLUG,
                    port=8080,
                    quant=QUANT,
                    preset_slot=1,
                    remote=False,
                )
                app._refresh_pid()
                panel = app.query_one(StatusPanel)
                self.assertEqual(panel.quant_display, QUANT)
                self.assertEqual(panel.preset_display, "[1] 120K-HighPower")
                rendered = render_text(panel.render())
                self.assertIn(f"RUNNING  {DISPLAY}  {QUANT}  [1] 120K-HighPower", rendered)
                self.assertIn(f"{DISPLAY}  •  {QUANT}  •  [1] 120K-HighPower", rendered)
        finally:
            harness.cleanup()

    async def test_refresh_pid_multi_instance_preset_column_includes_name(self) -> None:
        from tests.support import MODEL_SLUG, OTHER_DISPLAY, OTHER_SLUG, QUANT, Harness
        from tui.data.pidfile import record_instance
        from tui.data.presets import get_preset, save_presets

        harness = Harness()
        child = subprocess.Popen(["sleep", "30"])
        try:
            app = harness.app()
            async with app.run_test(size=(120, 40)):
                primary_preset = get_preset(app.preset_store, MODEL_SLUG, QUANT, 1)
                other_preset = get_preset(app.preset_store, OTHER_SLUG, QUANT, 1)
                self.assertIsNotNone(primary_preset)
                self.assertIsNotNone(other_preset)
                primary_preset.name = "120K-HighPower"
                other_preset.name = "coding"
                save_presets(harness.paths.presets_json, app.preset_store)
                record_instance(
                    pid_file=harness.paths.pid_file,
                    log_dir=harness.paths.log_dir,
                    pid=os.getpid(),
                    model=MODEL_SLUG,
                    port=8080,
                    quant=QUANT,
                    preset_slot=1,
                    remote=False,
                )
                record_instance(
                    pid_file=harness.paths.pid_file,
                    log_dir=harness.paths.log_dir,
                    pid=child.pid,
                    model=OTHER_SLUG,
                    port=8081,
                    quant=QUANT,
                    preset_slot=1,
                    remote=False,
                )
                app._refresh_pid()
                panel = app.query_one(StatusPanel)
                self.assertEqual(panel.instance_preset_names[os.getpid()], "120K-HighPower")
                self.assertEqual(panel.instance_preset_names[child.pid], "coding")
                collapsed = " ".join(render_text(panel.render()).split())
                self.assertIn(f"{QUANT} [1] 120K-HighPower", collapsed)
                self.assertIn(f"{QUANT} [1] coding", collapsed)
                self.assertIn(OTHER_DISPLAY, collapsed)
        finally:
            child.terminate()
            child.wait(timeout=5)
            harness.cleanup()


class VisionEyeballSpacingTests(unittest.TestCase):
    def test_on_off_and_missing_share_the_same_label_width(self) -> None:
        from rich.cells import cell_len
        from tui.widgets.nav import EYEBALL, VisionEyeball

        missing = VisionEyeball("demo", "missing")
        on = VisionEyeball("demo", "on")
        off = VisionEyeball("demo", "off")
        self.assertEqual(missing.content, f"{EYEBALL}!")
        self.assertEqual(on.content, f"{EYEBALL} ")
        self.assertEqual(off.content, f"{EYEBALL} ")
        self.assertEqual(cell_len(str(missing.content)), cell_len(str(on.content)))
        self.assertEqual(cell_len(str(on.content)), cell_len(str(off.content)))


class ModelNavCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from tests.support import Harness

        self.harness = Harness()

    async def asyncTearDown(self) -> None:
        self.harness.cleanup()

    async def test_hides_icons_when_capabilities_absent(self) -> None:
        from tui.widgets.nav import ModelNav, ToolsBadge, VisionEyeball

        app = self.harness.app()
        async with app.run_test(size=(120, 40)):
            nav = app.query_one(ModelNav)
            self.assertEqual(len(list(nav.query(VisionEyeball))), 0)
            self.assertEqual(len(list(nav.query(ToolsBadge))), 0)

    async def test_icons_only_on_model_cards(self) -> None:
        from tests.support import MODEL_SLUG
        from tui.data.models_json import save_registry
        from tui.widgets.nav import ModelNav, ToolsBadge, VisionEyeball

        app = self.harness.app()
        cfg = app.registry.models[MODEL_SLUG]
        cfg.params["capabilities"] = {"vision": True, "tools": True}
        cfg.params["mmproj"] = "demo/mmproj-Demo-f16.gguf"
        save_registry(app.paths.models_json, app.registry)
        async with app.run_test(size=(120, 40)) as pilot:
            nav = app.query_one(ModelNav)
            await pilot.pause()
            eyeballs = list(nav.query(VisionEyeball))
            hammers = list(nav.query(ToolsBadge))
            self.assertEqual(len(eyeballs), 1)
            self.assertEqual(len(hammers), 1)
            self.assertEqual(eyeballs[0].model_name, MODEL_SLUG)
            self.assertEqual(eyeballs[0].state, "missing")
            preset_items = list(nav.query("ListItem.nav-preset"))
            self.assertTrue(preset_items)
            for item in preset_items:
                self.assertEqual(len(list(item.query(VisionEyeball))), 0)
                self.assertEqual(len(list(item.query(ToolsBadge))), 0)

    async def test_eyeball_toggle_persists_and_keeps_missing_state(self) -> None:
        from tests.support import MODEL_SLUG
        from tui.data.models_json import load_registry, save_registry
        from tui.widgets.nav import ModelNav, VisionEyeball

        app = self.harness.app()
        cfg = app.registry.models[MODEL_SLUG]
        cfg.params["capabilities"] = {"vision": True, "tools": False}
        cfg.params["mmproj"] = "demo/mmproj-Demo-f16.gguf"
        save_registry(app.paths.models_json, app.registry)
        async with app.run_test(size=(120, 40)) as pilot:
            nav = app.query_one(ModelNav)
            await pilot.pause()
            eyeball = nav.query_one(VisionEyeball)
            self.assertEqual(eyeball.state, "missing")
            self.assertTrue(app.registry.models[MODEL_SLUG].params.get("vision_enabled", True))
            await pilot.click(eyeball)
            await pilot.pause()
            self.assertFalse(app.registry.models[MODEL_SLUG].params["vision_enabled"])
            saved = load_registry(app.paths.models_json, models_dir=app.paths.models_dir)
            self.assertFalse(saved.models[MODEL_SLUG].params["vision_enabled"])
            eyeball = nav.query_one(VisionEyeball)
            self.assertEqual(eyeball.state, "missing")
            self.assertTrue(eyeball.has_class("vision-missing"))

    async def test_eyeball_on_and_off_when_projector_present(self) -> None:
        from tests.support import MODEL_SLUG
        from tui.data.models_json import save_registry
        from tui.widgets.nav import ModelNav, VisionEyeball

        app = self.harness.app()
        mmproj = app.paths.models_dir / "demo" / "mmproj-Demo-f16.gguf"
        mmproj.write_bytes(b"gguf")
        cfg = app.registry.models[MODEL_SLUG]
        cfg.params["capabilities"] = {"vision": True, "tools": False}
        cfg.params["mmproj"] = "demo/mmproj-Demo-f16.gguf"
        save_registry(app.paths.models_json, app.registry)
        async with app.run_test(size=(120, 40)) as pilot:
            nav = app.query_one(ModelNav)
            await pilot.pause()
            eyeball = nav.query_one(VisionEyeball)
            self.assertEqual(eyeball.state, "on")
            self.assertTrue(eyeball.has_class("vision-on"))
            await pilot.click(eyeball)
            await pilot.pause()
            eyeball = nav.query_one(VisionEyeball)
            self.assertEqual(eyeball.state, "off")
            self.assertTrue(eyeball.has_class("vision-off"))
            self.assertFalse(app.registry.models[MODEL_SLUG].params["vision_enabled"])

    async def test_keyboard_selection_still_moves(self) -> None:
        from tui.widgets.nav import ModelNav

        app = self.harness.app()
        async with app.run_test(size=(120, 40)) as pilot:
            nav = app.query_one(ModelNav)
            nav.focus()
            first = nav.selected_data
            await pilot.press("down")
            await pilot.pause()
            self.assertIsNotNone(nav.selected_data)
            self.assertNotEqual(nav.selected_data, first)


if __name__ == "__main__":
    unittest.main()
