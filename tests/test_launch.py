"""Launch argv, env overrides, and pidfile-only stop."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tui.launch import (
    LaunchError,
    apply_env_overrides,
    build_server_args,
    launch_background,
    prepare_launch,
    rotate_log,
    status_text,
    stop_server,
)
from tui.data.pidfile import write_pid_file
from tui.data.preset_template import DEFAULT_PRESET_PARAMS
from tui.data.presets import PARAM_TO_ENV

from tests.support import DISPLAY, MODEL_SLUG, QUANT, Harness


class EnvOverrideTests(unittest.TestCase):
    def test_param_to_env_covers_preset_keys(self) -> None:
        for key in DEFAULT_PRESET_PARAMS:
            self.assertIn(key, PARAM_TO_ENV, msg=key)

    def test_context_size_env_overrides_ctx(self) -> None:
        merged = apply_env_overrides({"ctx": 32768, "gpu_layers": 99}, {"CONTEXT_SIZE": "4096"})
        self.assertEqual(merged["ctx"], "4096")
        self.assertEqual(merged["gpu_layers"], 99)


class ArgvTests(unittest.TestCase):
    def test_core_flags_and_optional_skip_empty(self) -> None:
        from pathlib import Path

        args = build_server_args(
            model_path=Path("/tmp/m.gguf"),
            host="127.0.0.1",
            port=8081,
            params={
                **DEFAULT_PRESET_PARAMS,
                "n_cpu_moe": "",
                "reasoning_format": "",
                "mtp": 0,
                "checkpoint_every": -1,
            },
            log_verbosity=4,
        )
        self.assertEqual(args[0:6], ["-m", "/tmp/m.gguf", "--host", "127.0.0.1", "--port", "8081"])
        self.assertIn("-ngl", args)
        self.assertIn("-c", args)
        self.assertIn("--jinja", args)
        self.assertIn("--metrics", args)
        self.assertNotIn("--n-cpu-moe", args)
        self.assertNotIn("--spec-type", args)
        self.assertNotIn("--checkpoint-min-step", args)
        self.assertNotIn("--reasoning-format", args)

    def test_mtp_and_thinking_flags(self) -> None:
        from pathlib import Path

        args = build_server_args(
            model_path=Path("/tmp/m.gguf"),
            host="0.0.0.0",
            port=9000,
            params={
                **DEFAULT_PRESET_PARAMS,
                "mtp": 1,
                "thinking": "off",
                "checkpoint_every": 256,
            },
            log_verbosity=3,
        )
        self.assertIn("--spec-type", args)
        self.assertIn("draft-mtp", args)
        i = args.index("--reasoning")
        self.assertEqual(args[i + 1], "off")
        self.assertIn("--checkpoint-min-step", args)


class PrepareLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        self.harness.cleanup()

    def test_resolves_display_name_and_remote_host(self) -> None:
        plan = prepare_launch(DISPLAY, remote=True, paths=self.harness.paths)
        self.assertEqual(plan.model_key, MODEL_SLUG)
        self.assertEqual(plan.host, "0.0.0.0")
        self.assertTrue(plan.remote)
        self.assertEqual(plan.quant, QUANT)
        self.assertIn("-c", plan.args)

    def test_env_overrides_context(self) -> None:
        env = {**os.environ, "CONTEXT_SIZE": "8192", "PORT": "9090"}
        plan = prepare_launch(MODEL_SLUG, env=env, paths=self.harness.paths)
        idx = plan.args.index("-c")
        self.assertEqual(plan.args[idx + 1], "8192")
        self.assertEqual(plan.port, 9090)

    def test_unknown_model(self) -> None:
        with self.assertRaises(LaunchError):
            prepare_launch("no-such-model", paths=self.harness.paths)


class ProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = Harness()

    def tearDown(self) -> None:
        from tui.data.pidfile import read_pid_file

        info = read_pid_file(self.harness.paths.pid_file)
        if info is not None and info.pid != os.getpid() and info.alive:
            stop_server(paths=self.harness.paths)
        self.harness.cleanup()

    def test_background_launch_writes_pid_and_stop_only_that_pid(self) -> None:
        plan = prepare_launch(MODEL_SLUG, paths=self.harness.paths)
        pid = launch_background(plan, paths=self.harness.paths, failfast_seconds=0.2)
        self.assertGreater(pid, 0)
        text, code = status_text(paths=self.harness.paths)
        self.assertEqual(code, 0)
        self.assertIn(MODEL_SLUG, text)

        killed: list[tuple[int, int]] = []
        real_kill = os.kill

        def tracking_kill(sig_pid: int, sig: int) -> None:
            killed.append((sig_pid, sig))
            real_kill(sig_pid, sig)

        with patch("tui.launch.os.kill", tracking_kill):
            message = stop_server(paths=self.harness.paths)
        self.assertIn("Stopped", message)
        self.assertTrue(all(sig_pid == pid for sig_pid, _ in killed))
        self.assertFalse(self.harness.paths.pid_file.exists())

    def test_stop_without_pidfile(self) -> None:
        self.assertEqual(stop_server(paths=self.harness.paths), "No model running")

    def test_stop_unknown_model(self) -> None:
        write_pid_file(
            self.harness.paths.pid_file,
            pid=os.getpid(),
            model=MODEL_SLUG,
            port=8081,
            quant=QUANT,
            preset_slot=1,
            remote=False,
        )
        with self.assertRaises(LaunchError):
            stop_server("no-such-model", paths=self.harness.paths)
        self.assertTrue(self.harness.paths.pid_file.exists())
        self.harness.paths.pid_file.unlink()


class RotateLogTests(unittest.TestCase):
    def test_truncates_to_half_max_when_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-serve.log"
            path.write_text("".join(f"{i}\n" for i in range(20)))
            rotate_log(path, max_lines=10)
            lines = path.read_text().splitlines()
            self.assertEqual(lines, [str(i) for i in range(15, 20)])

    def test_leaves_short_file_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-serve.log"
            path.write_text("".join(f"{i}\n" for i in range(5)))
            rotate_log(path, max_lines=10)
            self.assertEqual(path.read_text(), "".join(f"{i}\n" for i in range(5)))


if __name__ == "__main__":
    unittest.main()
