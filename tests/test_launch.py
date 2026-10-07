"""Launch argv, env overrides, and pidfile-only stop."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tui.launch import (
    LaunchError,
    allocate_port,
    apply_env_overrides,
    build_server_args,
    launch_background,
    prepare_launch,
    rotate_log,
    status_text,
    stop_server,
)
from tui.data.gpu import query_pid_vram_mb
from tui.data.pidfile import list_instances, write_pid_file
from tui.data.preset_template import DEFAULT_PRESET_PARAMS
from tui.data.presets import PARAM_TO_ENV

from tests.support import DISPLAY, MODEL_SLUG, OTHER_SLUG, QUANT, Harness


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

        for info in list_instances(
            pid_file=self.harness.paths.pid_file, log_dir=self.harness.paths.log_dir
        ):
            if info.pid != os.getpid() and info.alive:
                stop_server(info.model, paths=self.harness.paths)
        leftover = read_pid_file(self.harness.paths.pid_file)
        if leftover is not None and leftover.pid != os.getpid() and leftover.alive:
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

    def test_concurrent_launch_tracks_two_ports(self) -> None:
        first = prepare_launch(MODEL_SLUG, paths=self.harness.paths)
        pid1 = launch_background(first, paths=self.harness.paths, failfast_seconds=0.2)
        second = prepare_launch(OTHER_SLUG, paths=self.harness.paths)
        pid2 = launch_background(second, paths=self.harness.paths, failfast_seconds=0.2)
        self.assertNotEqual(pid1, pid2)
        self.assertNotEqual(first.port, second.port)
        instances = list_instances(
            pid_file=self.harness.paths.pid_file, log_dir=self.harness.paths.log_dir
        )
        models = {info.model for info in instances}
        ports = {info.port for info in instances}
        self.assertEqual(models, {MODEL_SLUG, OTHER_SLUG})
        self.assertEqual(ports, {first.port, second.port})
        text, code = status_text(paths=self.harness.paths)
        self.assertEqual(code, 0)
        self.assertIn(MODEL_SLUG, text)
        self.assertIn(OTHER_SLUG, text)

    def test_stop_one_instance_leaves_the_other(self) -> None:
        first = prepare_launch(MODEL_SLUG, paths=self.harness.paths)
        pid1 = launch_background(first, paths=self.harness.paths, failfast_seconds=0.2)
        second = prepare_launch(OTHER_SLUG, paths=self.harness.paths)
        pid2 = launch_background(second, paths=self.harness.paths, failfast_seconds=0.2)
        message = stop_server(OTHER_SLUG, paths=self.harness.paths)
        self.assertIn("Stopped", message)
        self.assertIn(OTHER_SLUG, message)
        remaining = list_instances(
            pid_file=self.harness.paths.pid_file, log_dir=self.harness.paths.log_dir
        )
        self.assertEqual([info.pid for info in remaining], [pid1])
        self.assertTrue(any(info.model == MODEL_SLUG for info in remaining))
        self.assertFalse(any(info.pid == pid2 for info in remaining))

    def test_stop_without_args_only_stops_tracked(self) -> None:
        first = prepare_launch(MODEL_SLUG, paths=self.harness.paths)
        pid1 = launch_background(first, paths=self.harness.paths, failfast_seconds=0.2)
        second = prepare_launch(OTHER_SLUG, paths=self.harness.paths)
        pid2 = launch_background(second, paths=self.harness.paths, failfast_seconds=0.2)
        message = stop_server(paths=self.harness.paths)
        self.assertIn(MODEL_SLUG, message)
        remaining = list_instances(
            pid_file=self.harness.paths.pid_file, log_dir=self.harness.paths.log_dir
        )
        self.assertEqual([info.pid for info in remaining], [pid2])
        self.assertFalse(self.harness.paths.pid_file.exists())

    def test_port_allocation_skips_running_instance_port(self) -> None:
        first = prepare_launch(MODEL_SLUG, paths=self.harness.paths)
        launch_background(first, paths=self.harness.paths, failfast_seconds=0.2)
        again = prepare_launch(MODEL_SLUG, paths=self.harness.paths)
        self.assertNotEqual(again.port, first.port)
        pid = launch_background(again, paths=self.harness.paths, failfast_seconds=0.2)
        instances = list_instances(
            pid_file=self.harness.paths.pid_file, log_dir=self.harness.paths.log_dir
        )
        self.assertEqual(len(instances), 2)
        self.assertEqual({info.port for info in instances}, {first.port, again.port})
        self.assertTrue(any(info.pid == pid for info in instances))

    def test_status_vram_from_nvidia_smi_and_degrades(self) -> None:
        plan = prepare_launch(MODEL_SLUG, paths=self.harness.paths)
        pid = launch_background(plan, paths=self.harness.paths, failfast_seconds=0.2)
        with patch("tui.launch.query_pid_vram_mb", return_value={pid: 8192.0}):
            text, code = status_text(paths=self.harness.paths)
        self.assertEqual(code, 0)
        self.assertIn("VRAM", text)
        self.assertIn("8.0G", text)
        with patch("tui.launch.query_pid_vram_mb", return_value=None):
            text, _ = status_text(paths=self.harness.paths)
        self.assertIn("n/a", text)
        with patch("tui.launch.query_pid_vram_mb", return_value={}):
            text, _ = status_text(paths=self.harness.paths)
        self.assertIn("n/a", text)


class PortAllocationTests(unittest.TestCase):
    def test_allocate_port_skips_taken(self) -> None:
        port = allocate_port(8081, {8081}, "127.0.0.1")
        self.assertNotEqual(port, 8081)
        self.assertGreater(port, 8081)


class VramQueryTests(unittest.TestCase):
    def test_query_pid_vram_sums_rows_and_degrades_when_missing(self) -> None:
        csv = (
            "pid, process_name, used_memory [MiB]\n"
            "10, /usr/bin/llama-server, 8192 MiB\n"
            "10, /usr/bin/llama-server, 1024 MiB\n"
        )
        result = Mock(returncode=0, stdout=csv)
        with patch("tui.data.gpu.subprocess.run", return_value=result):
            by_pid = query_pid_vram_mb()
        self.assertEqual(by_pid[10], 9216.0)
        with patch("tui.data.gpu.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(query_pid_vram_mb())
        failed = Mock(returncode=1, stdout="")
        with patch("tui.data.gpu.subprocess.run", return_value=failed):
            self.assertIsNone(query_pid_vram_mb())


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
