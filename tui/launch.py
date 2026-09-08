"""Build llama-server argv and manage the single tracked process.

CLI and TUI both call this module so preset flags live in one place.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping

from tui.data.models_json import Registry, load_registry, resolve_model_key
from tui.data.pidfile import (
    PidInfo,
    clear_pid_file,
    read_pid_file,
    write_pid_file,
)
from tui.data.preset_template import DEFAULT_PRESET_PARAMS
from tui.data.presets import (
    PARAM_TO_ENV,
    get_active_slot,
    get_preset,
    load_presets,
)
from tui.data.server_log import tail_lines
from tui.paths import AppPaths, default_paths

MAX_LOG_LINES = 1000
STOP_WAIT_SECONDS = 2.0
LAUNCH_FAILFAST_SECONDS = 1.0

ENV_TO_PARAM = {env_key: param for param, env_key in PARAM_TO_ENV.items()}

ALWAYS_FLAGS: tuple[tuple[str, str], ...] = (
    ("gpu_layers", "-ngl"),
    ("ctx", "-c"),
    ("flash_attn", "-fa"),
    ("threads", "-t"),
    ("parallel", "--parallel"),
    ("cache_k", "--cache-type-k"),
    ("cache_v", "--cache-type-v"),
)

OPTIONAL_FLAGS: tuple[tuple[str, str], ...] = (
    ("n_cpu_moe", "--n-cpu-moe"),
    ("batch", "-b"),
    ("ubatch", "--ubatch-size"),
    ("defrag", "--defrag-thold"),
    ("n_predict", "--n-predict"),
    ("threads_batch", "-tb"),
    ("timeout", "--timeout"),
    ("cache_reuse", "--cache-reuse"),
    ("ctx_checkpoints", "--ctx-checkpoints"),
    ("temp", "--temp"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
    ("seed", "--seed"),
    ("repeat_penalty", "--repeat-penalty"),
    ("repeat_last_n", "--repeat-last-n"),
    ("presence_penalty", "--presence-penalty"),
    ("frequency_penalty", "--frequency-penalty"),
    ("reasoning_format", "--reasoning-format"),
    ("reasoning_budget", "--reasoning-budget"),
)

BOOL_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("jinja", "--jinja", "--no-jinja"),
    ("cache_prompt", "--cache-prompt", "--no-cache-prompt"),
    ("cont_batching", "--cont-batching", "--no-cont-batching"),
    ("metrics", "--metrics", "--no-metrics"),
)

GPU_BIN_DIRS = (
    Path("/usr/local/cuda/bin"),
    Path("/opt/cuda/bin"),
    Path("/opt/rocm/bin"),
)
GPU_LIB_DIRS = (
    Path("/usr/local/cuda/lib64"),
    Path("/opt/cuda/lib64"),
    Path("/opt/rocm/lib"),
)


class LaunchError(Exception):
    """User-facing launch/stop failure."""


@dataclass
class LaunchPlan:
    requested: str
    model_key: str
    display: str
    quant: str
    preset_slot: int
    preset_name: str
    model_file: str
    model_path: Path
    host: str
    port: int
    remote: bool
    log_verbosity: int
    llama_server: Path
    args: list[str]
    env: dict[str, str]
    banner_lines: list[str] = field(default_factory=list)


def _present(value: object) -> bool:
    return value is not None and str(value) != ""


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _bool_flag(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"on", "true", "1", "yes"}:
        return True
    if text in {"off", "false", "0", "no"}:
        return False
    return None


def _env_override(env: Mapping[str, str], key: str) -> str | None:
    if key not in env:
        return None
    value = env[key]
    return value if value != "" else None


def gpu_runtime_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Prepend CUDA/ROCm bin and lib dirs when they exist on this machine."""
    env = dict(os.environ if base is None else base)
    path_parts = [p for p in GPU_BIN_DIRS if p.is_dir()]
    lib_parts = [p for p in GPU_LIB_DIRS if p.is_dir()]
    if path_parts:
        existing = env.get("PATH", "")
        prefix = os.pathsep.join(str(p) for p in path_parts)
        env["PATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    if lib_parts:
        existing = env.get("LD_LIBRARY_PATH", "")
        prefix = os.pathsep.join(str(p) for p in lib_parts)
        env["LD_LIBRARY_PATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    return env


def apply_env_overrides(params: dict[str, object], env: Mapping[str, str]) -> dict[str, object]:
    """Overlay PARAM_TO_ENV variables onto preset params."""
    merged = dict(params)
    for env_key, param in ENV_TO_PARAM.items():
        override = _env_override(env, env_key)
        if override is not None:
            merged[param] = override
    return merged


def quant_file_rel(cfg, quant: str) -> str:
    quants = cfg.params.get("quants") if cfg else None
    if isinstance(quants, dict):
        entry = quants.get(quant) or {}
        if isinstance(entry, dict) and entry.get("file"):
            return str(entry["file"])
    return str(cfg.file) if cfg else ""


def build_server_args(
    *,
    model_path: Path,
    host: str,
    port: int,
    params: Mapping[str, object],
    log_verbosity: int,
) -> list[str]:
    args = [
        "-m",
        str(model_path),
        "--host",
        str(host),
        "--port",
        str(port),
    ]
    for key, flag in ALWAYS_FLAGS:
        args.extend((flag, str(params[key])))
    for key, flag in OPTIONAL_FLAGS:
        value = params.get(key, "")
        if _present(value):
            args.extend((flag, str(value)))

    checkpoint = _as_int(params.get("checkpoint_every", -1), -1)
    if checkpoint > 0:
        args.extend(("--checkpoint-min-step", str(checkpoint)))

    for key, on_flag, off_flag in BOOL_FLAGS:
        flag = _bool_flag(params.get(key, ""))
        if flag is True:
            args.append(on_flag)
        elif flag is False:
            args.append(off_flag)

    args.extend(("-lv", str(log_verbosity)))

    thinking = str(params.get("thinking") or "").strip().lower()
    if thinking == "off":
        args.extend(("--reasoning", "off"))
    elif thinking == "on":
        args.extend(("--reasoning", "on"))

    mtp = params.get("mtp", 0)
    if str(mtp).strip() in {"1", "on", "true", "yes"}:
        args.extend(("--spec-type", "draft-mtp"))
        n_max = params.get("spec_draft_n_max", "")
        n_min = params.get("spec_draft_n_min", "")
        if _present(n_max):
            args.extend(("--spec-draft-n-max", str(n_max)))
        if _present(n_min):
            args.extend(("--spec-draft-n-min", str(n_min)))
    return args


def remote_urls(port: int) -> list[str]:
    ips: list[str] = []
    if shutil.which("ip"):
        try:
            output = subprocess.check_output(
                ["ip", "-4", "-o", "addr", "show", "scope", "global"],
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            output = ""
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                ips.append(parts[3].split("/", 1)[0])
    else:
        try:
            output = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            output = ""
        ips = [token for token in output.split() if token]
    seen: set[str] = set()
    urls: list[str] = []
    for ip in sorted(set(ips)):
        if ip in seen:
            continue
        seen.add(ip)
        urls.append(f"http://{ip}:{port}/v1")
    return urls


def _banner(plan: LaunchPlan, params: Mapping[str, object], notes: str, model_max_ctx: object) -> list[str]:
    parallel = max(_as_int(params.get("parallel"), 1), 1)
    ctx = _as_int(params.get("ctx"), 0)
    per_slot = ctx // parallel if parallel else ctx
    total_layers = _as_int(params.get("total_layers"), 0)
    gpu_layers = _as_int(params.get("gpu_layers"), 0)
    cpu_layers = max(total_layers - gpu_layers, 0) if total_layers > 0 and gpu_layers < total_layers else 0
    lines = [
        f"Starting llama-server: {plan.model_key}",
        f"  File:         {plan.model_file}",
        f"  URL:          http://{plan.host}:{plan.port}/v1",
        f"  Context:      {ctx} tokens  ({parallel} slot(s), {per_slot} per slot)",
    ]
    if _present(model_max_ctx) and str(model_max_ctx).isdigit():
        lines.append(f"  Model max:    {model_max_ctx} tokens")
    lines.append(
        f"  KV cache:     K={params.get('cache_k')}  V={params.get('cache_v')}"
    )
    if total_layers > 0:
        lines.append(f"  GPU layers:   {gpu_layers} / {total_layers}  ({cpu_layers} on CPU)")
    else:
        lines.append(f"  GPU layers:   {gpu_layers}")
    if _present(params.get("n_cpu_moe")):
        lines.append(
            f"  MoE offload:  experts of {params.get('n_cpu_moe')} layer(s) on CPU (--n-cpu-moe)"
        )
    if _present(params.get("batch")):
        lines.append(f"  Batch size:   {params.get('batch')}")
    if _present(params.get("ubatch")):
        lines.append(f"  UBatch:       {params.get('ubatch')}")
    if _present(params.get("n_predict")):
        lines.append(f"  Max response: {params.get('n_predict')} tokens")
    lines.append(
        f"  Sampling:     temp={params.get('temp')} top_p={params.get('top_p')} "
        f"top_k={params.get('top_k')} seed={params.get('seed')}"
    )
    lines.append(f"  Log level:    {plan.log_verbosity}  (3=info  4=trace  5=debug)")
    checkpoint = _as_int(params.get("checkpoint_every", -1), -1)
    if checkpoint > 0:
        lines.append(f"  Checkpoints:  min-step={checkpoint} tokens")
    else:
        lines.append("  Checkpoints:  llama-server default (min-step not overridden)")
    thinking = str(params.get("thinking") or "").strip().lower()
    if thinking == "off":
        lines.append("  Thinking:     disabled (--reasoning off)")
    elif thinking == "on":
        lines.append("  Thinking:     enabled (--reasoning on)")
    else:
        lines.append("  Thinking:     auto (llama-server default)")
    if str(params.get("mtp", "")).strip() in {"1", "on", "true", "yes"}:
        lines.append(f"  MTP:          enabled (draft-mtp, n-max {params.get('spec_draft_n_max')})")
    lines.append(f"  Notes:        {notes}")
    if plan.remote:
        lines.append("  Remote mode:  bound to 0.0.0.0 — reachable from other devices at:")
        urls = remote_urls(plan.port)
        if urls:
            lines.extend(f"      {url}" for url in urls)
        lines.append("      (Meshnet devices use the 100.x.x.x address; LAN devices use your local IP.)")
    lines.append("")
    return lines


def prepare_launch(
    requested: str,
    *,
    remote: bool = False,
    log_verbosity: int | None = None,
    env: Mapping[str, str] | None = None,
    paths: AppPaths | None = None,
    registry: Registry | None = None,
) -> LaunchPlan:
    env_map = os.environ if env is None else env
    paths = paths or default_paths(env=env_map)
    if not paths.models_json.is_file():
        raise LaunchError(f"models.json not found at {paths.models_json}")
    if not paths.presets_json.is_file():
        raise LaunchError(f"presets.json not found at {paths.presets_json}")

    data = json.loads(paths.models_json.read_text())
    model_key = resolve_model_key(data, requested)
    if not model_key:
        raise LaunchError(
            f"unknown model '{requested}'. Run 'llm-serve' to see available models."
        )

    registry = registry or load_registry(paths.models_json, models_dir=paths.models_dir)
    store = load_presets(paths.presets_json)
    cfg = registry.models[model_key]
    alias = registry.aliases.get(requested)

    quant = (
        alias.quant
        if alias is not None and alias.quant is not None
        else cfg.active_quant
    )
    if not quant:
        raise LaunchError(f"model '{requested}' has no active quant configured.")

    slot = (
        alias.preset_slot
        if alias is not None and alias.preset_slot is not None
        else get_active_slot(store, model_key, quant)
    )
    if slot is None:
        raise LaunchError(
            f"model '{requested}' ({quant}) has no active preset. Create one in the TUI."
        )
    preset = get_preset(store, model_key, quant, slot)
    if preset is None:
        raise LaunchError(
            f"preset [{slot}] for '{requested}' ({quant}) does not exist."
        )

    model_file = quant_file_rel(cfg, quant)
    if not model_file:
        raise LaunchError(f"quant '{quant}' for '{requested}' does not exist.")

    model_path_override = _env_override(env_map, "MODEL_PATH")
    model_path = Path(model_path_override) if model_path_override else paths.models_dir / model_file

    host_override = _env_override(env_map, "HOST")
    if host_override is not None:
        host = host_override
    elif remote:
        host = "0.0.0.0"
    else:
        host = str(cfg.host or "127.0.0.1")

    port_override = _env_override(env_map, "PORT")
    port = int(port_override) if port_override is not None else int(cfg.port)

    verbosity_override = _env_override(env_map, "LOG_VERBOSITY")
    if verbosity_override is not None:
        verbosity = _as_int(verbosity_override, 4)
    elif log_verbosity is not None:
        verbosity = log_verbosity
    else:
        verbosity = 4

    params: dict[str, object] = dict(DEFAULT_PRESET_PARAMS)
    params.update(preset.params)
    params["total_layers"] = cfg.params.get("total_layers", 0)
    params = apply_env_overrides(params, env_map)

    ctx = _as_int(params.get("ctx"), 0)
    parallel = max(_as_int(params.get("parallel"), 1), 1)
    per_slot_min = _as_int(params.get("per_slot_min"), 0)
    per_slot = ctx // parallel if parallel else ctx
    if per_slot_min > 0 and per_slot < per_slot_min:
        raise LaunchError(
            f"per-slot context is {per_slot} ({ctx} / {parallel}), "
            f"below the {per_slot_min}-token floor for '{model_key}'."
        )

    if not paths.llama_server.is_file() or not os.access(paths.llama_server, os.X_OK):
        raise LaunchError(f"llama-server not found at {paths.llama_server}")
    if not model_path.is_file():
        raise LaunchError(f"model not found at {model_path}")

    args = build_server_args(
        model_path=model_path,
        host=host,
        port=port,
        params=params,
        log_verbosity=verbosity,
    )
    plan = LaunchPlan(
        requested=requested,
        model_key=model_key,
        display=cfg.display,
        quant=quant,
        preset_slot=slot,
        preset_name=preset.name,
        model_file=model_file,
        model_path=model_path,
        host=host,
        port=port,
        remote=remote,
        log_verbosity=verbosity,
        llama_server=paths.llama_server,
        args=args,
        env=gpu_runtime_env(env_map),
    )
    plan.banner_lines = _banner(plan, params, cfg.notes, cfg.params.get("context_length"))
    model_max = cfg.params.get("context_length")
    if _present(model_max) and str(model_max).isdigit() and ctx > int(model_max):
        plan.banner_lines.insert(
            0,
            f"Warning: preset context {ctx} exceeds model max {model_max} — llama-server will cap it.",
        )
    return plan


def rotate_log(path: Path, *, max_lines: int = MAX_LOG_LINES) -> None:
    if not path.is_file():
        return
    lines = tail_lines(path, max_lines + 1, keepends=True)
    if len(lines) <= max_lines:
        return
    keep = lines[-(max_lines // 2) :]
    path.write_text("".join(keep))


def running_server(paths: AppPaths | None = None) -> PidInfo | None:
    paths = paths or default_paths()
    info = read_pid_file(paths.pid_file)
    if info is None:
        return None
    if info.alive:
        return info
    clear_pid_file(paths.pid_file)
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OverflowError, OSError):
        return False


def _stop_pid(pid: int, *, wait: float = STOP_WAIT_SECONDS) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.time() + wait
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return


def stop_server(
    model: str | None = None,
    *,
    paths: AppPaths | None = None,
) -> str:
    """Stop the tracked llama-server only (never pgrep unrelated processes)."""
    paths = paths or default_paths()
    info = read_pid_file(paths.pid_file)
    if info is None:
        return "No model running"

    if not info.alive:
        clear_pid_file(paths.pid_file)
        return f"Stale PID file (PID {info.pid} not running)"

    if model:
        data = json.loads(paths.models_json.read_text())
        resolved = resolve_model_key(data, model)
        if resolved is None:
            raise LaunchError(f"unknown model '{model}'")
        registry = load_registry(paths.models_json, models_dir=paths.models_dir)
        display = (
            registry.models[resolved].display if resolved in registry.models else resolved
        )
        if info.model not in {resolved, display, model}:
            return (
                f"Model '{resolved}' is not running (currently running: {info.model})"
            )

    _stop_pid(info.pid)
    clear_pid_file(paths.pid_file)
    return f"Stopped {info.model} (PID {info.pid})"


def status_text(paths: AppPaths | None = None) -> tuple[str, int]:
    paths = paths or default_paths()
    info = read_pid_file(paths.pid_file)
    if info is None:
        return "No model running", 0
    if not info.alive:
        clear_pid_file(paths.pid_file)
        return f"Stale PID file (PID {info.pid} not running)", 1
    try:
        started = int(info.ts)
        uptime = max(int(time.time()) - started, 0)
    except (TypeError, ValueError):
        uptime = 0
    hours, rem = divmod(uptime, 3600)
    minutes, seconds = divmod(rem, 60)
    access = "Remote (0.0.0.0)" if info.remote else "Local"
    lines = [
        f"Running: {info.model}",
        f"  PID:    {info.pid}",
        f"  Port:   {info.port}",
        f"  Access: {access}",
        f"  Uptime: {hours}h {minutes}m {seconds}s",
        "",
        "Last log lines:",
    ]
    if paths.log_file.is_file():
        tail = tail_lines(paths.log_file, 10)
        lines.extend(tail if tail else ["  (no log file)"])
    else:
        lines.append("  (no log file)")
    return "\n".join(lines), 0


def list_text(paths: AppPaths | None = None) -> str:
    paths = paths or default_paths()
    if not paths.models_json.is_file():
        raise LaunchError(f"models.json not found at {paths.models_json}")
    registry = load_registry(paths.models_json, models_dir=paths.models_dir)
    lines = ["Available models:", ""]
    for cfg in registry.models.values():
        lines.append(f"  {cfg.display:<22} {cfg.notes}")
    if registry.aliases:
        lines.extend(("", "Aliases:", ""))
        for name, target in registry.aliases.items():
            pin = ""
            if target.quant is not None and target.preset_slot is not None:
                pin = f" [{target.quant}/{target.preset_slot}]"
            lines.append(f"  {name} -> {target.model}{pin}")
    lines.extend(
        (
            "",
            "Commands:",
            "  llm-serve                      Open the interactive TUI",
            "  llm-serve --help               Show this model and command reference",
            "  llm-serve list                 Same as --help",
            "  llm-serve <model>              Start model in background",
            "  llm-serve <model> --live       Start model in foreground (live logs)",
            "  llm-serve status               Show running model info",
            "  llm-serve stop                 Stop the tracked server",
            "  llm-serve stop <model>         Stop specific model",
            "  llm-serve update               Pull + rebuild llama.cpp",
            "",
            "Env overrides:",
            "  PORT=           HOST=           CONTEXT_SIZE=   GPU_LAYERS=",
            "  THREADS=        UBATCH=         PARALLEL=       N_PREDICT=",
            "  CACHE_TYPE_K=   CACHE_TYPE_V=   BATCH_SIZE=     DEFRAG_THOLD=",
            "  TEMP=           TOP_P=          SEED=           ENABLE_MTP=0",
            "  N_CPU_MOE=      MODEL_PATH=     MODEL_DIR=",
            "",
            "Flags:",
            "  --live                Start in foreground with live logs",
            "  --remote              Bind 0.0.0.0 for LAN/Meshnet access",
            "  --dry-run             Show the command that would be run (no launch)",
        )
    )
    return "\n".join(lines)


def _write_launch_marker(log_file: Path, model_key: str, pid: int) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a") as handle:
        handle.write(f"── {stamp} launch: {model_key} (PID {pid}) ──\n")


def launch_foreground(plan: LaunchPlan) -> None:
    os.execvpe(str(plan.llama_server), [str(plan.llama_server), *plan.args], plan.env)


def launch_background(
    plan: LaunchPlan,
    *,
    paths: AppPaths | None = None,
    failfast_seconds: float = LAUNCH_FAILFAST_SECONDS,
) -> int:
    paths = paths or default_paths()
    existing = read_pid_file(paths.pid_file)
    if existing and existing.alive:
        raise LaunchError(
            f"Model '{existing.model}' is already running (PID: {existing.pid}). "
            "Stop it first with 'llm-serve stop'."
        )
    if existing:
        clear_pid_file(paths.pid_file)

    paths.log_dir.mkdir(parents=True, exist_ok=True)
    rotate_log(paths.log_file)
    log_handle = paths.log_file.open("a")
    try:
        proc = subprocess.Popen(
            [str(plan.llama_server), *plan.args],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=plan.env,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    _write_launch_marker(paths.log_file, plan.model_key, proc.pid)
    write_pid_file(
        paths.pid_file,
        pid=proc.pid,
        model=plan.model_key,
        port=plan.port,
        quant=plan.quant,
        preset_slot=plan.preset_slot,
        remote=plan.remote,
    )
    if failfast_seconds > 0:
        time.sleep(failfast_seconds)
        if not _pid_alive(proc.pid):
            tail = ""
            if paths.log_file.is_file():
                tail = "\n".join(tail_lines(paths.log_file, 10))
            clear_pid_file(paths.pid_file)
            extra = f"\nLast log lines:\n{tail}" if tail else ""
            raise LaunchError(
                f"llama-server exited immediately (PID: {proc.pid}).{extra}"
            )
    return proc.pid
