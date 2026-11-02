"""Default preset template and identity vs runtime param keys."""

from __future__ import annotations

from typing import Any

# Profile fields stored in models.json (shared per model family).
IDENTITY_KEYS = frozenset({
    "display",
    "file",
    "files",
    "source",
    "quants",
    "active_quant",
    "port",
    "host",
    "notes",
    "total_layers",
    "context_length",
    "mmproj",
    "mtp_file",
    "capabilities",
    "vision_enabled",
    "mtp_enabled",
})

# Legacy Hermes-only keys — stripped on migration.
HERMES_KEYS = frozenset({
    "tool_use_enforcement",
    "compression_enabled",
    "compression_threshold",
    "api_max_retries",
    "sync_aux",
})

# llama-server runtime params owned by presets.
PRESET_PARAM_KEYS = frozenset({
    "gpu_layers",
    "n_cpu_moe",
    "device",
    "ctx",
    "per_slot_min",
    "cache_k",
    "cache_v",
    "kv_offload",
    "batch",
    "ubatch",
    "threads",
    "threads_batch",
    "parallel",
    "n_predict",
    "thinking",
    "reasoning_format",
    "reasoning_budget",
    "jinja",
    "flash_attn",
    "checkpoint_every",
    "timeout",
    "cache_prompt",
    "cache_reuse",
    "cont_batching",
    "ctx_checkpoints",
    "temp",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "mtp",
    "spec_draft_n_max",
    "spec_draft_n_min",
    "spec_draft_p_min",
    "gpu_layers_draft",
    "metrics",
    "load_mode",
    "lazy_mode",
    "rope_scaling",
    "rope_scale",
    "rope_freq_base",
    "rope_freq_scale",
    "yarn_orig_ctx",
    "yarn_ext_factor",
    "yarn_attn_factor",
    "yarn_beta_slow",
    "yarn_beta_fast",
    "numa",
    "typical_p",
    "dynatemp_range",
    "dynatemp_exp",
    "mirostat",
    "mirostat_lr",
    "mirostat_ent",
})

DEFAULT_PRESET_PARAMS: dict[str, Any] = {
    "gpu_layers": 99,
    "n_cpu_moe": "",
    "device": "",
    "ctx": 32768,
    "per_slot_min": 0,
    "cache_k": "q4_0",
    "cache_v": "q4_0",
    "kv_offload": "on",
    "batch": 2048,
    "ubatch": 256,
    "threads": 8,
    "threads_batch": "",
    "parallel": 1,
    "n_predict": 8192,
    "thinking": "off",
    "reasoning_format": "",
    "reasoning_budget": "",
    "jinja": "on",
    "flash_attn": "on",
    "checkpoint_every": -1,
    "timeout": 600,
    "cache_prompt": "on",
    "cache_reuse": 0,
    "cont_batching": "on",
    "ctx_checkpoints": "",
    "temp": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "seed": -1,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mtp": 0,
    "spec_draft_n_max": 2,
    "spec_draft_n_min": 0,
    "spec_draft_p_min": "",
    "gpu_layers_draft": "",
    "metrics": "on",
    "load_mode": "",
    "lazy_mode": "",
    "rope_scaling": "",
    "rope_scale": "",
    "rope_freq_base": "",
    "rope_freq_scale": "",
    "yarn_orig_ctx": "",
    "yarn_ext_factor": "",
    "yarn_attn_factor": "",
    "yarn_beta_slow": "",
    "yarn_beta_fast": "",
    "numa": "",
    "typical_p": "",
    "dynatemp_range": "",
    "dynatemp_exp": "",
    "mirostat": "",
    "mirostat_lr": "",
    "mirostat_ent": "",
}

DEFAULT_PRESET_NAME = "default"

# Common groups are expanded in the editor. Advanced groups start collapsed.
PRESET_PARAM_GROUPS: dict[str, list[str]] = {
    "Architecture": ["total_layers", "context_length", "gpu_layers", "n_cpu_moe", "device"],
    "Context": ["ctx", "per_slot_min", "cache_k", "cache_v", "kv_offload"],
    "Batching": ["batch", "ubatch", "threads", "threads_batch", "parallel"],
    "Generation": ["n_predict", "thinking", "reasoning_format", "reasoning_budget", "jinja"],
    "Server": [
        "flash_attn",
        "timeout",
        "cache_prompt",
        "cache_reuse",
        "cont_batching",
        "metrics",
    ],
    "Sampling": [
        "temp",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
    ],
    "Speculative": ["mtp", "spec_draft_n_max", "spec_draft_n_min"],
}

ADVANCED_PRESET_PARAM_GROUPS: dict[str, list[str]] = {
    "Load / memory": ["load_mode", "lazy_mode"],
    "RoPE / YaRN": [
        "rope_scaling",
        "rope_scale",
        "rope_freq_base",
        "rope_freq_scale",
        "yarn_orig_ctx",
        "yarn_ext_factor",
        "yarn_attn_factor",
        "yarn_beta_slow",
        "yarn_beta_fast",
    ],
    "CPU topology": ["numa"],
    "Extra sampling": [
        "typical_p",
        "dynatemp_range",
        "dynatemp_exp",
        "mirostat",
        "mirostat_lr",
        "mirostat_ent",
    ],
    "Speculative extras": ["spec_draft_p_min", "gpu_layers_draft"],
    "Checkpoints": ["checkpoint_every", "ctx_checkpoints"],
}

IDENTITY_EDITOR_KEYS = (
    "vision_enabled",
    "mtp_enabled",
    "mmproj",
    "mtp_file",
)


def extract_runtime_params(params: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    for key, value in params.items():
        if key in PRESET_PARAM_KEYS:
            runtime[key] = value
    return runtime


def strip_to_identity(params: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for key in IDENTITY_KEYS:
        if key in params:
            identity[key] = params[key]
    return identity


def identity_from_seed(base_params: dict[str, Any] | None = None) -> dict[str, Any]:
    seed = dict(base_params or {})
    for key in HERMES_KEYS:
        seed.pop(key, None)
    identity = strip_to_identity(seed)
    identity.setdefault("port", 8081)
    identity.setdefault("host", "127.0.0.1")
    identity.setdefault("notes", "")
    return identity


def default_preset_params(
    *,
    seed: dict[str, Any] | None = None,
    max_ctx: int | None = None,
) -> dict[str, Any]:
    from tui.data.context_length import cap_context_value, clamp_preset_params

    params = dict(DEFAULT_PRESET_PARAMS)
    if seed:
        params.update(extract_runtime_params(seed))
    if max_ctx is not None:
        ctx = params.get("ctx", DEFAULT_PRESET_PARAMS["ctx"])
        try:
            params["ctx"] = cap_context_value(int(ctx), max_ctx)
        except (TypeError, ValueError):
            params["ctx"] = cap_context_value(int(DEFAULT_PRESET_PARAMS["ctx"]), max_ctx)
    return clamp_preset_params(params, max_ctx)
