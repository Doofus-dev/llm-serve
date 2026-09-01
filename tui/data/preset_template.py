"""Default preset template and identity vs runtime param keys."""

from __future__ import annotations

from typing import Any

# Profile fields stored in models.json (shared per model family).
IDENTITY_KEYS = frozenset({
    "display",
    "file",
    "source",
    "quants",
    "active_quant",
    "port",
    "host",
    "notes",
    "total_layers",
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
    "ctx",
    "per_slot_min",
    "cache_k",
    "cache_v",
    "batch",
    "ubatch",
    "threads",
    "threads_batch",
    "parallel",
    "n_predict",
    "defrag",
    "thinking",
    "reasoning_format",
    "reasoning_budget",
    "flash_attn",
    "checkpoint_every",
    "jinja",
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
    "metrics",
})

DEFAULT_PRESET_PARAMS: dict[str, Any] = {
    "gpu_layers": 99,
    "n_cpu_moe": "",
    "ctx": 32768,
    "per_slot_min": 0,
    "cache_k": "q4_0",
    "cache_v": "q4_0",
    "batch": 2048,
    "ubatch": 256,
    "threads": 8,
    "threads_batch": "",
    "parallel": 1,
    "n_predict": 8192,
    "defrag": 0.1,
    "thinking": "off",
    "reasoning_format": "",
    "reasoning_budget": "",
    "flash_attn": "on",
    "checkpoint_every": -1,
    "jinja": "on",
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
    "metrics": "on",
}

DEFAULT_PRESET_NAME = "default"

PRESET_PARAM_GROUPS: dict[str, list[str]] = {
    "Architecture": ["total_layers", "gpu_layers", "n_cpu_moe"],
    "Context": ["ctx", "per_slot_min", "cache_k", "cache_v"],
    "Batching": ["batch", "ubatch", "threads", "threads_batch", "parallel"],
    "Generation": ["n_predict", "defrag", "thinking", "reasoning_format", "reasoning_budget"],
    "Server": [
        "flash_attn",
        "checkpoint_every",
        "jinja",
        "timeout",
        "cache_prompt",
        "cache_reuse",
        "cont_batching",
        "ctx_checkpoints",
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
    "Metrics": ["metrics"],
}


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


def default_preset_params(*, seed: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(DEFAULT_PRESET_PARAMS)
    if seed:
        params.update(extract_runtime_params(seed))
    return params
