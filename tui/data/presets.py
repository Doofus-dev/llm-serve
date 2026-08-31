"""Presets — named override sets for models. TUI owns presets.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_PRESETS_PER_MODEL = 5

# Map models.json param names to llm-serve env var names
PARAM_TO_ENV: dict[str, str] = {
    "ctx": "CONTEXT_SIZE",
    "gpu_layers": "GPU_LAYERS",
    "cache_k": "CACHE_TYPE_K",
    "cache_v": "CACHE_TYPE_V",
    "batch": "BATCH_SIZE",
    "ubatch": "UBATCH",
    "n_predict": "N_PREDICT",
    "defrag": "DEFRAG_THOLD",
    "flash_attn": "FLASH_ATTN",
    "checkpoint_every": "CHECKPOINT_EVERY",
    "jinja": "JINJA",
    "threads_batch": "THREADS_BATCH",
    "timeout": "SERVER_TIMEOUT",
    "cache_prompt": "CACHE_PROMPT",
    "cache_reuse": "CACHE_REUSE",
    "cont_batching": "CONT_BATCHING",
    "ctx_checkpoints": "CTX_CHECKPOINTS",
    "temp": "TEMP",
    "top_p": "TOP_P",
    "top_k": "TOP_K",
    "min_p": "MIN_P",
    "seed": "SEED",
    "repeat_penalty": "REPEAT_PENALTY",
    "repeat_last_n": "REPEAT_LAST_N",
    "presence_penalty": "PRESENCE_PENALTY",
    "frequency_penalty": "FREQUENCY_PENALTY",
    "reasoning_format": "REASONING_FORMAT",
    "reasoning_budget": "REASONING_BUDGET",
    "mtp": "ENABLE_MTP",
    "spec_draft_n_max": "SPEC_DRAFT_N_MAX",
    "spec_draft_n_min": "SPEC_DRAFT_N_MIN",
    "n_cpu_moe": "N_CPU_MOE",
    "port": "PORT",
    "host": "HOST",
    "threads": "THREADS",
    "parallel": "PARALLEL",
    "thinking": "THINKING",
    "per_slot_min": "PER_SLOT_MIN",
}


@dataclass
class Preset:
    slot: int  # 1-5
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class PresetStore:
    """model_name -> {slot: Preset}"""
    presets: dict[str, dict[int, Preset]] = field(default_factory=dict)
    active: tuple[str, int] | None = None  # (model_name, slot)


def load_presets(path: Path) -> PresetStore:
    """Load presets.json."""
    if not path.exists():
        return PresetStore()

    data = json.loads(path.read_text())
    store = PresetStore()

    active = data.get("_active")
    if isinstance(active, dict):
        model = active.get("model")
        slot = active.get("slot")
        if model and slot is not None:
            store.active = (model, int(slot))

    for model_name, slots in data.items():
        if model_name.startswith("_"):
            continue
        store.presets[model_name] = {}
        for slot_str, preset_data in slots.items():
            slot = int(slot_str)
            store.presets[model_name][slot] = Preset(
                slot=slot,
                name=preset_data.get("name", f"slot-{slot}"),
                overrides=preset_data.get("overrides", {})
            )

    # Drop active if the preset was deleted or model removed
    if store.active:
        model, slot = store.active
        if get_preset(store, model, slot) is None:
            store.active = None

    return store


def save_presets(path: Path, store: PresetStore) -> None:
    """Save presets to presets.json."""
    data: dict[str, Any] = {}
    if store.active:
        model, slot = store.active
        data["_active"] = {"model": model, "slot": slot}
    for model_name, slots in store.presets.items():
        data[model_name] = {}
        for slot, preset in slots.items():
            data[model_name][str(slot)] = {
                "name": preset.name,
                "overrides": preset.overrides
            }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def set_active_preset(store: PresetStore, model: str, slot: int) -> None:
    """Mark a preset as active for its model."""
    store.active = (model, slot)


def clear_active_preset(store: PresetStore) -> None:
    """Clear the active preset selection."""
    store.active = None


def get_preset(store: PresetStore, model: str, slot: int) -> Preset | None:
    """Get a preset by model and slot."""
    return store.presets.get(model, {}).get(slot)


def set_preset(store: PresetStore, model: str, slot: int, name: str, overrides: dict[str, Any]) -> None:
    """Set a preset."""
    if model not in store.presets:
        store.presets[model] = {}
    store.presets[model][slot] = Preset(slot=slot, name=name, overrides=overrides)


def delete_preset(store: PresetStore, model: str, slot: int) -> None:
    """Delete a preset."""
    if model in store.presets and slot in store.presets[model]:
        del store.presets[model][slot]
        if not store.presets[model]:
            del store.presets[model]


def apply_preset(base_params: dict[str, Any], preset: Preset) -> dict[str, Any]:
    """Merge base model params with preset overrides."""
    merged = dict(base_params)
    merged.update(preset.overrides)
    return merged


def overrides_to_env(overrides: dict[str, Any]) -> dict[str, str]:
    """Convert preset overrides to llm-serve environment variables."""
    env: dict[str, str] = {}
    for key, value in overrides.items():
        env_key = PARAM_TO_ENV.get(key, key.upper())
        env[env_key] = str(value)
    return env
