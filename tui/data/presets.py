"""Presets — full llama-server configs per model quant. TUI owns presets.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tui.data.preset_template import (
    DEFAULT_PRESET_NAME,
    default_preset_params,
)

MAX_PRESETS_PER_MODEL = 5

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
    "threads": "THREADS",
    "parallel": "PARALLEL",
    "thinking": "THINKING",
    "per_slot_min": "PER_SLOT_MIN",
    "metrics": "METRICS",
}


@dataclass
class Preset:
    slot: int
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PresetStore:
    """model -> quant -> slot -> Preset"""

    presets: dict[str, dict[str, dict[int, Preset]]] = field(default_factory=dict)
    active: dict[str, dict[str, int]] = field(default_factory=dict)


def _preset_params_from_raw(preset_data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(preset_data.get("params"), dict):
        return dict(preset_data["params"])
    return dict(preset_data.get("overrides") or {})


def _is_slot_dict(value: dict[str, Any]) -> bool:
    if not value:
        return False
    for k, v in value.items():
        if not str(k).isdigit():
            return False
        if not isinstance(v, dict):
            return False
        if "params" not in v and "overrides" not in v and "name" not in v:
            return False
    return True


def _parse_active(data: dict[str, Any]) -> dict[str, dict[str, int]]:
    active_raw = data.get("_active")
    if not isinstance(active_raw, dict):
        return {}

    if "model" in active_raw and "slot" in active_raw:
        model = str(active_raw.get("model", ""))
        slot = active_raw.get("slot")
        if model and slot is not None:
            return {model: {"": int(slot)}}
        return {}

    result: dict[str, dict[str, int]] = {}
    for model, quant_map in active_raw.items():
        if isinstance(quant_map, dict):
            parsed: dict[str, int] = {}
            for quant, slot in quant_map.items():
                parsed[str(quant)] = int(slot)
            result[str(model)] = parsed
        elif isinstance(quant_map, int):
            result[str(model)] = {"": int(quant_map)}
    return result


def _migrate_flat_presets(data: dict[str, Any]) -> tuple[dict[str, dict[str, dict[int, Preset]]], dict[str, dict[str, int]], bool]:
    changed = False
    presets: dict[str, dict[str, dict[int, Preset]]] = {}
    active = _parse_active(data)

    for model_name, body in data.items():
        if model_name.startswith("_") or not isinstance(body, dict):
            continue
        if _is_slot_dict(body):
            changed = True
            quant = "LOCAL"
            presets.setdefault(model_name, {})[quant] = {}
            for slot_str, preset_data in body.items():
                slot = int(slot_str)
                presets[model_name][quant][slot] = Preset(
                    slot=slot,
                    name=preset_data.get("name", f"slot-{slot}"),
                    params=_preset_params_from_raw(preset_data),
                )
            if model_name in active and "" in active[model_name]:
                old_slot = active[model_name].pop("")
                active[model_name][quant] = old_slot
            continue

        presets[model_name] = {}
        for quant, slots in body.items():
            if not isinstance(slots, dict):
                continue
            presets[model_name][str(quant)] = {}
            for slot_str, preset_data in slots.items():
                if not str(slot_str).isdigit() or not isinstance(preset_data, dict):
                    continue
                slot = int(slot_str)
                params = _preset_params_from_raw(preset_data)
                if "params" not in preset_data and preset_data.get("overrides") is not None:
                    changed = True
                presets[model_name][str(quant)][slot] = Preset(
                    slot=slot,
                    name=preset_data.get("name", f"slot-{slot}"),
                    params=params,
                )

    return presets, active, changed


def load_presets(path: Path) -> PresetStore:
    if not path.exists():
        return PresetStore()

    data = json.loads(path.read_text())
    presets, active, changed = _migrate_flat_presets(data)
    store = PresetStore(presets=presets, active=active)

    if changed:
        save_presets(path, store)

    store.active = {
        model: {q: s for q, s in quants.items() if get_preset(store, model, q, s) is not None}
        for model, quants in store.active.items()
    }
    store.active = {m: q for m, q in store.active.items() if q}
    return store


def save_presets(path: Path, store: PresetStore) -> None:
    data: dict[str, Any] = {}
    if store.active:
        data["_active"] = {
            model: {quant: slot for quant, slot in sorted(quants.items())}
            for model, quants in sorted(store.active.items())
        }
    for model_name, quants in store.presets.items():
        data[model_name] = {}
        for quant, slots in quants.items():
            data[model_name][quant] = {}
            for slot, preset in slots.items():
                data[model_name][quant][str(slot)] = {
                    "name": preset.name,
                    "params": preset.params,
                }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def resolve_quant(model: str, quant: str | None, active_quant: str | None = None) -> str:
    return quant or active_quant or "LOCAL"


def get_active_slot(store: PresetStore, model: str, quant: str) -> int | None:
    return store.active.get(model, {}).get(quant)


def set_active_preset(store: PresetStore, model: str, quant: str, slot: int) -> None:
    store.active.setdefault(model, {})[quant] = slot


def clear_active_preset(store: PresetStore, model: str, quant: str | None = None) -> None:
    if quant is None:
        store.active.pop(model, None)
        return
    if model in store.active:
        store.active[model].pop(quant, None)
        if not store.active[model]:
            store.active.pop(model, None)


def get_preset(store: PresetStore, model: str, quant: str, slot: int) -> Preset | None:
    return store.presets.get(model, {}).get(quant, {}).get(slot)


def list_presets_for_quant(store: PresetStore, model: str, quant: str) -> dict[int, Preset]:
    return dict(store.presets.get(model, {}).get(quant, {}))


def next_free_slot(store: PresetStore, model: str, quant: str) -> int | None:
    used = store.presets.get(model, {}).get(quant, {})
    for slot in range(1, MAX_PRESETS_PER_MODEL + 1):
        if slot not in used:
            return slot
    return None


def set_preset(
    store: PresetStore,
    model: str,
    quant: str,
    slot: int,
    name: str,
    params: dict[str, Any],
) -> None:
    store.presets.setdefault(model, {}).setdefault(quant, {})[slot] = Preset(
        slot=slot, name=name, params=dict(params)
    )


def delete_preset(store: PresetStore, model: str, quant: str, slot: int) -> None:
    quants = store.presets.get(model, {})
    if quant in quants and slot in quants[quant]:
        del quants[quant][slot]
        if not quants[quant]:
            del quants[quant]
        if not quants:
            del store.presets[model]


def delete_all_presets_for_model(store: PresetStore, model: str) -> None:
    store.presets.pop(model, None)
    store.active.pop(model, None)


def seed_default_preset(
    store: PresetStore,
    model: str,
    quant: str,
    *,
    runtime_seed: dict[str, Any] | None = None,
    activate: bool = True,
) -> bool:
    """Create slot 1 from the fixed template if this quant has no presets yet."""
    if list_presets_for_quant(store, model, quant):
        return False
    params = default_preset_params(seed=runtime_seed)
    set_preset(store, model, quant, 1, DEFAULT_PRESET_NAME, params)
    if activate or get_active_slot(store, model, quant) is None:
        set_active_preset(store, model, quant, 1)
    return True


def migrate_preset_params(
    store: PresetStore,
    model: str,
    quant: str,
    runtime_seed: dict[str, Any],
) -> bool:
    """Upgrade legacy override-only presets to full params using a runtime seed."""
    changed = False
    slots = list_presets_for_quant(store, model, quant)
    if not slots:
        seed_default_preset(store, model, quant, runtime_seed=runtime_seed)
        return True

    full_seed = default_preset_params(seed=runtime_seed)
    for preset in slots.values():
        merged = dict(full_seed)
        merged.update(preset.params)
        if merged != preset.params:
            preset.params = merged
            changed = True
    if get_active_slot(store, model, quant) is None:
        set_active_preset(store, model, quant, 1)
        changed = True
    return changed


def merge_identity_and_preset(identity: dict[str, Any], preset: Preset) -> dict[str, Any]:
    merged = dict(identity)
    merged.update(preset.params)
    return merged


def params_to_env(params: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in params.items():
        env_key = PARAM_TO_ENV.get(key, key.upper())
        env[env_key] = str(value)
    return env


def overrides_to_env(overrides: dict[str, Any]) -> dict[str, str]:
    """Backward-compatible alias."""
    return params_to_env(overrides)


def apply_preset(base_params: dict[str, Any], preset: Preset) -> dict[str, Any]:
    """Backward-compatible merge helper."""
    return merge_identity_and_preset(base_params, preset)
