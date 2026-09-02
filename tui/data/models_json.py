"""Parse and write models.json — TUI owns this file."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tui.data.context_length import clamp_preset_params, resolve_context_length
from tui.data.gguf import apply_architecture_from_gguf, read_gguf_architecture
from tui.data.preset_template import (
    HERMES_KEYS,
    extract_runtime_params,
    identity_from_seed,
    strip_to_identity,
)
from tui.data.presets import (
    PresetStore,
    load_presets,
    migrate_preset_params,
    save_presets,
    seed_default_preset,
)
from tui.data.quant import (
    QuantEntry,
    author_size_label,
    default_model_slug,
    family_display,
    quant_entry_from_file,
    quant_from_filename,
)


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def file(self) -> str:
        return str(self.params.get("file", ""))

    @property
    def port(self) -> int:
        return int(self.params.get("port", 8081))

    @property
    def host(self) -> str:
        return str(self.params.get("host", "127.0.0.1"))

    @property
    def notes(self) -> str:
        return str(self.params.get("notes", ""))

    @property
    def display(self) -> str:
        return str(self.params.get("display") or self.name)

    @property
    def active_quant(self) -> str:
        return str(self.params.get("active_quant") or "")


@dataclass
class Registry:
    models: dict[str, ModelConfig] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


def _repo_id(params: dict[str, Any]) -> str | None:
    source = params.get("source")
    if isinstance(source, dict) and source.get("repo"):
        return str(source["repo"])
    return None


def _filename_from_params(params: dict[str, Any]) -> str:
    source = params.get("source")
    if isinstance(source, dict) and source.get("filename"):
        return str(source["filename"])
    file_rel = str(params.get("file", ""))
    return Path(file_rel).name if file_rel else ""


def _ensure_quants_block(params: dict[str, Any], models_dir: Path | None = None) -> dict[str, Any]:
    """Ensure params has quants/active_quant/display; derive from file if missing."""
    file_rel = str(params.get("file", ""))
    filename = _filename_from_params(params)
    if not filename and file_rel:
        filename = Path(file_rel).name

    quants = params.get("quants")
    if not isinstance(quants, dict):
        quants = {}

    if not quants and filename:
        qid = quant_from_filename(filename)
        quants[qid] = {"filename": filename, "file": file_rel}

    params["quants"] = quants

    if not params.get("active_quant") and quants:
        source = params.get("source")
        fn = filename
        if isinstance(source, dict) and source.get("filename"):
            fn = str(source["filename"])
        active = quant_from_filename(fn) if fn else next(iter(quants))
        if active not in quants:
            active = next(iter(quants))
        params["active_quant"] = active

    active = str(params.get("active_quant", ""))
    if active and active in quants:
        entry = quants[active]
        if isinstance(entry, dict):
            if entry.get("file"):
                params["file"] = entry["file"]
            if isinstance(params.get("source"), dict):
                params["source"]["filename"] = entry.get("filename", Path(str(entry.get("file", ""))).name)

    repo = _repo_id(params)
    if repo and filename and not params.get("display"):
        params["display"] = family_display(repo, filename)
    elif not params.get("display"):
        params["display"] = params.get("notes") or Path(file_rel).stem or "Model"

    return params


def resolve_model_key(data: dict[str, Any], requested: str) -> str | None:
    """Resolve CLI/TUI model name: alias, internal slug, or display name."""
    aliases = data.get("aliases", {})
    name = str(aliases.get(requested, requested))
    models = data.get("models", {})
    if name in models:
        return name
    for key, params in models.items():
        if str(params.get("display", "")) == name:
            return key
    if requested in models:
        return requested
    for key, params in models.items():
        if str(params.get("display", "")) == requested:
            return key
    return None


def validate_display_name(reg: Registry, display: str, *, exclude: str = "") -> str | None:
    display = display.strip()
    if not display:
        return "Display name cannot be empty"
    for name, cfg in reg.models.items():
        if name == exclude:
            continue
        if cfg.display == display:
            return f"Display name '{display}' is already in use"
    return None


def migrate_to_preset_architecture(data: dict[str, Any], presets_path: Path) -> bool:
    """Move runtime params from models.json into presets.json."""
    models = data.get("models", {})
    if not models:
        return False

    changed = False
    store = load_presets(presets_path) if presets_path.exists() else PresetStore()

    for model_name, raw_params in list(models.items()):
        params = dict(raw_params)
        for key in HERMES_KEYS:
            if key in params:
                del params[key]
                changed = True

        params = _ensure_quants_block(params)
        runtime = extract_runtime_params(params)
        if runtime:
            changed = True

        quants = params.get("quants")
        if isinstance(quants, dict) and quants:
            quant_ids = [str(q) for q in quants]
        else:
            quant_ids = [str(params.get("active_quant") or "LOCAL")]

        for quant_id in quant_ids:
            if migrate_preset_params(store, model_name, quant_id, runtime):
                changed = True

        identity = strip_to_identity(params)
        if identity != raw_params:
            changed = True
        data["models"][model_name] = identity

    if changed:
        save_presets(presets_path, store)
    return changed


def load_registry(path: Path, *, models_dir: Path | None = None) -> Registry:
    """Load models.json; migrate legacy profiles and preset-only architecture."""
    if not path.exists():
        return Registry()

    data = json.loads(path.read_text())
    migrated, preset_remap = migrate_models_json(data, models_dir=models_dir)
    presets_path = path.parent / "presets.json"
    preset_arch_migrated = migrate_to_preset_architecture(data, presets_path)
    if migrated or preset_arch_migrated:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        if preset_remap:
            _migrate_presets_file(presets_path, preset_remap)

    reg = Registry()
    for name, params in data.get("models", {}).items():
        if models_dir is not None:
            _ensure_quants_block(params, models_dir)
        else:
            _ensure_quants_block(params)
        reg.models[name] = ModelConfig(name=name, params=strip_to_identity(params))

    reg.aliases = dict(data.get("aliases", {}))
    return reg


def save_registry(path: Path, reg: Registry) -> None:
    """Save Registry back to models.json (identity fields only)."""
    data = {
        "models": {
            name: strip_to_identity(_ensure_quants_block(dict(cfg.params)))
            for name, cfg in reg.models.items()
        },
        "aliases": reg.aliases,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def find_model_by_repo(reg: Registry, repo_id: str) -> str | None:
    for name, cfg in reg.models.items():
        if _repo_id(cfg.params) == repo_id:
            return name
    return None


def get_quant_entries(cfg: ModelConfig, models_dir: Path) -> list[QuantEntry]:
    quants = cfg.params.get("quants") or {}
    entries: list[QuantEntry] = []
    for qid, raw in quants.items():
        if not isinstance(raw, dict):
            continue
        filename = str(raw.get("filename") or Path(str(raw.get("file", ""))).name)
        file_rel = str(raw.get("file") or cfg.file)
        entries.append(quant_entry_from_file(str(qid), filename, file_rel, models_dir))
    entries.sort(key=lambda e: e.quant_id)
    return entries


def merge_repo_catalog(params: dict[str, Any], files: list[tuple[str, str]]) -> None:
    """Merge Hub file list into quants (filename, optional size). files: (path, size)."""
    quants = params.setdefault("quants", {})
    if not isinstance(quants, dict):
        quants = {}
        params["quants"] = quants
    author = ""
    source = params.get("source")
    if isinstance(source, dict):
        author = str(source.get("author") or "")
    for path, _size in files:
        qid = quant_from_filename(path)
        file_rel = f"{author}/{path}" if author else path
        if qid not in quants:
            quants[qid] = {"filename": path, "file": file_rel}
        else:
            entry = quants[qid]
            if isinstance(entry, dict):
                entry.setdefault("filename", path)
                entry.setdefault("file", file_rel)


def set_active_quant(
    path: Path,
    model_name: str,
    quant_id: str,
    *,
    models_dir: Path | None = None,
) -> bool:
    reg = load_registry(path, models_dir=models_dir)
    cfg = reg.models.get(model_name)
    if not cfg:
        return False
    quants = cfg.params.get("quants") or {}
    if quant_id not in quants:
        return False
    entry = quants[quant_id]
    if not isinstance(entry, dict):
        return False
    cfg.params["active_quant"] = quant_id
    if entry.get("file"):
        cfg.params["file"] = entry["file"]
    if isinstance(cfg.params.get("source"), dict) and entry.get("filename"):
        cfg.params["source"]["filename"] = entry["filename"]
    gguf = None
    if models_dir and entry.get("file"):
        gguf = models_dir / str(entry["file"])
    apply_architecture_from_gguf(cfg.params, gguf, replace_cloned=False)
    save_registry(path, reg)
    return True


def delete_model(path: Path, name: str) -> None:
    reg = load_registry(path)
    if name in reg.models:
        del reg.models[name]
        save_registry(path, reg)


def create_model(path: Path, name: str, params: dict[str, Any]) -> None:
    reg = load_registry(path)
    reg.models[name] = ModelConfig(name=name, params=params)
    save_registry(path, reg)


def update_model(path: Path, name: str, params: dict[str, Any]) -> None:
    reg = load_registry(path)
    if name in reg.models:
        reg.models[name].params = strip_to_identity(_ensure_quants_block(dict(params)))
        save_registry(path, reg)


def merge_editor_params(existing: dict[str, Any], edited: dict[str, Any], known_keys: set[str]) -> dict[str, Any]:
    merged = dict(edited)
    for key, value in existing.items():
        if key not in known_keys:
            merged[key] = value
    return merged


def add_or_update_quant(
    path: Path,
    *,
    repo_id: str,
    filename: str,
    file_rel: str,
    source: dict[str, str],
    base_params: dict[str, Any],
    gguf_path: Path | None = None,
    models_dir: Path | None = None,
    slug: str | None = None,
    display: str | None = None,
    hf_context: int | None = None,
) -> tuple[str, str]:
    """Register a downloaded GGUF under its repo model. Returns (model_slug, quant_id)."""
    reg = load_registry(path, models_dir=models_dir)
    quant_id = quant_from_filename(filename)
    existing = find_model_by_repo(reg, repo_id)

    if existing:
        cfg = reg.models[existing]
        params = cfg.params
        quants = params.setdefault("quants", {})
        quants[quant_id] = {"filename": filename, "file": file_rel}
        params["active_quant"] = quant_id
        params["file"] = file_rel
        if isinstance(params.get("source"), dict):
            params["source"]["filename"] = filename
        apply_architecture_from_gguf(
            params, gguf_path, replace_cloned=True, hf_context=hf_context
        )
        save_registry(path, reg)
        _seed_preset_for_quant(
            path,
            existing,
            quant_id,
            identity=params,
            models_dir=models_dir,
            runtime_seed=extract_runtime_params(base_params),
        )
        return existing, quant_id

    author = source.get("author") or repo_id.split("/", 1)[0]
    model_slug = slug or default_model_slug(repo_id, filename, author)
    if model_slug in reg.models and _repo_id(reg.models[model_slug].params) != repo_id:
        model_slug = f"{model_slug}-{author}"[:48].strip("-")

    params = identity_from_seed(base_params)
    params["file"] = file_rel
    params["source"] = source
    params["display"] = display or family_display(repo_id, filename)
    params["active_quant"] = quant_id
    params["quants"] = {quant_id: {"filename": filename, "file": file_rel}}
    apply_architecture_from_gguf(
        params, gguf_path, replace_cloned=True, hf_context=hf_context
    )
    reg.models[model_slug] = ModelConfig(name=model_slug, params=params)
    save_registry(path, reg)
    _seed_preset_for_quant(
        path,
        model_slug,
        quant_id,
        identity=params,
        models_dir=models_dir,
        runtime_seed=extract_runtime_params(base_params),
    )
    return model_slug, quant_id


def _seed_preset_for_quant(
    models_path: Path,
    model_slug: str,
    quant_id: str,
    *,
    identity: dict[str, Any],
    models_dir: Path | None = None,
    runtime_seed: dict[str, Any],
) -> None:
    presets_path = models_path.parent / "presets.json"
    store = load_presets(presets_path)
    max_ctx = resolve_context_length(identity, models_dir)
    if seed_default_preset(
        store,
        model_slug,
        quant_id,
        runtime_seed=runtime_seed or None,
        max_ctx=max_ctx,
        activate=True,
    ):
        save_presets(presets_path, store)


def create_downloaded_model(
    path: Path,
    name: str,
    base_params: dict[str, Any],
    *,
    file_rel: str,
    source: dict[str, str],
    gguf_path: Path | None = None,
    models_dir: Path | None = None,
    hf_context: int | None = None,
) -> tuple[str, str]:
    """Backward-compatible wrapper; prefers repo merge over flat profile names."""
    repo = source.get("repo", "")
    filename = source.get("filename") or Path(file_rel).name
    if repo:
        return add_or_update_quant(
            path,
            repo_id=repo,
            filename=filename,
            file_rel=file_rel,
            source=source,
            base_params=base_params,
            gguf_path=gguf_path,
            models_dir=models_dir,
            slug=name if name and not re.search(r"(iq|q\d)", name.lower()) else None,
            display=family_display(repo, filename) if name and re.search(r"(iq|q\d)", name.lower()) else name,
            hf_context=hf_context,
        )
    params = identity_from_seed(base_params)
    params["file"] = file_rel
    params["source"] = source
    qid = quant_from_filename(filename)
    params["active_quant"] = qid
    params["quants"] = {qid: {"filename": filename, "file": file_rel}}
    params["display"] = name
    apply_architecture_from_gguf(
        params, gguf_path, replace_cloned=True, hf_context=hf_context
    )
    create_model(path, name, params)
    _seed_preset_for_quant(
        path,
        name,
        qid,
        identity=params,
        models_dir=models_dir,
        runtime_seed=extract_runtime_params(base_params),
    )
    return name, qid


def sync_gguf_architecture(registry_path: Path, models_dir: Path) -> list[tuple[str, str, int | None, int]]:
    reg = load_registry(registry_path, models_dir=models_dir)
    changes: list[tuple[str, str, int | None, int]] = []
    for name, cfg in reg.models.items():
        file_rel = str(cfg.params.get("file", ""))
        if not file_rel:
            continue
        info = read_gguf_architecture(models_dir / file_rel)
        if info.block_count is not None:
            old = cfg.params.get("total_layers")
            try:
                old_int = int(old) if old is not None else None
            except (TypeError, ValueError):
                old_int = None
            if old_int != info.block_count:
                cfg.params["total_layers"] = info.block_count
                changes.append((name, "total_layers", old_int, info.block_count))
        if info.context_length is not None:
            old = cfg.params.get("context_length")
            try:
                old_int = int(old) if old is not None else None
            except (TypeError, ValueError):
                old_int = None
            if old_int != info.context_length:
                cfg.params["context_length"] = info.context_length
                changes.append((name, "context_length", old_int, info.context_length))
    if changes:
        save_registry(registry_path, reg)
    return changes


def clamp_preset_contexts(
    registry_path: Path,
    presets_path: Path,
    *,
    models_dir: Path | None = None,
) -> list[tuple[str, str, int, int, int]]:
    """Cap preset ctx values that exceed each model's native context_length."""
    reg = load_registry(registry_path, models_dir=models_dir)
    if not presets_path.exists():
        return []
    store = load_presets(presets_path)
    changed: list[tuple[str, str, int, int, int]] = []
    dirty = False
    for model_name, quants in store.presets.items():
        cfg = reg.models.get(model_name)
        if cfg is None:
            continue
        max_ctx = resolve_context_length(cfg.params, models_dir)
        if max_ctx is None:
            continue
        for quant_id, slots in quants.items():
            for slot, preset in slots.items():
                try:
                    current = int(preset.params.get("ctx", 0))
                except (TypeError, ValueError):
                    continue
                capped = min(current, max_ctx)
                if capped != current:
                    preset.params["ctx"] = capped
                    changed.append((model_name, quant_id, slot, current, capped))
                    dirty = True
    if dirty:
        save_presets(presets_path, store)
    return changed


def migrate_models_json(data: dict[str, Any], *, models_dir: Path | None = None) -> tuple[bool, dict[str, tuple[str, str]]]:
    """Migrate flat per-quant profiles to repo-grouped models. Returns (changed, old->(model, quant))."""
    models = data.get("models", {})
    if not models:
        return False, {}

    repo_names: dict[str, list[str]] = {}
    for name, params in models.items():
        repo = _repo_id(params)
        if repo:
            repo_names.setdefault(repo, []).append(name)

    needs = any(
        not isinstance(m.get("quants"), dict) or not m.get("active_quant")
        for m in models.values()
    ) or any(len(names) > 1 for names in repo_names.values())

    if not needs:
        return False, {}

    preset_remap: dict[str, tuple[str, str]] = {}
    aliases = dict(data.get("aliases", {}))

    # Group hub models by repo
    repo_groups: dict[str, list[tuple[str, dict]]] = {}
    standalone: dict[str, dict] = {}

    for name, params in list(models.items()):
        repo = _repo_id(params)
        if repo:
            repo_groups.setdefault(repo, []).append((name, dict(params)))
            filename = _filename_from_params(params)
            preset_remap[name] = ("", quant_from_filename(filename))  # filled below
        else:
            standalone[name] = dict(params)

    new_models: dict[str, dict] = {}

    for name, params in standalone.items():
        filename = _filename_from_params(params)
        qid = quant_from_filename(filename)
        file_rel = str(params.get("file", ""))
        params["quants"] = {qid: {"filename": filename or Path(file_rel).name, "file": file_rel}}
        params["active_quant"] = qid
        params.setdefault("display", name)
        new_models[name] = params
        preset_remap[name] = (name, qid)

    for repo, entries in repo_groups.items():
        # Pick canonical slug: shortest name without quant in slug, or generate
        entries.sort(key=lambda x: len(x[0]))
        primary_name, primary_params = entries[0]
        filename = _filename_from_params(primary_params)
        author = ""
        source = primary_params.get("source")
        if isinstance(source, dict):
            author = str(source.get("author") or "")
        slug = default_model_slug(repo, filename, author)
        if slug in new_models:
            slug = f"{slug}-{author}"[:48].strip("-")

        merged = dict(primary_params)
        merged["display"] = family_display(repo, filename)
        merged["quants"] = {}
        active_quant = quant_from_filename(filename)

        for old_name, params in entries:
            fn = _filename_from_params(params)
            qid = quant_from_filename(fn)
            file_rel = str(params.get("file", ""))
            merged["quants"][qid] = {"filename": fn, "file": file_rel}
            preset_remap[old_name] = (slug, qid)
            if old_name == primary_name:
                active_quant = qid

        merged["active_quant"] = active_quant
        entry = merged["quants"].get(active_quant, {})
        if isinstance(entry, dict):
            merged["file"] = entry.get("file", merged.get("file", ""))
            if isinstance(merged.get("source"), dict):
                merged["source"]["filename"] = entry.get("filename", _filename_from_params(merged))

        new_models[slug] = merged

        # Remap aliases
        for alias, target in list(aliases.items()):
            if target in preset_remap and preset_remap[target][0]:
                if target != slug and any(old == target for old, _ in entries):
                    aliases[alias] = slug

    data["models"] = new_models
    data["aliases"] = aliases
    return True, preset_remap


def _migrate_presets_file(path: Path, remap: dict[str, tuple[str, str]]) -> None:
    if not path.exists() or not remap:
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return

    active_raw = data.get("_active")
    new_active: dict[str, Any] = {}
    new_data: dict[str, Any] = {}

    if isinstance(active_raw, dict):
        if "model" in active_raw and "slot" in active_raw:
            old_model = str(active_raw.get("model", ""))
            slot = active_raw.get("slot")
            if old_model in remap:
                model, quant = remap[old_model]
                new_active.setdefault(model, {})[quant] = slot
        else:
            for old_model, slot in active_raw.items():
                if old_model.startswith("_"):
                    continue
                if old_model in remap:
                    model, quant = remap[old_model]
                    if isinstance(slot, int):
                        new_active.setdefault(model, {})[quant] = slot
                    elif isinstance(slot, dict):
                        new_active[model] = slot

    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, dict):
            continue
        # Already nested by quant?
        if value and all(isinstance(v, dict) and ("name" in v or "1" in v or "overrides" in str(v)) for v in value.values()):
            first = next(iter(value.values()))
            if isinstance(first, dict) and "overrides" in first:
                # flat: model -> slot -> preset
                if key in remap:
                    model, quant = remap[key]
                    new_data.setdefault(model, {})[quant] = value
                else:
                    new_data[key] = {"LOCAL": value}
                continue
        # Nested quant format already
        new_data[key] = value

    if new_active:
        new_data["_active"] = new_active
    path.write_text(json.dumps(new_data, indent=2, ensure_ascii=False) + "\n")


def model_author_size_line(cfg: ModelConfig) -> str:
    filename = _filename_from_params(cfg.params)
    return author_size_label(cfg.params, filename)
