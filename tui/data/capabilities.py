"""Vision and tool-call capability detection for Hub/quant UI and launch."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tui.data.hf import HubFile
from tui.data.quant import family_key, is_mmproj_gguf, is_sidecar_gguf

VISION_PIPELINE_TAGS = frozenset({
    "image-text-to-text",
    "image-to-text",
    "visual-question-answering",
    "any-to-any",
    "image-text-to-text-generation",
    "multimodal",
})

VISION_TAGS = frozenset({
    "vision",
    "multimodal",
    "image-text-to-text",
    "image-to-text",
    "mmproj",
    "clip",
    "visual-question-answering",
    "any-to-any",
})

TOOL_TAGS = frozenset({
    "tool-use",
    "tool_use",
    "tools",
    "function-calling",
    "function_calling",
    "functions",
    "tool-calling",
})

# Families llama.cpp routinely serves with --jinja tool parsers.
TOOL_FAMILY_RE = re.compile(
    r"(qwen|llama|mistral|mixtral|gemma|glm|deepseek|kimi|phi-?\d|"
    r"command-r|hermes|nemotron|yi-|internlm|ministral|granite|devstral)",
    re.IGNORECASE,
)

MMPROJ_PREF = (
    ("f16", 0),
    ("fp16", 0),
    ("bf16", 1),
    ("f32", 2),
    ("q8", 3),
)


@dataclass(frozen=True)
class ModelCapabilities:
    vision: bool = False
    tools: bool = False
    vision_source: str = ""
    tools_source: str = ""

    def merge_fallback(self, other: ModelCapabilities) -> ModelCapabilities:
        """Keep HF/API detections; fill gaps from file/family fallback."""
        vision = self.vision
        vision_source = self.vision_source
        if not vision and other.vision:
            vision = True
            vision_source = other.vision_source
        tools = self.tools
        tools_source = self.tools_source
        if not tools and other.tools:
            tools = True
            tools_source = other.tools_source
        return ModelCapabilities(
            vision=vision,
            tools=tools,
            vision_source=vision_source,
            tools_source=tools_source,
        )

    def to_json(self) -> dict[str, bool]:
        return {"vision": self.vision, "tools": self.tools}


def badge_markup(caps: ModelCapabilities) -> str:
    """Compact LM-Studio-style chips for tables and titles."""
    parts: list[str] = []
    if caps.vision:
        parts.append("[reverse cyan] vision [/]")
    if caps.tools:
        parts.append("[reverse magenta] tools [/]")
    return " ".join(parts) if parts else "[dim]—[/]"


def _lower_tags(tags: Iterable[object]) -> set[str]:
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def capabilities_from_hf(
    *,
    tags: Sequence[object] = (),
    pipeline_tag: str | None = None,
    card_data: Mapping[str, Any] | None = None,
) -> ModelCapabilities:
    """Primary detector: Hub tags / pipeline_tag / model card."""
    collected = _lower_tags(tags)
    pipe = (pipeline_tag or "").strip().lower()
    if isinstance(card_data, Mapping):
        collected |= _lower_tags(card_data.get("tags") or [])
        card_pipe = str(card_data.get("pipeline_tag") or "").strip().lower()
        if not pipe:
            pipe = card_pipe
        collected |= _lower_tags(card_data.get("tasks") or [])

    vision = pipe in VISION_PIPELINE_TAGS or bool(collected & VISION_TAGS)
    tools = bool(collected & TOOL_TAGS)
    return ModelCapabilities(
        vision=vision,
        tools=tools,
        vision_source="hf" if vision else "",
        tools_source="hf" if tools else "",
    )


def capabilities_from_payload(data: Mapping[str, Any]) -> ModelCapabilities:
    tags = data.get("tags") or ()
    if not isinstance(tags, (list, tuple)):
        tags = ()
    card = data.get("cardData") or data.get("card_data")
    if not isinstance(card, Mapping):
        card = None
    return capabilities_from_hf(
        tags=tags,
        pipeline_tag=str(data.get("pipeline_tag") or "") or None,
        card_data=card,
    )


def capabilities_from_files(
    files: Sequence[HubFile | str],
    *,
    repo_id: str = "",
) -> ModelCapabilities:
    """Offline fallback: mmproj sidecar => vision; known family => tools."""
    names = [
        item.path if isinstance(item, HubFile) else str(item)
        for item in files
    ]
    vision = any(is_mmproj_gguf(name) for name in names)
    haystack = " ".join([repo_id, *names])
    tools = bool(TOOL_FAMILY_RE.search(haystack))
    return ModelCapabilities(
        vision=vision,
        tools=tools,
        vision_source="files" if vision else "",
        tools_source="family" if tools else "",
    )


def capabilities_from_stored(params: Mapping[str, Any]) -> ModelCapabilities:
    raw = params.get("capabilities")
    if not isinstance(raw, dict):
        return ModelCapabilities()
    vision = bool(raw.get("vision"))
    tools = bool(raw.get("tools"))
    return ModelCapabilities(
        vision=vision,
        tools=tools,
        vision_source="stored" if vision else "",
        tools_source="stored" if tools else "",
    )


def resolve_capabilities(
    *,
    hf: ModelCapabilities | None = None,
    files: Sequence[HubFile | str] = (),
    repo_id: str = "",
    params: Mapping[str, Any] | None = None,
) -> ModelCapabilities:
    """HF/API first, then stored flags, then file/family presence."""
    caps = hf or ModelCapabilities()
    if params:
        caps = caps.merge_fallback(capabilities_from_stored(params))
    return caps.merge_fallback(capabilities_from_files(files, repo_id=repo_id))


def identity_capability_inputs(
    params: Mapping[str, Any] | None,
) -> tuple[list[str], str]:
    """Filenames and Hub repo used to resolve caps without a live file listing."""
    if not params:
        return [], ""
    files: list[str] = []
    repo_id = ""
    source = params.get("source")
    if isinstance(source, Mapping):
        repo_id = str(source.get("repo") or "")
    for key in ("mmproj", "file"):
        value = str(params.get(key) or "").strip()
        if value:
            files.append(value)
    quants = params.get("quants")
    if isinstance(quants, dict):
        for entry in quants.values():
            if not isinstance(entry, dict):
                continue
            for key in ("filename", "file"):
                value = str(entry.get(key) or "").strip()
                if value:
                    files.append(value)
    return files, repo_id


def nav_capability_icons(
    params: Mapping[str, Any] | None,
    models_dir: Path,
) -> tuple[str | None, bool]:
    """Main-list icon state: (vision_state, show_tools).

    ``vision_state`` is None when the model has no vision capability,
    otherwise ``missing`` (projector not on disk), ``off``, or ``on``.
    Missing stays ``missing`` even if ``vision_enabled`` is toggled.
    """
    files, repo_id = identity_capability_inputs(params)
    caps = resolve_capabilities(params=params, files=files, repo_id=repo_id)
    vision_state = None
    if caps.vision:
        _, ready = vision_projector_ready(params, models_dir, files)
        if not ready:
            vision_state = "missing"
        elif is_vision_enabled(params):
            vision_state = "on"
        else:
            vision_state = "off"
    return vision_state, bool(caps.tools)


def mmproj_files(files: Sequence[HubFile | str]) -> list[str]:
    names = [
        item.path if isinstance(item, HubFile) else str(item)
        for item in files
    ]
    return [name for name in names if is_mmproj_gguf(name)]


def pick_mmproj_filename(
    files: Sequence[HubFile | str],
    *,
    quant_filename: str | None = None,
) -> str | None:
    """Choose the projector that matches this quant's family, preferring f16."""
    candidates = mmproj_files(files)
    if not candidates:
        return None
    if quant_filename:
        fam = family_key(quant_filename)
        matched = [name for name in candidates if fam and fam in family_key(name)]
        if matched:
            candidates = matched

    def score(name: str) -> tuple[int, str]:
        lower = name.lower()
        rank = 9
        for token, value in MMPROJ_PREF:
            if token in lower:
                rank = value
                break
        return (rank, lower)

    return sorted(candidates, key=score)[0]


def _family_hint(params: Mapping[str, Any]) -> str | None:
    """Filename used to keep projector lookup on this model family."""
    for value in (
        Path(str(params.get("mmproj") or "")).name,
        Path(str(params.get("file") or "")).name,
    ):
        if value:
            return value
    quants = params.get("quants")
    if isinstance(quants, dict):
        active = str(params.get("active_quant") or "")
        entry = quants.get(active) if active else None
        if isinstance(entry, dict) and entry.get("filename"):
            return str(entry.get("filename"))
        for item in quants.values():
            if isinstance(item, dict) and item.get("filename"):
                return str(item.get("filename"))
    return None


def local_mmproj_path(
    params: Mapping[str, Any],
    models_dir: Path,
    *,
    extra_files: Sequence[str] = (),
) -> Path | None:
    """Resolve an on-disk mmproj for launch, including mis-filed catalog entries."""
    recorded = str(params.get("mmproj") or "").strip()
    if recorded:
        path = models_dir / recorded
        if path.is_file():
            return path

    candidates: list[Path] = []
    author = ""
    source = params.get("source")
    if isinstance(source, Mapping):
        author = str(source.get("author") or "")
    if author:
        folder = models_dir / author
        if folder.is_dir():
            candidates.extend(sorted(folder.glob("mmproj*.gguf")))

    quants = params.get("quants")
    if isinstance(quants, dict):
        for entry in quants.values():
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("filename") or "")
            file_rel = str(entry.get("file") or "")
            if is_mmproj_gguf(filename) and file_rel:
                path = models_dir / file_rel
                if path.is_file():
                    candidates.append(path)

    for name in extra_files:
        if is_mmproj_gguf(name):
            if author:
                path = models_dir / author / Path(name).name
            else:
                path = models_dir / Path(name).name
            if path.is_file():
                candidates.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    hint = _family_hint(params)
    if hint:
        fam = family_key(hint)
        matched = [path for path in unique if fam and fam in family_key(path.name)]
        unique = matched
    if not unique:
        return None
    picked = pick_mmproj_filename(
        [path.name for path in unique],
        quant_filename=hint,
    )
    if not picked:
        return unique[0]
    for path in unique:
        if path.name == picked:
            return path
    return unique[0]


def model_files_only(files: Sequence[HubFile]) -> list[HubFile]:
    return [item for item in files if not is_sidecar_gguf(item.path)]


def apply_mmproj_identity(
    params: dict[str, Any],
    *,
    author: str,
    mmproj_filename: str | None,
    caps: ModelCapabilities | None = None,
) -> None:
    if mmproj_filename:
        rel = f"{author}/{mmproj_filename}" if author else mmproj_filename
        params["mmproj"] = rel
    if caps is not None:
        params["capabilities"] = caps.to_json()


def is_vision_enabled(params: Mapping[str, Any] | None) -> bool:
    """User toggle. Defaults on so a present projector is used."""
    if not params:
        return True
    raw = params.get("vision_enabled")
    if raw is None:
        return True
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def mmproj_filename_for(
    params: Mapping[str, Any] | None = None,
    files: Sequence[object] = (),
    *,
    quant_filename: str | None = None,
) -> str | None:
    """Best projector name from recorded identity, then Hub/local file list."""
    recorded = Path(str((params or {}).get("mmproj") or "")).name
    if recorded and is_mmproj_gguf(recorded):
        return recorded
    return pick_mmproj_filename(files, quant_filename=quant_filename)


def vision_projector_ready(
    params: Mapping[str, Any] | None,
    models_dir: Path,
    files: Sequence[object] = (),
) -> tuple[str | None, bool]:
    """Return (mmproj filename, True if that file is on disk)."""
    name = mmproj_filename_for(params, files)
    path = local_mmproj_path(params or {}, models_dir, extra_files=[name] if name else ())
    if path is not None:
        return path.name, True
    return name, False
