"""Hub add-on taxonomy: projectors, MTP drafts, and capability badges."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tui.data.hf import HubFile
from tui.data.models_json import coerce_rel_paths, quant_entry_rel_paths
from tui.data.quant import (
    ROLE_IMATRIX,
    ROLE_MTP,
    ROLE_PROJECTOR,
    family_key,
    is_draft_gguf,
    is_mmproj_gguf,
    is_runtime_addon,
    is_sidecar_gguf,
)

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

AUDIO_PIPELINE_TAGS = frozenset({
    "audio-text-to-text",
    "automatic-speech-recognition",
    "text-to-audio",
    "text-to-speech",
    "audio-to-audio",
    "any-to-any",
})

AUDIO_TAGS = frozenset({
    "audio",
    "speech",
    "asr",
    "tts",
    "ultravox",
    "voxtral",
    "audio-text-to-text",
    "automatic-speech-recognition",
    "any-to-any",
})

# Same-repo companion GGUFs, not standalone adapter/draft Hub repos.
MTP_TAGS = frozenset({
    "mtp",
    "speculative-decoding",
    "fastmtp",
    "draft-mtp",
})

MTP_REPO_RE = re.compile(r"(^|[-_/])mtp($|[-_/])", re.IGNORECASE)
AUDIO_REPO_RE = re.compile(
    r"(ultravox|voxtral|whisper|qwen2-audio|qwen2\.5-omni|qwen2-5-omni|"
    r"[-_/]asr[-_/]|(^|[-_/])asr($|[-_/]))",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class AddonKind:
    """One llama.cpp companion that can ship next to GGUF quants.

    Inventory (same-repo files only — do not add theoretical slots):
    - projector (``mmproj-*.gguf``, ``--mmproj``): vision and/or audio. One
      file, one flag; llama.cpp cannot load two projectors.
    - mtp (``mtp-*.gguf``, ``-md --spec-type draft-mtp``): speculative draft.
    - imatrix (``*imatrix*``): quant calibration. Never a runtime download.
    LoRA adapters (``--lora``) and generic ``draft-*`` small models live in
    their own Hub repos, not as companions of a base GGUF. TTS vocoders are
    not published as separate GGUFs; audio goes through mmproj.
    """

    id: str
    role: str
    label: str
    launch_flag: str | None
    extra_flags: tuple[str, ...]
    file_key: str
    enabled_key: str
    default_enabled: bool
    togglable: bool
    downloadable: bool
    show_in_file_table: bool


ADDON_KINDS: tuple[AddonKind, ...] = (
    AddonKind(
        id="projector",
        role=ROLE_PROJECTOR,
        label="vision",
        launch_flag="--mmproj",
        extra_flags=(),
        file_key="mmproj",
        enabled_key="vision_enabled",
        default_enabled=True,
        togglable=True,
        downloadable=True,
        show_in_file_table=True,
    ),
    AddonKind(
        id="mtp",
        role=ROLE_MTP,
        label="MTP",
        launch_flag="-md",
        extra_flags=("--spec-type", "draft-mtp"),
        file_key="mtp_file",
        enabled_key="mtp_enabled",
        default_enabled=False,
        togglable=True,
        downloadable=True,
        show_in_file_table=True,
    ),
    AddonKind(
        id="imatrix",
        role=ROLE_IMATRIX,
        label="imatrix",
        launch_flag=None,
        extra_flags=(),
        file_key="",
        enabled_key="",
        default_enabled=False,
        togglable=False,
        downloadable=False,
        show_in_file_table=False,
    ),
)

ADDON_BY_ID = {kind.id: kind for kind in ADDON_KINDS}
ADDON_BY_ROLE = {kind.role: kind for kind in ADDON_KINDS}
RUNTIME_ADDONS = tuple(kind for kind in ADDON_KINDS if kind.downloadable)

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
    audio: bool = False
    tools: bool = False
    mtp: bool = False
    vision_source: str = ""
    audio_source: str = ""
    tools_source: str = ""
    mtp_source: str = ""

    def merge_fallback(self, other: ModelCapabilities) -> ModelCapabilities:
        """Keep HF/API detections; fill gaps from file/family fallback.

        An audio-only Hub tag means the mmproj is the speech projector, so a
        file-based vision fallback must not override that.
        """
        audio = self.audio
        audio_source = self.audio_source
        if not audio and other.audio:
            audio = True
            audio_source = other.audio_source
        vision = self.vision
        vision_source = self.vision_source
        if not vision and other.vision and not (audio and not self.vision):
            vision = True
            vision_source = other.vision_source
        tools = self.tools
        tools_source = self.tools_source
        if not tools and other.tools:
            tools = True
            tools_source = other.tools_source
        mtp = self.mtp
        mtp_source = self.mtp_source
        if not mtp and other.mtp:
            mtp = True
            mtp_source = other.mtp_source
        return ModelCapabilities(
            vision=vision,
            audio=audio,
            tools=tools,
            mtp=mtp,
            vision_source=vision_source,
            audio_source=audio_source,
            tools_source=tools_source,
            mtp_source=mtp_source,
        )

    def to_json(self) -> dict[str, bool]:
        return {
            "vision": self.vision,
            "audio": self.audio,
            "tools": self.tools,
            "mtp": self.mtp,
        }


def badge_markup(caps: ModelCapabilities) -> str:
    """Compact LM-Studio-style chips for tables and titles."""
    parts: list[str] = []
    if caps.vision:
        parts.append("[reverse cyan] vision [/]")
    if caps.audio:
        parts.append("[reverse blue] audio [/]")
    if caps.mtp:
        parts.append("[reverse yellow] mtp [/]")
    if caps.tools:
        parts.append("[reverse magenta] tools [/]")
    return " ".join(parts) if parts else "[dim]—[/]"


def addon_row_label(role: str, caps: ModelCapabilities | None = None) -> str:
    """File-table prefix for a companion GGUF (MTP, vision, audio)."""
    kind = ADDON_BY_ROLE.get(role)
    if role == ROLE_PROJECTOR:
        if caps and caps.audio and not caps.vision:
            return "audio"
        return "vision"
    if kind is not None:
        return kind.label
    return role


def projector_control_label(caps: ModelCapabilities) -> str:
    """Editor label for the single --mmproj slot."""
    if caps.audio and not caps.vision:
        return "Audio"
    return "Vision"


def _lower_tags(tags: Iterable[object]) -> set[str]:
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def capabilities_from_hf(
    *,
    tags: Sequence[object] = (),
    pipeline_tag: str | None = None,
    card_data: Mapping[str, Any] | None = None,
    repo_id: str = "",
) -> ModelCapabilities:
    """Primary detector: Hub tags / pipeline_tag / model card / repo id."""
    collected = _lower_tags(tags)
    pipe = (pipeline_tag or "").strip().lower()
    if isinstance(card_data, Mapping):
        collected |= _lower_tags(card_data.get("tags") or [])
        card_pipe = str(card_data.get("pipeline_tag") or "").strip().lower()
        if not pipe:
            pipe = card_pipe
        collected |= _lower_tags(card_data.get("tasks") or [])

    haystack = repo_id
    vision = pipe in VISION_PIPELINE_TAGS or bool(collected & VISION_TAGS)
    audio = (
        pipe in AUDIO_PIPELINE_TAGS
        or bool(collected & AUDIO_TAGS)
        or bool(AUDIO_REPO_RE.search(haystack))
    )
    tools = bool(collected & TOOL_TAGS)
    mtp = bool(collected & MTP_TAGS) or bool(MTP_REPO_RE.search(haystack))
    return ModelCapabilities(
        vision=vision,
        audio=audio,
        tools=tools,
        mtp=mtp,
        vision_source="hf" if vision else "",
        audio_source="hf" if audio else "",
        tools_source="hf" if tools else "",
        mtp_source="hf" if mtp else "",
    )


def capabilities_from_payload(data: Mapping[str, Any]) -> ModelCapabilities:
    tags = data.get("tags") or ()
    if not isinstance(tags, (list, tuple)):
        tags = ()
    card = data.get("cardData") or data.get("card_data")
    if not isinstance(card, Mapping):
        card = None
    sibling_files = _payload_filenames(data)
    caps = capabilities_from_hf(
        tags=tags,
        pipeline_tag=str(data.get("pipeline_tag") or "") or None,
        card_data=card,
        repo_id=str(data.get("id") or ""),
    )
    if sibling_files:
        caps = caps.merge_fallback(
            capabilities_from_files(sibling_files, repo_id=str(data.get("id") or ""))
        )
    return caps


def _payload_filenames(data: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("siblings", "files", "filenames"):
        raw = data.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        for item in raw:
            if isinstance(item, str) and item.lower().endswith(".gguf"):
                names.append(item)
            elif isinstance(item, Mapping):
                name = str(item.get("rfilename") or item.get("path") or "")
                if name.lower().endswith(".gguf"):
                    names.append(name)
    return names


def capabilities_from_files(
    files: Sequence[HubFile | str],
    *,
    repo_id: str = "",
) -> ModelCapabilities:
    """Offline fallback: mmproj => vision; mtp-* => MTP; known family => tools."""
    names = [
        item.path if isinstance(item, HubFile) else str(item)
        for item in files
    ]
    projector = any(is_mmproj_gguf(name) for name in names)
    mtp = any(is_draft_gguf(name) for name in names)
    haystack = " ".join([repo_id, *names])
    audio = bool(AUDIO_REPO_RE.search(haystack))
    vision = projector and not audio
    tools = bool(TOOL_FAMILY_RE.search(haystack))
    return ModelCapabilities(
        vision=vision,
        audio=audio,
        tools=tools,
        mtp=mtp,
        vision_source="files" if vision else "",
        audio_source="files" if audio else "",
        tools_source="family" if tools else "",
        mtp_source="files" if mtp else "",
    )


def capabilities_from_stored(params: Mapping[str, Any]) -> ModelCapabilities:
    raw = params.get("capabilities")
    if not isinstance(raw, dict):
        return ModelCapabilities()
    vision = bool(raw.get("vision"))
    audio = bool(raw.get("audio"))
    tools = bool(raw.get("tools"))
    mtp = bool(raw.get("mtp"))
    return ModelCapabilities(
        vision=vision,
        audio=audio,
        tools=tools,
        mtp=mtp,
        vision_source="stored" if vision else "",
        audio_source="stored" if audio else "",
        tools_source="stored" if tools else "",
        mtp_source="stored" if mtp else "",
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
    mmproj = str(params.get("mmproj") or "").strip()
    if mmproj:
        files.append(mmproj)
    mtp_file = str(params.get("mtp_file") or "").strip()
    if mtp_file:
        files.append(mtp_file)
    files.extend(coerce_rel_paths(params.get("files")) or coerce_rel_paths(params.get("file")))
    quants = params.get("quants")
    if isinstance(quants, dict):
        for entry in quants.values():
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("filename") or "").strip()
            if filename:
                files.append(filename)
            files.extend(quant_entry_rel_paths(entry))
    return files, repo_id


def nav_capability_icons(
    params: Mapping[str, Any] | None,
    models_dir: Path,
) -> tuple[str | None, bool, str | None]:
    """Main-list icon state: (vision_state, show_tools, mtp_state).

    ``vision_state`` / ``mtp_state`` are None when the model has no such
    sidecar. Otherwise ``missing`` (file not on disk), ``off``, or ``on``.
    Missing stays ``missing`` even if the toggle is flipped.
    """
    files, repo_id = identity_capability_inputs(params)
    caps = resolve_capabilities(params=params, files=files, repo_id=repo_id)
    vision_state = None
    if caps.vision or caps.audio:
        _, ready = vision_projector_ready(params, models_dir, files)
        if not ready:
            vision_state = "missing"
        elif is_vision_enabled(params):
            vision_state = "on"
        else:
            vision_state = "off"
    mtp_state = None
    if caps.mtp:
        _, ready = mtp_draft_ready(params, models_dir, files)
        if not ready:
            mtp_state = "missing"
        elif is_mtp_enabled(params):
            mtp_state = "on"
        else:
            mtp_state = "off"
    return vision_state, bool(caps.tools), mtp_state


def mmproj_files(files: Sequence[HubFile | str]) -> list[str]:
    names = [
        item.path if isinstance(item, HubFile) else str(item)
        for item in files
        if not is_draft_gguf(item.path if isinstance(item, HubFile) else str(item))
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
    file_rels = coerce_rel_paths(params.get("files")) or coerce_rel_paths(params.get("file"))
    for value in (
        Path(str(params.get("mmproj") or "")).name,
        Path(file_rels[0]).name if file_rels else "",
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
            for file_rel in quant_entry_rel_paths(entry):
                if is_mmproj_gguf(filename or Path(file_rel).name) and file_rel:
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
    return [item for item in files if not is_sidecar_gguf(item.path) and not is_draft_gguf(item.path)]


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

def mtp_files(files: Sequence[HubFile | str]) -> list[str]:
    names = [
        item.path if isinstance(item, HubFile) else str(item)
        for item in files
    ]
    return [name for name in names if is_draft_gguf(name)]


def pick_mtp_filename(
    files: Sequence[HubFile | str],
    *,
    quant_filename: str | None = None,
) -> str | None:
    """Choose an MTP draft GGUF. Prefer self-contained Q8_0 over shared heads."""
    candidates = mtp_files(files)
    if not candidates:
        return None
    if quant_filename:
        fam = family_key(quant_filename)
        matched = [name for name in candidates if fam and fam in family_key(name)]
        if matched:
            candidates = matched

    def score(name: str) -> tuple:
        lower = Path(name).name.lower()
        shared = 1 if "shared" in lower else 0
        if "q8_0" in lower:
            qrank = 0
        elif "q4" in lower:
            qrank = 1
        elif "bf16" in lower or "f16" in lower:
            qrank = 2
        else:
            qrank = 3
        return (shared, qrank, name.count("/"), lower)

    return sorted(candidates, key=score)[0]


def apply_mtp_identity(
    params: dict[str, Any],
    *,
    author: str,
    mtp_filename: str | None,
    caps: ModelCapabilities | None = None,
) -> None:
    if mtp_filename:
        name = mtp_filename
        if author and name.startswith(f"{author}/"):
            rel = name
        elif author:
            rel = f"{author}/{name}"
        else:
            rel = name
        params["mtp_file"] = rel
    if caps is not None:
        stored = dict(params.get("capabilities") or {})
        stored.update(caps.to_json())
        params["capabilities"] = stored


def local_mtp_path(
    params: Mapping[str, Any],
    models_dir: Path,
    *,
    extra_files: Sequence[str] = (),
) -> Path | None:
    """Resolve an on-disk MTP draft GGUF."""
    recorded = str(params.get("mtp_file") or "").strip()
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
            candidates.extend(sorted(folder.rglob("mtp-*.gguf")))

    for name in extra_files:
        if is_draft_gguf(name):
            path = models_dir / author / name if author else models_dir / name
            if author:
                path = models_dir / author / name
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
        if matched:
            unique = matched
    if not unique:
        return None
    picked = pick_mtp_filename([_mtp_rel(path, models_dir, author) for path in unique], quant_filename=hint)
    if not picked:
        return unique[0]
    for path in unique:
        rel = _mtp_rel(path, models_dir, author)
        if rel == picked or path.name == Path(picked).name:
            return path
    return unique[0]


def _mtp_rel(path: Path, models_dir: Path, author: str) -> str:
    try:
        return str(path.relative_to(models_dir / author if author else models_dir))
    except ValueError:
        return path.name


def mtp_filename_for(
    params: Mapping[str, Any] | None = None,
    files: Sequence[object] = (),
    *,
    quant_filename: str | None = None,
) -> str | None:
    recorded = str((params or {}).get("mtp_file") or "").strip()
    if recorded and is_draft_gguf(recorded):
        # keep Hub-relative path (may include MTP/)
        source = (params or {}).get("source")
        author = ""
        if isinstance(source, Mapping):
            author = str(source.get("author") or "")
        name = recorded
        if author and name.startswith(f"{author}/"):
            name = name[len(author) + 1 :]
        return name
    return pick_mtp_filename(files, quant_filename=quant_filename)


def mtp_draft_ready(
    params: Mapping[str, Any] | None,
    models_dir: Path,
    files: Sequence[object] = (),
) -> tuple[str | None, bool]:
    name = mtp_filename_for(params, files)
    path = local_mtp_path(params or {}, models_dir, extra_files=[name] if name else ())
    if path is not None:
        return (str(path.relative_to(models_dir)) if models_dir in path.parents or path.parent == models_dir else path.name), True
    return name, False


def is_mtp_enabled(params: Mapping[str, Any] | None) -> bool:
    """Identity toggle or preset ``mtp`` flag."""
    if not params:
        return False
    for key in ("mtp_enabled", "mtp"):
        raw = params.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool) and raw:
            return True
        if isinstance(raw, (int, float)) and raw:
            return True
        if str(raw).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def pick_addon_filename(
    kind_id: str,
    files: Sequence[HubFile | str],
    *,
    quant_filename: str | None = None,
) -> str | None:
    """Pick the companion file for a runtime add-on kind."""
    if kind_id == "projector":
        return pick_mmproj_filename(files, quant_filename=quant_filename)
    if kind_id == "mtp":
        return pick_mtp_filename(files, quant_filename=quant_filename)
    return None


def extra_download_files(
    files: Sequence[HubFile | str],
    *,
    quant_filename: str,
    models_dir: Path,
    author: str = "",
    params: Mapping[str, Any] | None = None,
) -> list[str]:
    """Companions to fetch with a main quant (projector and/or MTP draft).

    Skip when the selected row is itself an add-on. Skip files already on disk.
    """
    if is_runtime_addon(quant_filename) or is_sidecar_gguf(quant_filename):
        return []
    extras: list[str] = []
    for kind in RUNTIME_ADDONS:
        name = pick_addon_filename(kind.id, files, quant_filename=quant_filename)
        if not name and params:
            name = str(params.get(kind.file_key) or "").strip() or None
        if not name:
            continue
        if author and name.startswith(f"{author}/"):
            name = name[len(author) + 1 :]
        local = models_dir / author / name if author else models_dir / name
        if local.is_file():
            continue
        if name not in extras:
            extras.append(name)
    return extras

