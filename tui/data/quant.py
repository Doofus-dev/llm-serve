"""GGUF filename / repo parsing for model family, quant, and display labels."""

from __future__ import annotations

import re
from pathlib import Path


QUANT_RE = re.compile(
    r"(UD-)?(Q\d+_K(?:_[A-Z0-9]+)?|Q\d+_\d+|IQ\d+_[A-Z0-9]+|F16|BF16)",
    re.IGNORECASE,
)
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?[BMbm])")
VERSION_RE = re.compile(r"v\d+(?:\.\d+)?")
FAMILY_RE = re.compile(r"([A-Za-z]+(?:[-_]?[\d]+(?:\.[\d]+)?)+)")


def parse_gguf_filename(filename: str) -> tuple[str | None, str | None, str | None]:
    """Extract (size, variant, quant) from a GGUF filename."""
    stem = Path(filename).stem

    size = None
    if m := SIZE_RE.search(stem):
        size = m.group(1).upper()

    variant = None
    if re.search(r"[-_]UD[-_]", stem, re.I):
        variant = "UD"
    elif re.search(r"Instruct", stem, re.I):
        variant = "Instruct"
    elif re.search(r"Chat", stem, re.I):
        variant = "Chat"

    quant = None
    if m := QUANT_RE.search(stem):
        ud = m.group(1)
        q = m.group(2)
        if q:
            quant = (ud or "") + q.upper()

    return size, variant, quant


# Companion GGUF roles that ship next to a model's quants. A role other than
# ``model`` is never a catalog quant. See ``tui.data.capabilities.ADDON_KINDS``.
ROLE_MODEL = "model"
ROLE_PROJECTOR = "projector"  # mmproj-*.gguf — vision and/or audio via --mmproj
ROLE_MTP = "mtp"              # mtp-*.gguf — speculative draft via -md
ROLE_IMATRIX = "imatrix"      # *imatrix* — quant calibration, not a runtime file


def family_key(filename: str) -> str:
    """Stable id for sibling quants of the same GGUF, e.g. Qwen 3.5 9B."""
    stem = Path(filename).stem.lower()
    stem = re.sub(r"^mmproj[-_]", "", stem)
    stem = re.sub(r"^mtp[-_]", "", stem)
    stem = re.sub(r"^draft[-_]", "", stem)
    stem = re.sub(r"[-_]imatrix$", "", stem)
    stem = QUANT_RE.sub("", stem)
    return re.sub(r"[-_\.]+", "-", stem).strip("-")


def gguf_role(filename: str) -> str:
    """Classify a GGUF as a main quant or a known companion role.

    Split shards stay ``model``. LoRA adapters and generic ``draft-*`` small
    models ship as their own Hub repos, not companions here, so they are not
    roles. Audio/speech uses the same ``mmproj`` projector as vision.
    """
    name = Path(filename).name.lower()
    if name.startswith("mmproj"):
        return ROLE_PROJECTOR
    if "imatrix" in name:
        return ROLE_IMATRIX
    if name.startswith("mtp-"):
        return ROLE_MTP
    return ROLE_MODEL


def is_mmproj_gguf(filename: str) -> bool:
    """True for multimodal projector GGUFs (vision and/or audio via --mmproj)."""
    return gguf_role(filename) == ROLE_PROJECTOR


def is_sidecar_gguf(filename: str) -> bool:
    """True for non-model companions that must not enter the quant catalog.

    Projectors and imatrix files. MTP drafts use ``is_draft_gguf`` so they can
    still appear as labeled download rows.
    """
    return gguf_role(filename) in {ROLE_PROJECTOR, ROLE_IMATRIX}


def is_draft_gguf(filename: str) -> bool:
    """True for MTP speculative-decoding draft GGUFs (e.g. mtp-Model-Q8_0.gguf).

    Drafts are selectable downloads, not catalog quants: they must appear in
    the file picker but stay out of the quant catalog, VRAM calibration, and
    projector matching, where they would masquerade as the model's main quant.
    """
    return gguf_role(filename) == ROLE_MTP


def is_runtime_addon(filename: str) -> bool:
    """True for companions llama.cpp loads at launch (projector or MTP draft)."""
    return gguf_role(filename) in {ROLE_PROJECTOR, ROLE_MTP}


def is_hidden_sidecar(filename: str) -> bool:
    """True for companions that should not appear as Hub/picker rows (imatrix)."""
    return gguf_role(filename) == ROLE_IMATRIX


def quant_from_filename(filename: str) -> str:
    """Quant id for presets/catalog; falls back to stem or LOCAL."""
    _, _, quant = parse_gguf_filename(filename)
    if quant:
        return quant
    stem = Path(filename).stem
    return stem[:32] if stem else "LOCAL"


def family_token(repo_id: str, filename: str) -> str:
    """Short family token for slugs, e.g. qwen38-27b."""
    for text in (Path(filename).stem, repo_id.split("/")[-1]):
        family = _family_from_text(text)
        size, _, _ = parse_gguf_filename(filename)
        if family and size:
            fam_slug = re.sub(r"[^a-z0-9]+", "", family.lower())
            size_slug = size.lower()
            return f"{fam_slug}-{size_slug}"
        if family:
            return re.sub(r"[^a-z0-9]+", "", family.lower())
    return slugify(repo_id.split("/")[-1])


def family_display(repo_id: str, filename: str) -> str:
    """Human title for the tree root, e.g. Qwen 3.8."""
    for text in (Path(filename).stem, repo_id.split("/")[-1]):
        raw = _family_from_text(text)
        if raw:
            return _prettify_family(raw)
    return repo_id.split("/")[-1].replace("-", " ").replace("_", " ")


def _family_from_text(text: str) -> str | None:
    cleaned = QUANT_RE.sub("", text)
    cleaned = SIZE_RE.sub("", cleaned)
    cleaned = VERSION_RE.sub("", cleaned)
    if m := FAMILY_RE.search(cleaned):
        return m.group(1)
    return None


def _prettify_family(raw: str) -> str:
    # Qwen3.8 -> Qwen 3.8, Llama-3.1 -> Llama 3.1, Llama3.1 -> Llama 3.1
    return re.sub(r"([a-zA-Z])(?:-?)(\d)", r"\1 \2", raw, count=1)


def author_size_label(params: dict, filename: str) -> str:
    """Second tree line: Bartowski 27B."""
    author = None
    source = params.get("source")
    if isinstance(source, dict) and source.get("author"):
        author = str(source["author"])
    file_value = params.get("file")
    if isinstance(file_value, (list, tuple)) and file_value:
        file_path = str(file_value[0])
    else:
        file_path = str(file_value or filename)
    if not author and "/" in file_path:
        author = file_path.split("/", 1)[0]
    size, variant, _ = parse_gguf_filename(Path(filename).name)
    parts: list[str] = []
    if author:
        parts.append(author.title())
    if size:
        parts.append(size)
    if variant:
        parts.append(variant)
    return " ".join(parts) if parts else "local model"


def slugify(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "model"


def default_model_slug(repo_id: str, filename: str, author: str) -> str:
    base = family_token(repo_id, filename)
    # Keep slug unique per repo (author distinguishes same family from another uploader).
    author_bit = slugify(author, max_len=16)
    if author_bit and author_bit not in base:
        return f"{base}-{author_bit}"[:48].strip("-")
    return base[:48].strip("-")
