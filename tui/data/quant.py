"""GGUF filename / repo parsing for model family, quant, and display labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


QUANT_RE = re.compile(
    r"(Q\d+_K(?:_[A-Z0-9]+)?|Q\d+_\d+|IQ\d+_[A-Z0-9]+|F16|BF16)",
    re.IGNORECASE,
)
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?[BMbm])")
FAMILY_RE = re.compile(r"([A-Za-z]+[\d]+(?:\.[\d]+)?)")


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
        quant = m.group(1).upper()

    return size, variant, quant


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
    if m := FAMILY_RE.search(text):
        return m.group(1)
    return None


def _prettify_family(raw: str) -> str:
    # Qwen3.8 -> Qwen 3.8, Llama3.1 -> Llama 3.1
    return re.sub(r"([a-zA-Z])(\d)", r"\1 \2", raw, count=1)


def author_size_label(params: dict, filename: str) -> str:
    """Second tree line: Bartowski 27B."""
    author = None
    source = params.get("source")
    if isinstance(source, dict) and source.get("author"):
        author = str(source["author"])
    file_path = str(params.get("file") or filename)
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


@dataclass(frozen=True)
class QuantEntry:
    quant_id: str
    filename: str
    file: str
    downloaded: bool
    size: int = 0


def quant_entry_from_file(quant_id: str, filename: str, file_rel: str, models_dir: Path) -> QuantEntry:
    path = models_dir / file_rel
    downloaded = path.is_file()
    size = path.stat().st_size if downloaded else 0
    return QuantEntry(
        quant_id=quant_id,
        filename=filename,
        file=file_rel,
        downloaded=downloaded,
        size=size,
    )
