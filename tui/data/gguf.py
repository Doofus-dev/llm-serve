"""Read architecture facts from a GGUF header.

Layer count is `{architecture}.block_count` in the file metadata — not in the
Hub `expand=gguf` catalog blob (that only has architecture, context_length,
and size). This reads the header only; tokenizer tables are skipped.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)


@dataclass(frozen=True)
class GgufArchitecture:
    architecture: str | None = None
    block_count: int | None = None
    context_length: int | None = None


def read_gguf_architecture(path: Path | None) -> GgufArchitecture:
    """Return architecture / layer count from a local GGUF, or empty on failure."""
    if path is None or not path.is_file():
        return GgufArchitecture()
    try:
        with path.open("rb") as handle:
            return _parse_header(handle)
    except (OSError, struct.error, ValueError, UnicodeDecodeError):
        return GgufArchitecture()


def apply_architecture_from_gguf(
    params: dict,
    path: Path | None,
    *,
    replace_cloned: bool = False,
) -> GgufArchitecture:
    """Copy GGUF block_count onto total_layers when the header can be read."""
    info = read_gguf_architecture(path)
    if info.block_count is not None:
        params["total_layers"] = info.block_count
    elif replace_cloned:
        params.pop("total_layers", None)
    return info


def _parse_header(handle: BinaryIO) -> GgufArchitecture:
    if handle.read(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = _u32(handle)
    if version < 1 or version > 3:
        raise ValueError(f"unsupported GGUF version {version}")
    if version == 1:
        _u32(handle)  # tensor_count
        kv_count = _u32(handle)
    else:
        _u64(handle)  # tensor_count
        kv_count = _u64(handle)

    metadata: dict[str, object] = {}
    for _ in range(kv_count):
        key = _string(handle)
        value = _read_value(handle)
        if key.startswith("tokenizer."):
            continue
        metadata[key] = value
        arch = metadata.get("general.architecture")
        if isinstance(arch, str) and isinstance(metadata.get(f"{arch}.block_count"), int):
            break

    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str):
        architecture = None
    block_count = _as_int(metadata.get(f"{architecture}.block_count") if architecture else None)
    context_length = _as_int(metadata.get(f"{architecture}.context_length") if architecture else None)
    return GgufArchitecture(
        architecture=architecture,
        block_count=block_count,
        context_length=context_length,
    )


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _read_value(handle: BinaryIO) -> object:
    vtype = _u32(handle)
    return _read_typed(handle, vtype)


def _read_typed(handle: BinaryIO, vtype: int) -> object:
    if vtype == UINT8:
        return struct.unpack("B", handle.read(1))[0]
    if vtype == INT8:
        return struct.unpack("b", handle.read(1))[0]
    if vtype == UINT16:
        return struct.unpack("<H", handle.read(2))[0]
    if vtype == INT16:
        return struct.unpack("<h", handle.read(2))[0]
    if vtype == UINT32:
        return _u32(handle)
    if vtype == INT32:
        return struct.unpack("<i", handle.read(4))[0]
    if vtype == FLOAT32:
        return struct.unpack("<f", handle.read(4))[0]
    if vtype == BOOL:
        return handle.read(1) != b"\x00"
    if vtype == STRING:
        return _string(handle)
    if vtype == ARRAY:
        return _skip_array(handle)
    if vtype == UINT64:
        return _u64(handle)
    if vtype == INT64:
        return struct.unpack("<q", handle.read(8))[0]
    if vtype == FLOAT64:
        return struct.unpack("<d", handle.read(8))[0]
    raise ValueError(f"unknown GGUF type {vtype}")


def _skip_array(handle: BinaryIO) -> None:
    elem_type = _u32(handle)
    count = _u64(handle)
    for _ in range(count):
        _read_typed(handle, elem_type)
    return None


def _string(handle: BinaryIO) -> str:
    length = _u64(handle)
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("truncated GGUF string")
    return raw.decode("utf-8", errors="replace")


def _u32(handle: BinaryIO) -> int:
    raw = handle.read(4)
    if len(raw) != 4:
        raise ValueError("truncated GGUF")
    return struct.unpack("<I", raw)[0]


def _u64(handle: BinaryIO) -> int:
    raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError("truncated GGUF")
    return struct.unpack("<Q", raw)[0]
