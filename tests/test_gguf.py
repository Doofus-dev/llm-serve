"""Tests for GGUF header architecture parsing."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.data.gguf import (
    ARRAY,
    STRING,
    UINT32,
    apply_architecture_from_gguf,
    read_gguf_architecture,
)
from tui.data.models_json import load_registry, sync_gguf_architecture


def _pack_string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_kv_string(key: str, value: str) -> bytes:
    return _pack_string(key) + struct.pack("<I", STRING) + _pack_string(value)


def _pack_kv_u32(key: str, value: int) -> bytes:
    return _pack_string(key) + struct.pack("<I", UINT32) + struct.pack("<I", value)


def _pack_kv_string_array(key: str, values: list[str]) -> bytes:
    body = struct.pack("<I", STRING) + struct.pack("<Q", len(values))
    for item in values:
        body += _pack_string(item)
    return _pack_string(key) + struct.pack("<I", ARRAY) + body


def write_gguf(path: Path, kvs: list[bytes]) -> None:
    payload = b"".join(kvs)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    path.write_bytes(header + payload)


class GgufArchitectureTests(unittest.TestCase):
    def test_reads_block_count_for_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            write_gguf(
                path,
                [
                    _pack_kv_string("general.architecture", "qwen3"),
                    _pack_kv_u32("qwen3.context_length", 262144),
                    _pack_kv_u32("qwen3.block_count", 64),
                ],
            )
            info = read_gguf_architecture(path)
            self.assertEqual(info.architecture, "qwen3")
            self.assertEqual(info.block_count, 64)
            self.assertEqual(info.context_length, 262144)

    def test_skips_tokenizer_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            write_gguf(
                path,
                [
                    _pack_kv_string("general.architecture", "llama"),
                    _pack_kv_string_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
                    _pack_kv_u32("llama.block_count", 32),
                ],
            )
            self.assertEqual(read_gguf_architecture(path).block_count, 32)

    def test_missing_file_is_empty(self) -> None:
        info = read_gguf_architecture(Path("/no/such/model.gguf"))
        self.assertIsNone(info.block_count)

    def test_apply_overwrites_cloned_layer_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            write_gguf(
                path,
                [
                    _pack_kv_string("general.architecture", "qwen2"),
                    _pack_kv_u32("qwen2.block_count", 28),
                ],
            )
            params = {"total_layers": 64, "ctx": 32768}
            info = apply_architecture_from_gguf(params, path, replace_cloned=True)
            self.assertEqual(info.block_count, 28)
            self.assertEqual(params["total_layers"], 28)

    def test_apply_drops_clone_when_header_unreadable(self) -> None:
        params = {"total_layers": 28, "ctx": 4096}
        apply_architecture_from_gguf(params, Path("/no/such.gguf"), replace_cloned=True)
        self.assertNotIn("total_layers", params)
        self.assertEqual(params["ctx"], 4096)

    def test_local_qwen25_gguf_if_present(self) -> None:
        path = Path(__file__).parent.parent / "models" / "Qwen2.5-7B-Instruct-Q8_0.gguf"
        if not path.is_file():
            self.skipTest("local Qwen2.5 GGUF not present")
        info = read_gguf_architecture(path)
        self.assertEqual(info.architecture, "qwen2")
        self.assertEqual(info.block_count, 28)

    def test_sync_rewrites_stale_total_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir()
            gguf = models_dir / "demo.gguf"
            write_gguf(
                gguf,
                [
                    _pack_kv_string("general.architecture", "qwen35"),
                    _pack_kv_u32("qwen35.block_count", 65),
                ],
            )
            path = root / "models.json"
            path.write_text(
                '{"models": {"demo": {"file": "demo.gguf", "total_layers": 28}}, "aliases": {}}'
            )
            changes = sync_gguf_architecture(path, models_dir)
            self.assertEqual(changes, [("demo", 28, 65)])
            self.assertEqual(load_registry(path).models["demo"].params["total_layers"], 65)


if __name__ == "__main__":
    unittest.main()
