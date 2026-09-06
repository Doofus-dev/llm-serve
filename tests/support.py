"""Shared test harness: temp repo with a fixture model, preset, and fake llama-server."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tui.app import LLMServeApp
from tui.data.preset_template import DEFAULT_PRESET_PARAMS
from tui.paths import AppPaths

MODEL_SLUG = "demo-model"
OTHER_SLUG = "other-model"
DISPLAY = "Demo"
OTHER_DISPLAY = "Other"
QUANT = "Q4_K_M"
ALIAS_DEFAULT = "default"
ALIAS_CODING = "coding"


def write_harness(root: Path) -> AppPaths:
    models_dir = root / "models"
    logs = root / "logs"
    llama_bin = root / "llama.cpp" / "build" / "bin"
    models_dir.mkdir(parents=True)
    logs.mkdir()
    llama_bin.mkdir(parents=True)
    server = llama_bin / "llama-server"
    server.write_text("#!/bin/sh\nexec sleep 60\n")
    server.chmod(0o755)
    gguf = models_dir / "demo" / "Demo-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF-fixture")

    models = {
        "models": {
            MODEL_SLUG: {
                "display": DISPLAY,
                "host": "127.0.0.1",
                "port": 8081,
                "notes": "fixture model",
                "file": "demo/Demo-Q4_K_M.gguf",
                "active_quant": QUANT,
                "total_layers": 32,
                "context_length": 32768,
                "quants": {
                    QUANT: {
                        "filename": "Demo-Q4_K_M.gguf",
                        "file": "demo/Demo-Q4_K_M.gguf",
                    }
                },
            },
            OTHER_SLUG: {
                "display": OTHER_DISPLAY,
                "host": "127.0.0.1",
                "port": 8082,
                "notes": "second fixture model",
                "file": "demo/Demo-Q4_K_M.gguf",
                "active_quant": QUANT,
                "total_layers": 32,
                "context_length": 32768,
                "quants": {
                    QUANT: {
                        "filename": "Demo-Q4_K_M.gguf",
                        "file": "demo/Demo-Q4_K_M.gguf",
                    }
                },
            },
        },
        "aliases": {
            ALIAS_DEFAULT: {"model": MODEL_SLUG},
            ALIAS_CODING: {"model": MODEL_SLUG, "quant": QUANT, "preset": 1},
        },
    }
    (root / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    presets = {
        "_active": {MODEL_SLUG: {QUANT: 1}, OTHER_SLUG: {QUANT: 1}},
        MODEL_SLUG: {
            QUANT: {
                "1": {"name": "default", "params": dict(DEFAULT_PRESET_PARAMS)},
            }
        },
        OTHER_SLUG: {
            QUANT: {
                "1": {"name": "default", "params": dict(DEFAULT_PRESET_PARAMS)},
            }
        },
    }
    (root / "presets.json").write_text(json.dumps(presets, indent=2) + "\n")
    return AppPaths(
        root=root,
        models_json=root / "models.json",
        presets_json=root / "presets.json",
        models_dir=models_dir,
        llama_dir=root / "llama.cpp",
        llama_server=server,
        log_dir=logs,
        log_file=logs / "llm-serve.log",
        pid_file=logs / ".llm-serve.pid",
        settings_json=root / "tui-settings.json",
        baselines_json=root / "tui-baselines.json",
    )


class Harness:
    """Owns a TemporaryDirectory for the life of a test."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.paths = write_harness(self.root)

    def app(self) -> LLMServeApp:
        return LLMServeApp(paths=self.paths)

    def cleanup(self) -> None:
        self._tmp.cleanup()
