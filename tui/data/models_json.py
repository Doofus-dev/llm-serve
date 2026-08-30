"""Parse and write models.json — TUI owns this file."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass
class Registry:
    models: dict[str, ModelConfig] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


def load_registry(path: Path) -> Registry:
    """Load models.json into a Registry."""
    if not path.exists():
        return Registry()
    
    data = json.loads(path.read_text())
    reg = Registry()
    
    for name, params in data.get("models", {}).items():
        reg.models[name] = ModelConfig(name=name, params=params)
    
    reg.aliases = data.get("aliases", {})
    return reg


def save_registry(path: Path, reg: Registry) -> None:
    """Save Registry back to models.json."""
    data = {
        "models": {name: cfg.params for name, cfg in reg.models.items()},
        "aliases": reg.aliases,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def delete_model(path: Path, name: str) -> None:
    """Delete a model from models.json."""
    reg = load_registry(path)
    if name in reg.models:
        del reg.models[name]
        save_registry(path, reg)


def create_model(path: Path, name: str, params: dict[str, Any]) -> None:
    """Create a new model in models.json."""
    reg = load_registry(path)
    reg.models[name] = ModelConfig(name=name, params=params)
    save_registry(path, reg)


def update_model(path: Path, name: str, params: dict[str, Any]) -> None:
    """Update an existing model in models.json."""
    reg = load_registry(path)
    if name in reg.models:
        reg.models[name].params = params
        save_registry(path, reg)
