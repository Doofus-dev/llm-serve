"""Parse models.conf: extract register_model and register_alias entries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str
    params: dict[str, str] = field(default_factory=dict)

    @property
    def file(self) -> str:
        return self.params.get("file", "")

    @property
    def port(self) -> int:
        return int(self.params.get("port", "8081"))

    @property
    def host(self) -> str:
        return self.params.get("host", "127.0.0.1")

    @property
    def notes(self) -> str:
        return self.params.get("notes", "")


@dataclass
class Registry:
    models: dict[str, ModelConfig] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        # remove comments that aren't inside quotes
        out, in_q, i = [], False, 0
        while i < len(line):
            c = line[i]
            if c == '"':
                in_q = not in_q
                out.append(c)
            elif c == "#" and not in_q:
                break
            else:
                out.append(c)
            i += 1
        lines.append("".join(out))
    return "\n".join(lines)


def parse_models_conf(path: Path) -> Registry:
    reg = Registry()
    text = _strip_comments(path.read_text())

    for m in re.finditer(r'register_alias\s+"?([\w.-]+)"?\s+"?([\w.-]+)"?', text):
        reg.aliases[m.group(1)] = m.group(2)

    for m in re.finditer(
        r'register_model\s+"?([\w.-]+)"?\s*\\\n(.*?)(?=\n\s*\n|\Z)',
        text,
        re.DOTALL,
    ):
        name = m.group(1)
        body = m.group(2)
        params: dict[str, str] = {}
        for pm in re.finditer(r'([\w-]+)="([^"]*)"', body):
            params[pm.group(1)] = pm.group(2)
        for pm in re.finditer(r"([\w-]+)=([^\s\\]+)", body):
            if pm.group(1) not in params:
                params[pm.group(1)] = pm.group(2)
        reg.models[name] = ModelConfig(name=name, params=params)

    return reg
