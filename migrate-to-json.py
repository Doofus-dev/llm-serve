#!/usr/bin/env python3
"""Migrate models.conf (bash) → models.json (JSON). Verify before deleting."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_CONF = SCRIPT_DIR / "models.conf"
MODELS_JSON = SCRIPT_DIR / "models.json"


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        out, in_q = [], False
        for c in line:
            if c == '"':
                in_q = not in_q
                out.append(c)
            elif c == "#" and not in_q:
                break
            else:
                out.append(c)
        lines.append("".join(out))
    return "\n".join(lines)


def parse_models_conf(path: Path) -> dict:
    text = strip_comments(path.read_text())
    
    # Parse aliases
    aliases = {}
    for m in re.finditer(r'register_alias\s+"?([\w.-]+)"?\s+"?([\w.-]+)"?', text):
        aliases[m.group(1)] = m.group(2)
    
    # Parse models
    models = {}
    for m in re.finditer(
        r'register_model\s+"?([\w.-]+)"?\s*\\\n(.*?)(?=\n\s*\n|\Z)',
        text,
        re.DOTALL,
    ):
        name = m.group(1)
        body = m.group(2)
        params = {}
        
        # Parse key="value" and key=value
        for pm in re.finditer(r'([\w-]+)="([^"]*)"', body):
            params[pm.group(1)] = pm.group(2)
        for pm in re.finditer(r"([\w-]+)=([^\s\\]+)", body):
            if pm.group(1) not in params:
                params[pm.group(1)] = pm.group(2)
        
        # Convert numeric strings to numbers where appropriate
        for key, val in params.items():
            if val and val.lstrip('-').replace('.', '').isdigit():
                try:
                    if '.' in val:
                        params[key] = float(val)
                    else:
                        params[key] = int(val)
                except ValueError:
                    pass
        
        models[name] = params
    
    return {"models": models, "aliases": aliases}


def main():
    if not MODELS_CONF.exists():
        print(f"Error: {MODELS_CONF} not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Reading {MODELS_CONF}...")
    data = parse_models_conf(MODELS_CONF)
    
    print(f"\nParsed {len(data['models'])} models:")
    for name in data['models']:
        params = data['models'][name]
        print(f"  - {name} ({len(params)} params)")
        print(f"    file: {params.get('file', '?')}")
    
    print(f"\nParsed {len(data['aliases'])} aliases:")
    for alias, target in data['aliases'].items():
        print(f"  - {alias} → {target}")
    
    # Write JSON
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Generated models.json:")
    print('='*60)
    print(json_str)
    print('='*60)
    
    # Confirm
    if "--yes" in sys.argv:
        response = "y"
    else:
        response = input("\nWrite this to models.json? [y/N] ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    MODELS_JSON.write_text(json_str + '\n')
    print(f"\n✓ Wrote {MODELS_JSON}")
    
    # Verify
    print("\nVerifying...")
    loaded = json.loads(MODELS_JSON.read_text())
    assert loaded == data, "Verification failed!"
    print("✓ Verification passed")
    
    # Offer to delete models.conf
    if "--yes" in sys.argv:
        response = "y"
    else:
        response = input(f"\nDelete {MODELS_CONF}? [y/N] ")
    if response.lower() == 'y':
        MODELS_CONF.unlink()
        print(f"✓ Deleted {MODELS_CONF}")
    else:
        print(f"Kept {MODELS_CONF} (you can delete it manually later)")


if __name__ == "__main__":
    main()
