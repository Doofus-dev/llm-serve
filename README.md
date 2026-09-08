# llm-serve

A llama.cpp launcher with a Textual TUI. Register GGUF models in JSON, pick a quant and preset, and start one `llama-server` — from the TUI or the command line.

## What it does

- **Model registry** — Profiles in `models.json` (display name, port, Hugging Face source, quants). Runtime knobs live in `presets.json`.
- **Interactive TUI** — Browse models and aliases, edit presets, download GGUFs from Hugging Face, and watch live decode speed and GPU use.
- **Aliases** — `llm-serve coding` can follow a model’s current quant, or pin a quant and preset.
- **Presets** — Up to five numbered slots per quant (`gpu_layers`, context, KV cache type, sampling, MTP, reasoning, …).
- **Hub** — Search GGUF repos, compare estimated vs measured VRAM and tok/s, queue downloads.
- **One server** — A single tracked `llama-server`. Launch refuses if one is already running.
- **Remote** — `--remote` or **R** in the TUI binds `0.0.0.0` so other devices on a trusted LAN or VPN can connect. There is no authentication.
- **Setup** — `./setup.sh` installs a project `.venv`, clones and builds llama.cpp (CUDA / ROCm / CPU), and puts `llm-serve` on `PATH`.

## Prerequisites

- **Linux** with pacman, apt, dnf, or zypper (for `setup.sh` package installs)
- **Bash 4.4+**
- **Python 3.9+** with `venv` (Ubuntu/Debian: `python3-venv`)
- **NVIDIA** (`nvidia-smi`) or **AMD** (`lspci`) GPU optional — otherwise a CPU build
- **`hf` CLI** optional, for Hub browse/download: `curl -LsSf https://hf.co/cli/install.sh | bash`

macOS and Windows are not supported by `setup.sh`. The launcher can work on macOS if you install Bash 4.4+ and build llama.cpp yourself.

## Quick start

```bash
git clone https://github.com/Doofus-dev/llm-serve.git
cd llm-serve

./setup.sh              # Auto-detects GPU (NVIDIA→CUDA, AMD→ROCm, else CPU)
./setup.sh --cpu        # Force CPU-only
./setup.sh --cuda       # Force CUDA
./setup.sh --rocm       # Force ROCm
```

Then:

```bash
# TUI — download a GGUF with H, or drop files in models/<author>/
llm-serve

# Or launch a profile / alias from any directory (after setup)
llm-serve my-model
llm-serve coding --remote
```

Default bind is `127.0.0.1:8081` (see `models.json`):

```bash
curl http://127.0.0.1:8081/v1/models
```

## Command line

```
llm-serve                      Open the TUI
llm-serve --help               Models, aliases, and this command list
llm-serve list                 Same as --help
llm-serve <model>              Start in the background
llm-serve <model> --live       Foreground with live logs
llm-serve <model> --dry-run    Print the llama-server command; do not start
llm-serve <model> --remote     Bind 0.0.0.0
llm-serve status               PID, port, local/remote, last log lines
llm-serve stop                 Stop the tracked server
llm-serve stop <model>         Stop only if that model is running
llm-serve update               Pull and rebuild llama.cpp
llm-serve update --yes         Same, no confirm
```

`<model>` matches an alias name, a model slug, or a display name.

### Environment overrides

Common launch knobs (also `LOG_VERBOSITY`):

```bash
GPU_LAYERS=50 CONTEXT_SIZE=32768 PORT=9000 llm-serve my-model
MODEL_DIR=/mnt/models MODEL_PATH=/mnt/models/foo.gguf llm-serve my-model
LLAMA_DIR=/opt/llama.cpp llm-serve my-model
```

`PORT`, `HOST`, `CONTEXT_SIZE`, `GPU_LAYERS`, `THREADS`, `UBATCH`, `PARALLEL`, `N_PREDICT`, `CACHE_TYPE_K`, `CACHE_TYPE_V`, `BATCH_SIZE`, `DEFRAG_THOLD`, `TEMP`, `TOP_P`, `SEED`, `ENABLE_MTP`, `N_CPU_MOE`, `MODEL_PATH`, `MODEL_DIR`

Path defaults (all under the repo that contains the `llm-serve` script, not the current directory):

| Variable | Default |
|----------|---------|
| `LLAMA_DIR` | `{repo}/llama.cpp` |
| `MODEL_DIR` | `{repo}/models` |
| `LOG_DIR` | `{repo}/logs` |

After setup, `~/.local/bin/llm-serve` points at the repo script, so the TUI and CLI work from any directory.

## TUI

```
┌─ MODELS / ALIASES ─┬─ status (running line, throughput, GPU) ─┐
│  cards + presets   │  active preset / runtime config            │
└────────────────────┴─ translated server log ────────────────────┘
```

The running line shows family, quant, and preset slot: `RUNNING  Qwen 3.5  Q8_0  [1]`. **R** and **V** set remote and log verbosity for the *next* launch (saved in `tui-settings.json`).

| Key | Action |
|-----|--------|
| **L** / **S** | Launch / stop |
| **E** | Edit profile, preset, or rename alias |
| **P** | Quant picker (switch file or queue a download) |
| **N** / **D** | New / delete (preset or alias, depending on focus) |
| **1–5** | Activate that preset (or pin it on an alias) |
| **H** | Hugging Face Hub |
| **R** | Next launch local ↔ remote |
| **V** | Next launch log level (INFO → TRACE → DEBUG) |
| **O** | Extra lines in the log pane |
| **T** | Theme |
| **F1** | Key help |
| **Q** | Quit |
| **Tab** | Models ↔ Aliases |
| **←/→** | On an alias: change target model |

In the **preset editor**, **Ctrl+S** saves, **Esc** cancels, **F2** toggles field help from `param-help.conf`.

### Hub and quant picker

Need the `hf` CLI for browse/download. Local models still work without it.

File tables show estimated VRAM and tok/s for this GPU, plus **Act.** columns from runs saved in `tui-baselines.json` (written while a server is up). Context and GPU-offload sliders change the estimates. Downloads are queued one at a time; keep Hub open until they finish. Gated repos need **Login** (Hugging Face token) — that is not API auth for llama-server.

## Configuration

### `models.json`

Identity only. First setup copies `models.json.example` if the file is missing.

```json
{
  "models": {
    "qwen38-27b-bartowski": {
      "display": "Qwen 3.8",
      "host": "127.0.0.1",
      "port": 8081,
      "file": "bartowski/Qwen3.8-27B-IQ2_XXS.gguf",
      "active_quant": "IQ2_XXS",
      "total_layers": 65,
      "source": {
        "hub": "huggingface",
        "repo": "bartowski/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-IQ2_XXS.gguf",
        "author": "bartowski",
        "revision": "main"
      },
      "quants": {
        "IQ2_XXS": {
          "filename": "Qwen3.8-27B-IQ2_XXS.gguf",
          "file": "bartowski/Qwen3.8-27B-IQ2_XXS.gguf"
        }
      }
    }
  },
  "aliases": {
    "default": { "model": "qwen38-27b-bartowski" },
    "coding": { "model": "qwen38-27b-bartowski", "quant": "IQ2_XXS", "preset": 1 }
  }
}
```

`quant` and `preset` on an alias only apply when **both** are set.

### `presets.json`

Created by the TUI (not by setup). Per model, per quant, slots `1`–`5`. Slot 1 is seeded as `default` when a quant is added. Launch uses the active slot for the active (or alias-pinned) quant.

### Other files (repo root, gitignored except the example)

| File | Purpose |
|------|---------|
| `tui-settings.json` | Theme, Hub author recents, next-launch remote and log verbosity |
| `tui-baselines.json` | Measured VRAM and generation averages for Hub **Act.** columns |
| `param-help.conf` | Preset-editor field docs (**F2**). Not a config file. |

## Layout

```
llm-serve/
├── llm-serve              # Entrypoint (TUI if no args; otherwise CLI)
├── setup.sh
├── pyproject.toml         # textual, httpx → project .venv
├── models.json            # Registry (from models.json.example on first setup)
├── presets.json           # Preset slots (TUI)
├── param-help.conf
├── tui/                   # Textual app
├── tests/
├── lib/                   # llama.cpp build + venv helpers
├── patches/llama.cpp/     # Applied at build time
├── models/                # GGUFs as author/filename.gguf
├── llama.cpp/             # Clone + build
├── logs/                  # llm-serve.log, .llm-serve.pid
└── .venv/
```

`models/` is independent of the llama.cpp tree so an update does not touch weights.

## Remote access

```bash
llm-serve my-model --remote
```

Binds `0.0.0.0` and prints reachable URLs. Only use this on a trusted LAN or private VPN (Tailscale, Meshnet, …). Do not expose the port to the public internet.

## Development

```bash
./setup.sh          # or: python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

## FAQ

**Non-GGUF models?** No. llama.cpp GGUF only.

**Several models at once?** No. One PID file, one server.

**vLLM or another backend?** No.

**macOS?** `setup.sh` is Linux-only. The Python launcher may work with a manual llama.cpp build and Bash 4.4+.

**What does `update` do?** Fetches llama.cpp and rebuilds it (same GPU backend as last setup). It does not remove `models/` or your JSON.

## License

MIT — see [LICENSE](LICENSE)
