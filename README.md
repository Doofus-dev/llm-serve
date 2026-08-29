# llm-serve

A declarative launcher for [llama.cpp](https://github.com/ggml-org/llama.cpp) that manages model profiles, aliases, and server parameters through a single configuration file. Launch any configured model with one command.

## Features

- **Model registry** — Define all your models and their optimal settings in one `models.conf` file
- **Friendly aliases** — `llm-serve coding` instead of `llm-serve qwen3-coder-3b-q4_k_m`
- **Fine-grained parameter control** — GPU offload, KV cache quantization, MoE expert offload, speculative decoding, reasoning/thinking control, and more
- **Environment overrides** — Tune any parameter at runtime without editing config (`GPU_LAYERS=50 llm-serve my-model`)
- **Dry-run mode** — Preview the generated command without launching (`llm-serve --dry-run my-model`)
- **Remote access** — `llm-serve my-model --remote` binds to all interfaces so other devices on your LAN or VPN mesh can reach it
- **Background/foreground modes** — Run servers in the background or attached to your terminal
- **Status management** — `llm-serve status` and `llm-serve stop` for running servers
- **One-command setup** — `./setup.sh` handles prerequisites, clones & builds llama.cpp, creates directories
- **Clean uninstall** — `llm-serve uninstall` removes build artifacts while preserving models and config
- **PATH integration** — Automatically symlinks `llm-serve` into `~/.local/bin/` for global access
- **Hermes Agent integration** — (Optional) Automatically sync model settings to [Hermes Agent](https://hermes-agent.nousresearch.com/docs) configuration

## Prerequisites

- **Linux** — setup.sh currently requires a Linux distro with pacman, apt, dnf, or zypper (for auto-installing missing packages)
- **Bash 4.4+** — Required for associative arrays. Most modern Linux distributions include this.
- **NVIDIA GPU** — Detected automatically via `nvidia-smi`; setup installs CUDA toolkit and builds with GPU support
- **AMD GPU** — Detected automatically via `lspci`; setup installs ROCm HIP SDK and builds with `-DGGML_HIP=ON`
- **Overrides** — `./setup.sh --cpu`, `--cuda`, or `--rocm` to force a build type
- **Hermes Agent** — (Optional) Required only for the Hermes config sync feature

## Quick Start

```bash
git clone https://github.com/doofus-dev/llm-serve.git
cd llm-serve

# Run setup (handles everything: prereqs, llama.cpp clone + build, dirs, config)
./setup.sh              # Auto-detects GPU (NVIDIA→CUDA, AMD→ROCm, else CPU)
./setup.sh --cpu        # Force CPU-only
./setup.sh --cuda       # Force CUDA (NVIDIA)
./setup.sh --rocm       # Force ROCm (AMD)

# Drop a .gguf model in models/

# Edit models.conf to register your model (or use the example entries)

# Launch
./llm-serve my-model
```

Verify it's working:
```bash
curl http://127.0.0.1:8081/v1/models
```

## Directory Structure

```
llm-serve/ ← this repo
├── llm-serve              # Launcher script
├── setup.sh               # One-command environment setup
├── models.conf            # Your model registry (created from example on first setup)
├── models.conf.example    # Annotated template with all parameters documented
├── README.md              # This file
├── LICENSE                # MIT
├── .gitignore
├── models/                # Your .gguf model files (created by setup.sh)
├── llama.cpp/             # llama.cpp clone + build (created by setup.sh)
│   └── build/bin/llama-server
└── logs/                  # Runtime logs (created by setup.sh)
```

`models/` lives at the repo root — separate from `llama.cpp/`. This keeps your model weights independent from the upstream source tree, so updating llama.cpp never touches your models.

You can override any path at runtime:
```bash
LLAMA_DIR=/path/to/llama.cpp ./llm-serve my-model
MODEL_DIR=/mnt/models ./llm-serve my-model
```

## Usage

```
Usage: llm-serve [command] [options]

Commands:
  <profile>              Launch the named model profile
  list                   List all configured models
  status                 Show running server status
  stop                   Stop all running servers
  stop <model>           Stop a specific model
  uninstall              Remove build artifacts (keep models/ and models.conf)
  uninstall --force      Non-interactive uninstall
  --dry-run <profile>    Show the command that would be run
  --help                 Show this help message

Options:
  --live                 Run in foreground with live logs
  --remote               Bind 0.0.0.0 so other devices (LAN/Meshnet) can reach it
  --no-hermes            Skip Hermes Agent config sync
  --no-gateway-restart   Skip Hermes gateway restart after config sync
```

### Environment Variable Overrides

Any parameter from `models.conf` can be overridden at runtime:

```bash
GPU_LAYERS=50 ./llm-serve my-model
CONTEXT_SIZE=131072 ./llm-serve my-model
MODEL_PATH=/path/to/other-model.gguf ./llm-serve my-model
MODEL_DIR=/mnt/models ./llm-serve my-model
PORT=9000 ./llm-serve my-model
GPU_LAYERS=99 CONTEXT_SIZE=32768 THREADS=16 ./llm-serve my-model
```

### Remote Access (LAN / Meshnet)

By default the server binds to `127.0.0.1` (localhost only). To serve other devices — a laptop, phone, or anything on your LAN or a private VPN mesh like NordVPN Meshnet or Tailscale — launch with `--remote`:

```bash
./llm-serve my-model --remote
```

This binds `0.0.0.0` (all interfaces), skips the Hermes config sync (so your local `base_url` isn't overwritten with `0.0.0.0`), and prints the addresses other devices can use:

```
Remote mode: bound to 0.0.0.0 — reachable from other devices at:
    http://192.168.1.100:8081/v1    # LAN (example — use your machine's address)
    http://100.64.0.10:8081/v1      # Meshnet / VPN (example — use your mesh address)
```

Security note: `--remote` exposes the server to anything that can reach those addresses. Only use it on trusted networks (LAN, a private VPN mesh). Do not forward the port to the public internet — there is no authentication.

### Hermes Agent Integration

If you use [Hermes Agent](https://hermes-agent.nousresearch.com/docs), llm-serve can automatically update your Hermes profile configuration to match the launched model's settings — context length, sampling parameters, and server URL.

```bash
# Add register_hermes_config entries to models.conf (see models.conf.example)
# Launch models normally — Hermes config is synced automatically

# Or disable:
./llm-serve --no-hermes my-model
```

## Configuration Reference

See `models.conf.example` for a complete annotated template with every parameter documented. Key concepts:

- **`register_model`** — Defines a model profile with all its parameters (file, gpu_layers, ctx, cache types, sampling, reasoning, MTP, etc.)
- **`register_alias`** — Creates a friendly name that maps to a profile
- **`register_hermes_config`** — (Optional) Syncs model settings to Hermes Agent
- Every parameter has a documented default
- Parameters follow a fixed order (see the annotated template)

## FAQ

**Do I need Hermes Agent to use this?** No. Hermes integration is optional. The core launcher works independently.

**Can I use this with non-GGUF models?** No. This launcher is designed specifically for llama.cpp's GGUF format.

**Can I run multiple models at once?** Not currently. The script manages a single server instance.

**Does this work on macOS?** The launcher works with Bash 4.4+ (install via Homebrew: `brew install bash`). setup.sh currently only supports Linux.

**Can I use this with vLLM or other backends?** Not yet. The launcher is llama.cpp-specific.

**What does uninstall remove?** llama.cpp clone+build, logs, and the ~/.local/bin symlink. It preserves models/, models.conf, and any system packages installed by setup.sh.

## License

MIT — see [LICENSE](LICENSE)