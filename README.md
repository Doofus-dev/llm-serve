# llm-serve

A declarative launcher for [llama.cpp](https://github.com/ggml-org/llama.cpp) that manages model profiles, aliases, and server parameters through a single configuration file. Launch any configured model with one command.

## Features

- **Model registry** — Define all your models and their optimal settings in one `models.conf` file
- **Friendly aliases** — `llm-serve coding` instead of `llm-serve qwen3-coder-3b-q4_k_m`
- **Fine-grained parameter control** — GPU offload, KV cache quantization, MoE expert offload, speculative decoding, reasoning/thinking control, and more
- **Environment overrides** — Tune any parameter at runtime without editing config (`GPU_LAYERS=50 llm-serve my-model`)
- **Dry-run mode** — Preview the generated command without launching (`llm-serve --dry-run my-model`)
- **Background/foreground modes** — Run servers in the background or attached to your terminal
- **Status management** — `llm-serve status` and `llm-serve stop` for running servers
- **Hermes Agent integration** — (Optional) Automatically sync model settings to [Hermes Agent](https://hermes-agent.nousresearch.com/docs) configuration

## Prerequisites

- **llama.cpp** — Built and installed. The `llama-server` binary must be accessible.
- **Bash 4.4+** — Required for associative arrays. Most modern Linux distributions include this. macOS users: install Bash 5 via Homebrew (`brew install bash`).
- **CUDA drivers** — (Optional) Required for GPU acceleration.
- **Hermes Agent** — (Optional) Required only for the Hermes config sync feature.

## Installation

```bash
git clone https://github.com/doofus-dev/llm-serve.git
cd llm-serve

# Copy the example config and customize it for your models
cp models.conf.example models.conf
```

### Directory Structure

The script expects your llama.cpp build and models in a specific layout:

```
llm-serve/                    ← this repo
├── llm-serve                 ← launcher script
├── models.conf               ← your model registry
├── llama.cpp/                ← llama.cpp repo (git submodule or symlink)
│   ├── build/bin/llama-server
│   └── models/               ← your .gguf model files
└── logs/                     ← runtime logs
```

Set up the llama.cpp directory however you prefer:

```bash
# Option 1: Clone llama.cpp as a sibling directory
git clone https://github.com/ggml-org/llama.cpp.git

# Option 2: Symlink to an existing build
ln -s /path/to/your/llama.cpp llama.cpp

# Option 3: Override at runtime
LLAMA_DIR=/path/to/your/llama.cpp ./llm-serve my-model
```

## Quick Start

1. **Set up llama.cpp** (see above)
2. **Download a model** to `llama.cpp/models/`
3. **Copy and edit the config:**
   ```bash
   cp models.conf.example models.conf
   # Edit models.conf to point at your actual model files
   ```
4. **Launch a model:**
   ```bash
   ./llm-serve my-model
   ```
5. **Connect to the server:**
   ```bash
   curl http://127.0.0.1:8081/v1/models
   ```

## Usage

```
Usage: llm-serve [command] [options]

Commands:
  <profile>              Launch the named model profile
  list                   List all configured models
  status                 Show running server status
  stop                   Stop the current server
  restart                Restart the current server
  --dry-run <profile>    Show the command that would be run
  --help                 Show this help message

Options:
  --no-hermes            Skip Hermes Agent config sync
```

### Environment Variable Overrides

Any parameter from `models.conf` can be overridden at runtime:

```bash
# Change GPU offload
GPU_LAYERS=50 ./llm-serve my-model

# Increase context window
CONTEXT_SIZE=131072 ./llm-serve my-model

# Use a different model file
MODEL_PATH=/path/to/other-model.gguf ./llm-serve my-model

# Change the server port
PORT=9000 ./llm-serve my-model

# Multiple overrides
GPU_LAYERS=99 CONTEXT_SIZE=32768 THREADS=16 ./llm-serve my-model
```

### Hermes Agent Integration

If you use [Hermes Agent](https://hermes-agent.nousresearch.com/docs), llm-serve can automatically update your Hermes profile configuration to match the launched model's settings — context length, sampling parameters, and server URL.

To use this feature:
1. Install Hermes Agent
2. Add `register_hermes_config` entries to your `models.conf` (see the example)
3. Launch models normally — Hermes config is synced automatically

To disable: `./llm-serve --no-hermes my-model`

## Configuration Reference

See `models.conf.example` for a complete annotated template. Key concepts:

- **`register_model`** — Defines a model profile with all its parameters
- **`register_alias`** — Creates a friendly name that maps to a profile
- Every parameter has a documented default
- Parameters are listed in a fixed order (see the example for the template)

## FAQ

**Do I need Hermes Agent to use this?** No. Hermes integration is optional. The core launcher works independently.

**Can I use this with non-GGUF models?** No. This launcher is designed specifically for llama.cpp's GGUF format.

**Can I run multiple models at once?** Not currently. The script manages a single server instance. See the [roadmap](#roadmap) for multi-model support.

**Does this work on macOS?** Yes, with Bash 4.4+ installed. GPU offload uses Metal through llama.cpp's built-in support.

**Can I use this with vLLM or other backends?** Not yet. The launcher is llama.cpp-specific. See the [roadmap](#roadmap) for a plugin system.

## Roadmap

- Multi-model simultaneous serving
- Web dashboard for status/management
- Plugin system for alternative backends (vLLM, Ollama)
- Docker container for easy deployment
- CI/CD (shellcheck linting, models.conf validation)

## License

MIT — see [LICENSE](LICENSE)
