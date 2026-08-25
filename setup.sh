#!/usr/bin/env bash
###############################################################################
# setup.sh — llm-serve environment setup
#
# Prepares a fresh clone of the llm-serve repo for first use:
#   - Checks prerequisites (bash, git, cmake, C++ compiler)
#   - Clones llama.cpp if not present
#   - Builds llama-server binary
#   - Creates models/ and logs/ directories
#   - Copies models.conf.example → models.conf (if no config exists)
#
# Usage:
#   ./setup.sh              # CPU-only build
#   ./setup.sh --cuda       # Build with CUDA GPU support
#   ./setup.sh --help       # Show this help
#
# Idempotent — safe to re-run. Skips steps that are already done.
###############################################################################

set -euo pipefail

# ── Color helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── Resolve script directory ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
LLAMA_DIR="${SCRIPT_DIR}/llama.cpp"
LLAMA_SERVER="${LLAMA_DIR}/build/bin/llama-server"
CUDA_BUILD=0

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            CUDA_BUILD=1
            shift
            ;;
        --help|-h)
            echo "llm-serve setup script"
            echo ""
            echo "Usage:"
            echo "  ./setup.sh              CPU-only build"
            echo "  ./setup.sh --cuda       Build with CUDA GPU support"
            echo "  ./setup.sh --help       Show this help"
            echo ""
            echo "Checks prerequisites, clones & builds llama.cpp,"
            echo "creates models/ and logs/ directories, and sets up"
            echo "a default models.conf from the example template."
            echo ""
            echo "Idempotent — safe to re-run."
            exit 0
            ;;
        *)
            fail "Unknown argument: $1 (use --help for usage)"
            ;;
    esac
done

echo ""
info "llm-serve setup — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

###############################################################################
# Phase 1: Prerequisites
###############################################################################
info "Checking prerequisites..."

# Bash version (need 4.4+ for associative arrays in llm-serve)
BASH_MAJOR="${BASH_VERSINFO[0]}"
if [[ "$BASH_MAJOR" -lt 4 ]]; then
    fail "Bash ${BASH_VERSION} is too old. Need Bash 4.4+."
fi
if [[ "$BASH_MAJOR" -eq 4 && "${BASH_VERSINFO[1]}" -lt 4 ]]; then
    fail "Bash ${BASH_VERSION} is too old. Need Bash 4.4+."
fi
ok "Bash ${BASH_VERSION}"

# Git
if ! command -v git &>/dev/null; then
    fail "git not found. Install it first: sudo apt install git  (Debian/Ubuntu)"
fi
ok "git $(git --version | cut -d' ' -f3)"

# CMake
if ! command -v cmake &>/dev/null; then
    fail "cmake not found. Install it first: sudo apt install cmake"
fi
ok "cmake $(cmake --version | head -1 | cut -d' ' -f3)"

# C++ compiler
if command -v g++ &>/dev/null; then
    CC_NAME="g++"
    CC_VER="$(g++ --version | head -1)"
elif command -v c++ &>/dev/null; then
    CC_NAME="c++"
    CC_VER="$(c++ --version | head -1)"
elif command -v clang++ &>/dev/null; then
    CC_NAME="clang++"
    CC_VER="$(clang++ --version | head -1)"
else
    fail "No C++ compiler found. Install one: sudo apt install g++  (Debian/Ubuntu)"
fi
ok "${CC_NAME} ${CC_VER}"

# Make (or ninja)
if ! command -v make &>/dev/null; then
    warn "make not found. Installing..."
    if command -v sudo &>/dev/null; then
        if command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm make
        elif command -v apt-get &>/dev/null; then
            sudo apt-get install -y make
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y make
        elif command -v zypper &>/dev/null; then
            sudo zypper install -y make
        else
            fail "make not found. Install it via your package manager (make is in base-devel on Arch, build-essential on Debian/Ubuntu)."
        fi
    else
        fail "make not found and no sudo available. Install it manually."
    fi
fi
ok "make $(make --version | head -1 | cut -d' ' -f4)"

# CUDA (optional — only if requested)
if [[ "$CUDA_BUILD" -eq 1 ]]; then
    if ! command -v nvcc &>/dev/null; then
        fail "CUDA build requested (--cuda) but nvcc not found."
        echo ""
        echo "Install the NVIDIA driver and CUDA toolkit first:"
        echo "  Ubuntu:  sudo apt install nvidia-cuda-toolkit"
        echo "  Or visit: https://developer.nvidia.com/cuda-downloads"
    fi
    ok "nvcc $(nvcc --version | grep 'release' | cut -d' ' -f5 | tr -d ',')"

    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        ok "GPU: ${GPU_NAME}"
    else
        warn "nvidia-smi not found — CUDA build will proceed but GPU may not be available"
    fi
fi

# nproc (for parallel builds)
NPROC="$(nproc 2>/dev/null || echo 1)"
info "CPU cores available: ${NPROC}"

###############################################################################
# Phase 2: Clone llama.cpp (if needed)
###############################################################################
echo ""
if [[ -d "${LLAMA_DIR}/.git" ]]; then
    ok "llama.cpp already cloned at ${LLAMA_DIR}"
else
    if [[ -d "${LLAMA_DIR}" ]]; then
        warn "Directory ${LLAMA_DIR} exists but is not a git repo. Skipping clone."
    else
        info "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
        cd "${LLAMA_DIR}"
        git checkout main
        cd "${SCRIPT_DIR}"
        ok "llama.cpp cloned"
    fi
fi

###############################################################################
# Phase 3: Build llama-server
###############################################################################
echo ""
BUILD_DIR="${LLAMA_DIR}/build"

if [[ -x "${LLAMA_SERVER}" ]]; then
    info "llama-server already built at ${LLAMA_SERVER}"
    # Check if it was built with CUDA if we requested CUDA
    if [[ "$CUDA_BUILD" -eq 1 ]]; then
        if "${LLAMA_SERVER}" --help 2>&1 | grep -qi 'cuda'; then
            ok "llama-server already built with CUDA support"
        else
            warn "Existing build lacks CUDA support. Rebuilding with CUDA..."
            rm -rf "${BUILD_DIR}"
        fi
    else
        ok "Skipping build (already exists)"
    fi
fi

# Build if needed
if [[ ! -x "${LLAMA_SERVER}" ]]; then
    info "Building llama-server (${NPROC} cores)..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # Detect CUDA for cmake
    if [[ "$CUDA_BUILD" -eq 1 ]]; then
        info "Configuring with CUDA support..."
        cmake -DBUILD_SHARED_LIBS=OFF -DLLAMA_CUDA=ON ..
    else
        info "Configuring (CPU-only)..."
        cmake -DBUILD_SHARED_LIBS=OFF ..
    fi

    info "Compiling (this may take a while)..."
    make -j"${NPROC}" llama-server

    cd "${SCRIPT_DIR}"

    if [[ -x "${LLAMA_SERVER}" ]]; then
        ok "llama-server built successfully"
    else
        fail "Build failed. Check the output above for errors."
    fi
fi

# Show build info
"${LLAMA_SERVER}" --help 2>&1 | head -1

###############################################################################
# Phase 4: Create runtime directories
###############################################################################
echo ""
MODELS_DIR="${SCRIPT_DIR}/models"
LOGS_DIR="${SCRIPT_DIR}/logs"

if [[ ! -d "${MODELS_DIR}" ]]; then
    mkdir -p "${MODELS_DIR}"
    ok "Created models/ directory"
else
    ok "models/ directory already exists"
fi

if [[ ! -d "${LOGS_DIR}" ]]; then
    mkdir -p "${LOGS_DIR}"
    ok "Created logs/ directory"
else
    ok "logs/ directory already exists"
fi

###############################################################################
# Phase 5: Set up models.conf (if needed)
###############################################################################
CONF="${SCRIPT_DIR}/models.conf"
CONF_EXAMPLE="${SCRIPT_DIR}/models.conf.example"

if [[ ! -f "${CONF}" ]]; then
    if [[ -f "${CONF_EXAMPLE}" ]]; then
        cp "${CONF_EXAMPLE}" "${CONF}"
        ok "Created models.conf from example template"
    else
        warn "No models.conf.example found. You'll need to create models.conf manually."
    fi
else
    ok "models.conf already exists (skipping)"
fi

# Make llm-serve executable
chmod +x "${SCRIPT_DIR}/llm-serve"

###############################################################################
# Done
###############################################################################
echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo ""
ok "Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "  1. Download a .gguf model and put it in: ${MODELS_DIR}/"
echo ""
echo "  2. Edit ${CONF}:"
echo "     - Set MODEL_DIR=${MODELS_DIR}  (or leave empty for default)"
echo "     - Add a register_model entry for your model:"
echo "       register_model \"my-model\" \\"
echo "           file=\"my-model.gguf\" \\"
echo "           gpu_layers=99 \\"
echo "           ctx=32768 \\"
echo "           ... (see models.conf.example for all parameters)"
echo ""
echo "  3. Test it:"
echo "     cd ${SCRIPT_DIR}"
echo "     ./llm-serve --dry-run my-model     # verify config without starting"
echo "     ./llm-serve my-model                # start the server"
echo ""
echo "  4. Launch Hermes with the local model:"
echo "     LLAMA_PORT=8081 hermes config set provider local"
echo ""
echo "─────────────────────────────────────────────────────────────────────"
