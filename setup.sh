#!/usr/bin/env bash
###############################################################################
# setup.sh — llm-serve environment setup
#
# Prepares a fresh clone of the llm-serve repo for first use:
#   - Checks prerequisites (bash, git, cmake, C++ compiler)
#   - Clones llama.cpp if not present
#   - Builds llama-server binary (auto-detects NVIDIA GPU → CUDA, else CPU)
#   - Creates models/ and logs/ directories
#   - Copies models.conf.example → models.conf (if no config exists)
#
# Usage:
#   ./setup.sh              # Auto-detect GPU (NVIDIA → CUDA build, else CPU)
#   ./setup.sh --cpu        # Force CPU-only build
#   ./setup.sh --cuda       # Force CUDA build
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

# ── OS guard ──────────────────────────────────────────────────────────────────
case "$(uname -s)" in
    Linux)  ;;
    Darwin) fail "macOS is not yet supported by setup.sh. The llm-serve launcher \
works on macOS (with Homebrew bash 5), but building llama.cpp requires \
a different setup. See: https://github.com/Doofus-dev/llm-serve" ;;
    MINGW*|MSYS*|CYGWIN*)
            fail "Windows is not yet supported by setup.sh. \
Use WSL2 with a Linux distribution, or build llama.cpp manually \
and point LLAMA_DIR at your build." ;;
    *)      fail "Unsupported OS: $(uname -s). setup.sh only supports Linux." ;;
esac

# ── Distro-aware package hints ────────────────────────────────────────────────
# Returns a one-line install hint for the given tool name.
pkg_hint() {
    local tool="$1"
    if command -v pacman &>/dev/null; then
        case "$tool" in
            git)   echo "sudo pacman -S git" ;;
            cmake) echo "sudo pacman -S cmake" ;;
            g++)   echo "sudo pacman -S gcc" ;;
            nvcc)  echo "sudo pacman -S cuda" ;;
            *)     echo "sudo pacman -S $tool" ;;
        esac
    elif command -v apt-get &>/dev/null; then
        case "$tool" in
            git)   echo "sudo apt-get install git" ;;
            cmake) echo "sudo apt-get install cmake" ;;
            g++)   echo "sudo apt-get install g++" ;;
            nvcc)  echo "sudo apt-get install nvidia-cuda-toolkit" ;;
            *)     echo "sudo apt-get install $tool" ;;
        esac
    elif command -v dnf &>/dev/null; then
        case "$tool" in
            git)   echo "sudo dnf install git" ;;
            cmake) echo "sudo dnf install cmake" ;;
            g++)   echo "sudo dnf install gcc-c++" ;;
            nvcc)  echo "sudo dnf install nvidia-cuda-toolkit" ;;
            *)     echo "sudo dnf install $tool" ;;
        esac
    elif command -v zypper &>/dev/null; then
        case "$tool" in
            git)   echo "sudo zypper install git" ;;
            cmake) echo "sudo zypper install cmake" ;;
            g++)   echo "sudo zypper install gcc-c++" ;;
            nvcc)  echo "sudo zypper install nvidia-cuda-toolkit" ;;
            *)     echo "sudo zypper install $tool" ;;
        esac
    else
        echo "your package manager"
    fi
}

# ── Defaults ─────────────────────────────────────────────────────────────────
LLAMA_DIR="${SCRIPT_DIR}/llama.cpp"
LLAMA_SERVER="${LLAMA_DIR}/build/bin/llama-server"
BUILD_FORCE=""   # empty=auto-detect, cpu, or cuda

# shellcheck source=lib/llama-build.sh
source "${SCRIPT_DIR}/lib/llama-build.sh"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            [[ -z "$BUILD_FORCE" || "$BUILD_FORCE" == "cpu" ]] \
                || fail "Cannot combine --cpu and --cuda"
            BUILD_FORCE="cpu"
            shift
            ;;
        --cuda)
            [[ -z "$BUILD_FORCE" || "$BUILD_FORCE" == "cuda" ]] \
                || fail "Cannot combine --cpu and --cuda"
            BUILD_FORCE="cuda"
            shift
            ;;
        --help|-h)
            echo "llm-serve setup script"
            echo ""
            echo "Usage:"
            echo "  ./setup.sh              Auto-detect GPU (NVIDIA → CUDA, else CPU)"
            echo "  ./setup.sh --cpu        Force CPU-only build"
            echo "  ./setup.sh --cuda       Force CUDA build"
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
    warn "git not found. Installing..."
    if command -v sudo &>/dev/null; then
        if command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm git
        elif command -v apt-get &>/dev/null; then
            sudo apt-get install -y git
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y git
        elif command -v zypper &>/dev/null; then
            sudo zypper install -y git
        else
            fail "git not found. Install it via your package manager."
        fi
    else
        fail "git not found and no sudo available. Install it manually."
    fi
fi
ok "git $(git --version | cut -d' ' -f3)"

# CMake
if ! command -v cmake &>/dev/null; then
    warn "cmake not found. Installing..."
    if command -v sudo &>/dev/null; then
        if command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm cmake
        elif command -v apt-get &>/dev/null; then
            sudo apt-get install -y cmake
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y cmake
        elif command -v zypper &>/dev/null; then
            sudo zypper install -y cmake
        else
            fail "cmake not found. Install it via your package manager."
        fi
    else
        fail "cmake not found and no sudo available. Install it manually."
    fi
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
    warn "No C++ compiler found. Installing g++..."
    if command -v sudo &>/dev/null; then
        if command -v pacman &>/dev/null; then
            # g++ lives in the gcc package on Arch
            sudo pacman -S --noconfirm gcc
        elif command -v apt-get &>/dev/null; then
            sudo apt-get install -y g++
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y gcc-c++
        elif command -v zypper &>/dev/null; then
            sudo zypper install -y gcc-c++
        else
            fail "No C++ compiler found. Install one via your package manager."
        fi
    else
        fail "No C++ compiler found and no sudo available. Install one manually."
    fi
    # Re-check after install
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
        fail "C++ compiler install failed. Try manually: $(pkg_hint g++)"
    fi
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

# GPU / build type (auto-detect NVIDIA by default)
echo ""
info "Detecting build type..."
if ! llama_resolve_build_type "$BUILD_FORCE"; then
    exit 1
fi
BUILD_TYPE="$_LLAMA_BUILD_TYPE"

if [[ "$BUILD_TYPE" == "cuda" ]]; then
    [[ -n "$_LLAMA_GPU_NAME" ]] && ok "NVIDIA GPU: ${_LLAMA_GPU_NAME}"
    ok "Build type: CUDA (GPU acceleration)"
    ok "nvcc $(nvcc --version | grep 'release' | cut -d' ' -f5 | tr -d ',')"
else
    case "$BUILD_FORCE" in
        cpu) ok "Build type: CPU (--cpu)" ;;
        *)
            if llama_has_nvidia_gpu; then
                ok "Build type: CPU (CUDA toolkit unavailable — see warnings above)"
            else
                ok "Build type: CPU (no NVIDIA GPU detected)"
            fi
            ;;
    esac
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
        ok "llama.cpp cloned"
    fi
fi

###############################################################################
# Phase 3: Sync llama.cpp + build llama-server (stamp-aware, auto-pull)
###############################################################################
echo ""

if [[ -d "${LLAMA_DIR}/.git" ]]; then
    info "Checking llama.cpp for updates..."
    if llama_fetch_status; then
        if [[ "$_LLAMA_UPSTREAM_AHEAD" -eq 1 ]]; then
            info "Pulling llama.cpp (${_LLAMA_LOCAL_HEAD:0:7} → ${_LLAMA_REMOTE_HEAD:0:7})..."
            llama_pull_upstream
            ok "llama.cpp updated"
        else
            ok "llama.cpp is current @ $(llama_local_head | cut -c1-7)"
        fi
    else
        warn "Could not reach llama.cpp upstream (offline?). Using local clone."
    fi
fi

if llama_binary_is_current "$BUILD_TYPE"; then
    llama_read_stamp
    ok "llama-server is current (${BUILD_TYPE} build @ ${_LLAMA_STAMP_COMMIT:0:7}). Skipping build."
else
    if [[ -x "${LLAMA_SERVER}" ]]; then
        llama_read_stamp
        if [[ -z "$_LLAMA_STAMP_COMMIT" ]]; then
            warn "Existing binary has no build stamp. Rebuilding to establish one..."
        elif [[ "$_LLAMA_STAMP_TYPE" != "$BUILD_TYPE" ]]; then
            warn "Existing build is ${_LLAMA_STAMP_TYPE}; you requested ${BUILD_TYPE}. Rebuilding..."
        else
            local_head="$(llama_local_head)"
            warn "llama.cpp moved since last build (@ ${_LLAMA_STAMP_COMMIT:0:7} → ${local_head:0:7}). Rebuilding..."
        fi
    fi

    info "Building llama-server (${BUILD_TYPE}, ${NPROC} cores)..."
    llama_wipe_build
    if [[ "$BUILD_TYPE" == "cuda" ]]; then
        info "Configuring with CUDA support..."
    else
        info "Configuring (CPU-only)..."
    fi
    info "Compiling (this may take a while)..."
    if llama_build_server "$BUILD_TYPE" "$NPROC"; then
        ok "llama-server built successfully (${BUILD_TYPE} @ $(llama_local_head | cut -c1-7))"
    else
        fail "Build failed. Check the output above for errors."
    fi
fi

# Show build info
[[ -x "${LLAMA_SERVER}" ]] || fail "llama-server not found at ${LLAMA_SERVER}"
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
# Phase 6: Install to PATH (~/.local/bin symlink)
###############################################################################
echo ""
LOCAL_BIN="${HOME}/.local/bin"
LLM_SERVE_LINK="${LOCAL_BIN}/llm-serve"

if [[ -L "${LLM_SERVE_LINK}" ]]; then
    ok "llm-serve already linked: ${LLM_SERVE_LINK} -> $(readlink "${LLM_SERVE_LINK}")"
elif [[ -e "${LLM_SERVE_LINK}" ]]; then
    warn "${LLM_SERVE_LINK} exists but is not a symlink (skipping)"
else
    mkdir -p "${LOCAL_BIN}"
    ln -s "${SCRIPT_DIR}/llm-serve" "${LLM_SERVE_LINK}"
    ok "Linked llm-serve -> ${LOCAL_BIN}/llm-serve"

    # If ~/.local/bin isn't on PATH, hint about it
    if ! echo "${PATH}" | tr ':' '\n' | grep -qxF "${LOCAL_BIN}"; then
        warn "~/.local/bin is not on your PATH."
        echo ""
        echo "  Add this to your ~/.bashrc or ~/.zshrc:"
        echo "    export PATH=\"\${HOME}/.local/bin:\${PATH}\""
        echo ""
        echo "  Then restart your shell or run: source ~/.bashrc"
        echo ""
    fi
fi

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
