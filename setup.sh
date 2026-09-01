#!/usr/bin/env bash
###############################################################################
# setup.sh — llm-serve environment setup
#
# Prepares a fresh clone of the llm-serve repo for first use:
#   - Checks prerequisites (bash, git, cmake, C++ compiler, Python 3)
#   - Clones llama.cpp if not present
#   - Builds llama-server binary (auto-detects GPU: NVIDIA→CUDA, AMD→ROCm, else CPU)
#   - Creates a Python venv and installs TUI packages (textual, httpx)
#   - Creates models/ and logs/ directories
#   - Copies models.conf.example → models.conf (if no config exists)
#
# Usage:
#   ./setup.sh              # Auto-detect GPU
#   ./setup.sh --cpu        # Force CPU-only build
#   ./setup.sh --cuda       # Force CUDA build (NVIDIA)
#   ./setup.sh --rocm       # Force ROCm build (AMD)
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
            git)      echo "sudo pacman -S git" ;;
            cmake)    echo "sudo pacman -S cmake" ;;
            g++)      echo "sudo pacman -S gcc" ;;
            nvcc)     echo "sudo pacman -S cuda" ;;
            rocm)     echo "sudo pacman -S rocm-hip-sdk" ;;
            python3)  echo "sudo pacman -S python" ;;
            *)        echo "sudo pacman -S $tool" ;;
        esac
    elif command -v apt-get &>/dev/null; then
        case "$tool" in
            git)      echo "sudo apt-get install git" ;;
            cmake)    echo "sudo apt-get install cmake" ;;
            g++)      echo "sudo apt-get install g++" ;;
            nvcc)     echo "sudo apt-get install nvidia-cuda-toolkit" ;;
            rocm)     echo "sudo apt-get install rocm-hip-sdk" ;;
            python3)  echo "sudo apt-get install python3 python3-venv python3-pip" ;;
            *)        echo "sudo apt-get install $tool" ;;
        esac
    elif command -v dnf &>/dev/null; then
        case "$tool" in
            git)      echo "sudo dnf install git" ;;
            cmake)    echo "sudo dnf install cmake" ;;
            g++)      echo "sudo dnf install gcc-c++" ;;
            nvcc)     echo "sudo dnf install nvidia-cuda-toolkit" ;;
            python3)  echo "sudo dnf install python3 python3-pip" ;;
            *)        echo "sudo dnf install $tool" ;;
        esac
    elif command -v zypper &>/dev/null; then
        case "$tool" in
            git)      echo "sudo zypper install git" ;;
            cmake)    echo "sudo zypper install cmake" ;;
            g++)      echo "sudo zypper install gcc-c++" ;;
            nvcc)     echo "sudo zypper install nvidia-cuda-toolkit" ;;
            python3)  echo "sudo zypper install python3 python3-pip python3-venv" ;;
            *)        echo "sudo zypper install $tool" ;;
        esac
    else
        echo "your package manager"
    fi
}

# Install native distro packages. Names must match the detected package manager.
install_packages() {
    [[ $# -gt 0 ]] || return 0
    if ! command -v sudo &>/dev/null; then
        fail "Need sudo to install packages: $*. Try: $(pkg_hint "$1")"
    fi
    if command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm "$@"
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y "$@"
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y "$@"
    elif command -v zypper &>/dev/null; then
        sudo zypper install -y "$@"
    else
        fail "No supported package manager found. Install manually: $*"
    fi
}

# Distro packages needed for python3 + venv + pip (Ubuntu needs python3-venv).
python_system_packages() {
    if command -v pacman &>/dev/null; then
        echo python
    elif command -v apt-get &>/dev/null; then
        echo python3 python3-venv python3-pip
    elif command -v dnf &>/dev/null; then
        echo python3 python3-pip
    elif command -v zypper &>/dev/null; then
        echo python3 python3-pip python3-venv
    else
        return 1
    fi
}

# ── Defaults ─────────────────────────────────────────────────────────────────
LLAMA_DIR="${SCRIPT_DIR}/llama.cpp"
LLAMA_SERVER="${LLAMA_DIR}/build/bin/llama-server"
BUILD_FORCE=""   # empty=auto-detect, cpu, cuda, or rocm

# shellcheck source=lib/llama-build.sh
source "${SCRIPT_DIR}/lib/llama-build.sh"
# shellcheck source=lib/tui-venv.sh
source "${SCRIPT_DIR}/lib/tui-venv.sh"

_set_build_force() {
    local val="$1"
    [[ -z "$BUILD_FORCE" ]] || fail "Cannot combine --cpu, --cuda, and --rocm"
    BUILD_FORCE="$val"
}

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            _set_build_force "cpu"
            shift
            ;;
        --cuda)
            _set_build_force "cuda"
            shift
            ;;
        --rocm)
            _set_build_force "rocm"
            shift
            ;;
        --help|-h)
            echo "llm-serve setup script"
            echo ""
            echo "Usage:"
            echo "  ./setup.sh              Auto-detect GPU (NVIDIA→CUDA, AMD→ROCm, else CPU)"
            echo "  ./setup.sh --cpu        Force CPU-only build"
            echo "  ./setup.sh --cuda       Force CUDA build (NVIDIA)"
            echo "  ./setup.sh --rocm       Force ROCm build (AMD)"
            echo "  ./setup.sh --help       Show this help"
            echo ""
            echo "Checks prerequisites, clones & builds llama.cpp, installs"
            echo "Python and TUI dependencies, creates models/ and logs/"
            echo "directories, and sets up a default models.conf."
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

# Python 3 + venv (TUI). Ubuntu/Debian need python3-venv for ensurepip.
echo ""
info "Checking Python for the TUI..."
if ! command -v python3 &>/dev/null; then
    warn "python3 not found. Installing..."
    pkgs="$(python_system_packages)" || fail "No supported package manager. Install Python 3.9+ manually."
    # shellcheck disable=SC2086
    install_packages ${pkgs} || fail "Could not install Python. Try: $(pkg_hint python3)"
fi
if ! command -v python3 &>/dev/null; then
    fail "python3 not found. Install it via your package manager: $(pkg_hint python3)"
fi
if ! tui_python_version_ok; then
    fail "Python $(tui_python_version) is too old. Need Python ${TUI_MIN_PY_MAJOR}.${TUI_MIN_PY_MINOR}+."
fi
if ! tui_venv_module_ok; then
    warn "Python venv support missing (Ubuntu/Debian: python3-venv). Installing..."
    pkgs="$(python_system_packages)" || fail "No supported package manager. Install python3-venv / equivalent manually."
    # shellcheck disable=SC2086
    install_packages ${pkgs} || fail "Could not install Python venv support. Try: $(pkg_hint python3)"
fi
if ! tui_venv_module_ok; then
    fail "Python venv/ensurepip still missing. Install it via your package manager ($(pkg_hint python3)) and re-run."
fi
ok "Python $(tui_python_version)"

TUI_VENV="$(tui_venv_dir "${SCRIPT_DIR}")"
if ! tui_venv_healthy "${TUI_VENV}"; then
    if [[ -e "${TUI_VENV}" ]]; then
        warn "Existing .venv is incomplete. Recreating..."
    else
        info "Creating Python virtualenv at .venv..."
    fi
    tui_recreate_venv "${TUI_VENV}" || fail "Failed to create .venv. On Ubuntu/Debian: sudo apt-get install python3-venv && ./setup.sh"
    ok "Created .venv"
else
    ok ".venv already exists"
fi

info "Installing TUI packages (textual, httpx)..."
tui_pip_install "${SCRIPT_DIR}" || fail "pip install failed. Check network access to PyPI."
tui_deps_ok "${TUI_VENV}" || fail "TUI modules failed to import after install (textual/httpx)."
ok "TUI Python packages installed"

# GPU / build type (auto-detect by default)
echo ""
info "Detecting build type..."
if ! llama_resolve_build_type "$BUILD_FORCE"; then
    exit 1
fi
BUILD_TYPE="$_LLAMA_BUILD_TYPE"

case "$BUILD_TYPE" in
    cuda)
        [[ -n "$_LLAMA_GPU_NAME" ]] && ok "NVIDIA GPU: ${_LLAMA_GPU_NAME}"
        ok "Build type: CUDA (GPU acceleration)"
        ok "nvcc $(nvcc --version | grep 'release' | cut -d' ' -f5 | tr -d ',')"
        ;;
    rocm)
        [[ -n "$_LLAMA_GPU_NAME" ]] && ok "AMD GPU: ${_LLAMA_GPU_NAME}"
        ok "Build type: ROCm/HIP (GPU acceleration)"
        ok "hipcc $(hipcc --version 2>/dev/null | head -1 || echo 'installed')"
        ;;
    *)
        case "$BUILD_FORCE" in
            cpu) ok "Build type: CPU (--cpu)" ;;
            *)
                if llama_has_nvidia_gpu || llama_has_amd_gpu; then
                    ok "Build type: CPU (GPU toolkit unavailable — see warnings above)"
                else
                    ok "Build type: CPU (no supported GPU detected)"
                fi
                ;;
        esac
        ;;
esac

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
        echo "    llama-server needs the llama.cpp source tree to build."
        echo "    Remove it and re-run setup:  rm -rf ${LLAMA_DIR} && ./setup.sh"
        exit 1
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
    case "$BUILD_TYPE" in
        cuda) info "Configuring with CUDA support..." ;;
        rocm) info "Configuring with ROCm/HIP support..." ;;
        *)    info "Configuring (CPU-only)..." ;;
    esac
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

MODELS_JSON="${SCRIPT_DIR}/models.json"
MODELS_JSON_EXAMPLE="${SCRIPT_DIR}/models.json.example"

if [[ ! -f "${MODELS_JSON}" ]]; then
    if [[ -f "${MODELS_JSON_EXAMPLE}" ]]; then
        cp "${MODELS_JSON_EXAMPLE}" "${MODELS_JSON}"
        ok "Created models.json from example template"
    else
        warn "No models.json.example found. Create models.json manually or run migrate-to-json.py."
    fi
else
    ok "models.json already exists (skipping)"
fi

# Make launchers executable
chmod +x "${SCRIPT_DIR}/llm-serve" "${SCRIPT_DIR}/llm-serve-tui"

###############################################################################
# Phase 6: Install to PATH (~/.local/bin symlink)
###############################################################################
echo ""
LOCAL_BIN="${HOME}/.local/bin"
mkdir -p "${LOCAL_BIN}"

link_into_local_bin() {
    local name="$1"
    local src="${SCRIPT_DIR}/${name}"
    local dest="${LOCAL_BIN}/${name}"
    local current

    if [[ -L "${dest}" ]]; then
        current="$(readlink "${dest}")"
        if [[ "$current" == "$src" ]]; then
            ok "${name} already linked: ${dest} -> ${current}"
        else
            warn "${name} symlink points to a different location."
            echo "       Current: ${dest} -> ${current}"
            echo "       New:     ${dest} -> ${src}"
            info "Updating symlink..."
            rm -f "${dest}"
            ln -s "${src}" "${dest}"
            ok "Symlink updated."
        fi
    elif [[ -e "${dest}" ]]; then
        warn "${dest} exists but is not a symlink (skipping)"
    else
        ln -s "${src}" "${dest}"
        ok "Linked ${name} -> ${dest}"
    fi
}

link_into_local_bin llm-serve
link_into_local_bin llm-serve-tui

if ! echo "${PATH}" | tr ':' '\n' | grep -qxF "${LOCAL_BIN}"; then
    warn "~/.local/bin is not on your PATH."
    echo ""
    echo "  Add this to your ~/.bashrc or ~/.zshrc:"
    echo "    export PATH=\"\${HOME}/.local/bin:\${PATH}\""
    echo ""
    echo "  Then restart your shell or run: source ~/.bashrc"
    echo ""
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
echo "  1. Launch the TUI (Python env is already set up):"
echo "     cd ${SCRIPT_DIR}"
echo "     ./llm-serve-tui"
echo ""
echo "  2. Download a .gguf model and put it in: ${MODELS_DIR}/"
echo "     (or press H in the TUI to browse/download from Hugging Face; optional: install hf CLI)"
echo ""
echo "  3. Edit ${CONF} (or use the TUI editor):"
echo "     - Set MODEL_DIR=${MODELS_DIR}  (or leave empty for default)"
echo "     - Add a register_model entry for your model:"
echo "       register_model \"my-model\" \\"
echo "           file=\"my-model.gguf\" \\"
echo "           gpu_layers=99 \\"
echo "           ctx=32768 \\"
echo "           ... (see models.conf.example for all parameters)"
echo ""
echo "  4. Test it:"
echo "     ./llm-serve --dry-run my-model     # verify config without starting"
echo "     ./llm-serve my-model                # start the server"
echo ""
echo "  5. Launch Hermes with the local model:"
echo "     LLAMA_PORT=8081 hermes config set provider local"
echo ""
echo "─────────────────────────────────────────────────────────────────────"
