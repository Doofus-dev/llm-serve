#!/usr/bin/env bash
# Shared llama.cpp clone/build/update helpers for setup.sh and llm-serve.
# Source this file; do not execute directly.
#
# Requires callers to set LLAMA_DIR and LLAMA_SERVER before sourcing.

llama_build_dir() { echo "${LLAMA_DIR}/build"; }
llama_stamp_file()  { echo "$(llama_build_dir)/.llm-serve-build"; }

llama_nproc() {
    nproc 2>/dev/null || echo 1
}

llama_local_head() {
    if [[ -d "${LLAMA_DIR}/.git" ]]; then
        git -C "${LLAMA_DIR}" rev-parse HEAD 2>/dev/null || true
    fi
}

# Populates _LLAMA_STAMP_TYPE and _LLAMA_STAMP_COMMIT.
llama_read_stamp() {
    local stamp_file
    stamp_file="$(llama_stamp_file)"
    _LLAMA_STAMP_TYPE=""
    _LLAMA_STAMP_COMMIT=""
    [[ -f "$stamp_file" ]] || return 0
    _LLAMA_STAMP_TYPE="$(grep -m1 '^type='   "$stamp_file" 2>/dev/null | cut -d= -f2- || true)"
    _LLAMA_STAMP_COMMIT="$(grep -m1 '^commit=' "$stamp_file" 2>/dev/null | cut -d= -f2- || true)"
}

llama_write_stamp() {
    local build_type="$1" commit="$2"
    mkdir -p "$(llama_build_dir)"
    {
        echo "type=${build_type}"
        echo "commit=${commit}"
        echo "date=$(date '+%Y-%m-%d %H:%M:%S')"
    } > "$(llama_stamp_file)"
}

# True when an NVIDIA GPU + driver are present (nvidia-smi works).
llama_has_nvidia_gpu() {
    command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null
}

llama_nvidia_gpu_name() {
    if llama_has_nvidia_gpu; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true
    fi
}

# Put distro CUDA toolkit locations on PATH/LD_LIBRARY_PATH for cmake and runtime.
llama_export_cuda_paths() {
    local d
    for d in /opt/cuda/bin /usr/local/cuda/bin; do
        if [[ -d "$d" ]]; then
            export PATH="${d}:${PATH}"
        fi
    done
    for d in /opt/cuda/lib64 /usr/local/cuda/lib64; do
        if [[ -d "$d" ]]; then
            export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
    done
}

# Install the CUDA compiler toolkit via the distro package manager.
# Returns 0 when nvcc is available afterward.
llama_install_nvcc() {
    if command -v nvcc &>/dev/null; then
        return 0
    fi
    if ! command -v sudo &>/dev/null; then
        return 1
    fi
    echo "CUDA toolkit (nvcc) not found. Installing via package manager..." >&2
    if command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm cuda
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y nvidia-cuda-toolkit
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y nvidia-cuda-toolkit
    elif command -v zypper &>/dev/null; then
        sudo zypper install -y nvidia-cuda-toolkit
    else
        return 1
    fi
    llama_export_cuda_paths
    command -v nvcc &>/dev/null
}

# Ensure nvcc exists; install via package manager when possible.
llama_ensure_nvcc() {
    llama_export_cuda_paths
    if command -v nvcc &>/dev/null; then
        return 0
    fi
    llama_install_nvcc
}

# Resolve cpu|cuda for setup.sh.
#   force=""   → auto-detect (NVIDIA GPU → cuda, else cpu)
#   force=cpu  → force CPU
#   force=cuda → force CUDA
# Sets _LLAMA_BUILD_TYPE. Optionally sets _LLAMA_GPU_NAME when a GPU is found.
# When CUDA is selected, tries to install nvcc; on auto-detect, falls back to
# CPU if the toolkit cannot be installed.
llama_resolve_build_type() {
    local force="${1:-}"

    _LLAMA_GPU_NAME=""

    if [[ "$force" == "cpu" ]]; then
        _LLAMA_BUILD_TYPE="cpu"
        return 0
    fi

    if [[ "$force" == "cuda" ]]; then
        _LLAMA_BUILD_TYPE="cuda"
    elif llama_has_nvidia_gpu; then
        _LLAMA_BUILD_TYPE="cuda"
        _LLAMA_GPU_NAME="$(llama_nvidia_gpu_name)"
    else
        _LLAMA_BUILD_TYPE="cpu"
        return 0
    fi

    if [[ -n "$_LLAMA_GPU_NAME" ]]; then
        : # caller prints GPU name
    elif [[ "$force" == "cuda" ]]; then
        echo "Warning: --cuda requested but nvidia-smi not found; proceeding with CUDA build anyway." >&2
    fi

    if llama_ensure_nvcc; then
        return 0
    fi

    if [[ "$force" == "cuda" ]]; then
        echo "Error: CUDA build requested but nvcc could not be installed." >&2
        echo "Install the CUDA toolkit manually, then re-run setup." >&2
        if command -v pacman &>/dev/null; then
            echo "  sudo pacman -S cuda" >&2
        elif command -v apt-get &>/dev/null; then
            echo "  sudo apt-get install nvidia-cuda-toolkit" >&2
        else
            echo "  See: https://developer.nvidia.com/cuda-downloads" >&2
        fi
        return 1
    fi

    echo "Warning: NVIDIA GPU detected but CUDA toolkit could not be installed (need sudo?). Building CPU-only." >&2
    echo "         Install the toolkit and re-run ./setup.sh to enable GPU acceleration." >&2
    _LLAMA_BUILD_TYPE="cpu"
    return 0
}

# Resolve cpu|cuda: explicit override > stamp > binary heuristic > NVIDIA auto-detect > cpu.
llama_detect_build_type() {
    local explicit="${1:-}"
    if [[ -n "$explicit" ]]; then
        echo "$explicit"
        return 0
    fi
    llama_read_stamp
    if [[ -n "$_LLAMA_STAMP_TYPE" ]]; then
        echo "$_LLAMA_STAMP_TYPE"
        return 0
    fi
    if [[ -x "${LLAMA_SERVER}" ]] && "${LLAMA_SERVER}" --help 2>&1 | grep -qi 'cuda'; then
        echo "cuda"
        return 0
    fi
    if llama_has_nvidia_gpu; then
        echo "cuda"
        return 0
    fi
    echo "cpu"
}

# Fetch upstream HEAD. Sets _LLAMA_LOCAL_HEAD, _LLAMA_REMOTE_HEAD, _LLAMA_UPSTREAM_AHEAD (0|1).
# Returns 0 on success, 1 if fetch failed or not a git repo.
llama_fetch_status() {
    _LLAMA_LOCAL_HEAD="$(llama_local_head)"
    _LLAMA_REMOTE_HEAD=""
    _LLAMA_UPSTREAM_AHEAD=0

    [[ -d "${LLAMA_DIR}/.git" ]] || return 1
    git -C "${LLAMA_DIR}" fetch --quiet origin HEAD 2>/dev/null || return 1

    _LLAMA_REMOTE_HEAD="$(git -C "${LLAMA_DIR}" rev-parse FETCH_HEAD 2>/dev/null || true)"
    if [[ -n "$_LLAMA_LOCAL_HEAD" && -n "$_LLAMA_REMOTE_HEAD" \
          && "$_LLAMA_LOCAL_HEAD" != "$_LLAMA_REMOTE_HEAD" ]]; then
        _LLAMA_UPSTREAM_AHEAD=1
    fi
    return 0
}

llama_pull_upstream() {
    git -C "${LLAMA_DIR}" pull --ff-only
}

llama_wipe_build() {
    rm -rf "$(llama_build_dir)"
}

# Return 0 when the existing binary matches build_type @ local HEAD.
llama_binary_is_current() {
    local build_type="$1"
    local head
    head="$(llama_local_head)"

    [[ -x "${LLAMA_SERVER}" ]] || return 1
    llama_read_stamp
    [[ -n "$_LLAMA_STAMP_TYPE" && -n "$_LLAMA_STAMP_COMMIT" ]] || return 1
    [[ "$_LLAMA_STAMP_TYPE" == "$build_type" ]] || return 1
    [[ -n "$head" && "$_LLAMA_STAMP_COMMIT" == "$head" ]] || return 1
    return 0
}

# Return 0 when a rebuild is needed.
llama_needs_rebuild() {
    local build_type="$1"
    llama_binary_is_current "$build_type" && return 1
    return 0
}

# Configure, compile, and stamp. Caller should wipe build/ first when refreshing.
llama_build_server() {
    local build_type="$1"
    local nproc="${2:-$(llama_nproc)}"
    local build_dir head

    build_dir="$(llama_build_dir)"
    mkdir -p "$build_dir"
    (
        if [[ "$build_type" == "cuda" ]]; then
            llama_export_cuda_paths
        fi
        cd "$build_dir"
        if [[ "$build_type" == "cuda" ]]; then
            cmake -DBUILD_SHARED_LIBS=OFF -DLLAMA_CUDA=ON ..
        else
            cmake -DBUILD_SHARED_LIBS=OFF ..
        fi
        make -j"${nproc}" llama-server
    )

    [[ -x "${LLAMA_SERVER}" ]] || return 1
    head="$(llama_local_head)"
    [[ -n "$head" ]] && llama_write_stamp "$build_type" "$head"
    return 0
}

# Pull (if upstream is ahead), wipe build/, rebuild, stamp.
llama_do_update() {
    local build_type="$1"
    local nproc="${2:-$(llama_nproc)}"

    if llama_fetch_status && [[ "$_LLAMA_UPSTREAM_AHEAD" -eq 1 ]]; then
        llama_pull_upstream
    fi

    llama_wipe_build
    llama_build_server "$build_type" "$nproc"
}

# Sets _LLAMA_OUTDATED (0|1) and _LLAMA_OUTDATED_REASON.
llama_check_outdated() {
    local head
    _LLAMA_OUTDATED=0
    _LLAMA_OUTDATED_REASON=""

    llama_read_stamp
    head="$(llama_local_head)"

    if llama_fetch_status && [[ "$_LLAMA_UPSTREAM_AHEAD" -eq 1 ]]; then
        _LLAMA_OUTDATED=1
        _LLAMA_OUTDATED_REASON="upstream has newer commits (local ${_LLAMA_LOCAL_HEAD:0:7} → upstream ${_LLAMA_REMOTE_HEAD:0:7})"
        return 0
    fi

    if [[ -n "$head" && -n "$_LLAMA_STAMP_COMMIT" && "$_LLAMA_STAMP_COMMIT" != "$head" ]]; then
        _LLAMA_OUTDATED=1
        _LLAMA_OUTDATED_REASON="binary was built @ ${_LLAMA_STAMP_COMMIT:0:7} but llama.cpp is now @ ${head:0:7}"
        return 0
    fi

    if [[ -x "${LLAMA_SERVER}" && -z "$_LLAMA_STAMP_COMMIT" ]]; then
        _LLAMA_OUTDATED=1
        _LLAMA_OUTDATED_REASON="no build stamp (cannot verify llama-server version)"
    fi
}

llama_warn_if_outdated() {
    [[ -x "${LLAMA_SERVER}" ]] || return 0
    llama_check_outdated
    if [[ "$_LLAMA_OUTDATED" -eq 1 ]]; then
        echo "[!] llama-server is out of date: ${_LLAMA_OUTDATED_REASON}." >&2
        echo "    To update:  llm-serve update" >&2
    fi
}
