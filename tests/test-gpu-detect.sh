#!/usr/bin/env bash
# tests/test-gpu-detect.sh — smoke-test GPU detection and build-type resolution
#
# Runs in a temp directory; does NOT touch the real llama.cpp clone or install packages.
# Usage: ./tests/test-gpu-detect.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d /tmp/llm-serve-test.XXXXXX)"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL=$((FAIL + 1)); }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }

cleanup() { rm -rf "${TEST_ROOT}"; }
trap cleanup EXIT

# Isolate from the real install
export LLAMA_DIR="${TEST_ROOT}/llama.cpp"
export LLAMA_SERVER="${LLAMA_DIR}/build/bin/llama-server"
mkdir -p "${LLAMA_DIR}/build/bin"

# shellcheck source=../lib/llama-build.sh
source "${SCRIPT_DIR}/lib/llama-build.sh"

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        pass "$desc (got: $actual)"
    else
        fail "$desc (expected: $expected, got: $actual)"
    fi
}

echo ""
info "Test workspace: ${TEST_ROOT}"
echo ""

# ── 1. Hardware detection (read-only) ───────────────────────────────────────
info "=== Hardware detection ==="
if llama_has_amd_gpu; then
    pass "AMD GPU detected via lspci: $(llama_amd_gpu_name)"
else
    info "No AMD GPU on this machine (skip AMD-specific resolve test later)"
fi

if llama_has_nvidia_gpu; then
    pass "NVIDIA GPU detected: $(llama_nvidia_gpu_name)"
else
    pass "No NVIDIA GPU (expected on Framework laptop)"
fi

# ── 2. Force flags ────────────────────────────────────────────────────────────
info "=== Build type: force flags ==="
llama_resolve_build_type "cpu"
assert_eq "--cpu forces CPU" "cpu" "$_LLAMA_BUILD_TYPE"

# Mock ROCm/CUDA toolchains so force tests don't need sudo
llama_ensure_nvcc() { return 0; }
llama_ensure_rocm() { return 0; }

llama_resolve_build_type "cuda"
assert_eq "--cuda forces CUDA (mocked toolchain)" "cuda" "$_LLAMA_BUILD_TYPE"

llama_resolve_build_type "rocm"
assert_eq "--rocm forces ROCm (mocked toolchain)" "rocm" "$_LLAMA_BUILD_TYPE"

# ── 3. Auto-detect (mocked toolchains) ────────────────────────────────────────
info "=== Build type: auto-detect (mocked install) ==="
llama_resolve_build_type ""
if llama_has_nvidia_gpu; then
    assert_eq "auto-detect prefers NVIDIA" "cuda" "$_LLAMA_BUILD_TYPE"
elif llama_has_amd_gpu; then
    assert_eq "auto-detect prefers AMD → ROCm" "rocm" "$_LLAMA_BUILD_TYPE"
else
    assert_eq "auto-detect with no GPU → CPU" "cpu" "$_LLAMA_BUILD_TYPE"
fi

# ── 4. Auto-detect fallback when toolkit missing ──────────────────────────────
info "=== Build type: auto fallback when install fails ==="
llama_ensure_nvcc() { return 1; }
llama_ensure_rocm() { return 1; }

llama_resolve_build_type ""
if llama_has_nvidia_gpu || llama_has_amd_gpu; then
    assert_eq "auto falls back to CPU when toolkit unavailable" "cpu" "$_LLAMA_BUILD_TYPE"
else
    assert_eq "auto with no GPU stays CPU" "cpu" "$_LLAMA_BUILD_TYPE"
fi

# ── 5. Stamp file logic ───────────────────────────────────────────────────────
info "=== Build stamp read/write ==="
llama_write_stamp "rocm" "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
llama_read_stamp
assert_eq "stamp type written/read" "rocm" "$_LLAMA_STAMP_TYPE"
assert_eq "stamp commit written/read" "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef" "$_LLAMA_STAMP_COMMIT"

# Fake a local git HEAD for stamp match test
mkdir -p "${LLAMA_DIR}/.git"
git -C "${LLAMA_DIR}" init -q
git -C "${LLAMA_DIR}" config user.email "test@test.local"
git -C "${LLAMA_DIR}" config user.name "test"
echo "x" > "${LLAMA_DIR}/README"
git -C "${LLAMA_DIR}" add README
git -C "${LLAMA_DIR}" commit -q -m "test"
HEAD="$(llama_local_head)"
llama_write_stamp "rocm" "$HEAD"
touch "${LLAMA_SERVER}"
chmod +x "${LLAMA_SERVER}"

if llama_binary_is_current "rocm"; then
    pass "stamp match → binary is current (rocm @ ${HEAD:0:7})"
else
    fail "stamp match → binary should be current"
fi

if llama_binary_is_current "cpu"; then
    fail "wrong build type should not match stamp"
else
    pass "cpu build type correctly rejected against rocm stamp"
fi

# ── 6. detect_build_type priority ─────────────────────────────────────────────
info "=== detect_build_type priority ==="
assert_eq "explicit override wins" "cpu" "$(llama_detect_build_type cpu)"
assert_eq "stamp wins over hardware" "rocm" "$(llama_detect_build_type "")"

# ── 7. Optional: shallow llama.cpp clone + cmake dry-run ────────────────────
info "=== Optional: llama.cpp clone + ROCm cmake configure (dry) ==="
if [[ "${RUN_CMAKE_TEST:-0}" == "1" ]]; then
    if llama_ensure_rocm; then
        rm -rf "${LLAMA_DIR}"
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
        BUILD_DIR="${LLAMA_DIR}/build"
        mkdir -p "$BUILD_DIR"
        (
            cd "$BUILD_DIR"
            llama_export_rocm_paths
            HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
                cmake -DBUILD_SHARED_LIBS=OFF -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release ..
        ) && pass "ROCm cmake configure succeeded in temp clone" \
          || fail "ROCm cmake configure failed (ROCm/iGPU compatibility issue?)"
    else
        info "Skipping cmake test — ROCm HIP SDK not installed (set RUN_CMAKE_TEST=1 after installing rocm-hip-sdk)"
    fi
else
    info "Skipping cmake clone test (set RUN_CMAKE_TEST=1 to enable full configure test)"
fi

echo ""
echo "────────────────────────────────────────"
if (( FAIL == 0 )); then
    echo -e "${GREEN}All ${PASS} tests passed.${NC}"
    exit 0
else
    echo -e "${RED}${FAIL} failed, ${PASS} passed.${NC}"
    exit 1
fi
