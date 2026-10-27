#!/usr/bin/env bash
# tests/test-llama-build-targets.sh — llama-server + llama-bench must both build
#
# Guards the Benchmark (TUI key B) dependency: the build has to produce
# build/bin/llama-bench, and llama_binary_is_current must refuse to call a
# server-only build "current" (existing installs carry a valid stamp from one).
#
# Runs in a temp directory with stub cmake/make/ninja on PATH; does NOT compile
# anything or touch the real llama.cpp clone.
# Usage: ./tests/test-llama-build-targets.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d /tmp/llm-serve-targets-test.XXXXXX)"
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

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        pass "$desc (got: $actual)"
    else
        fail "$desc (expected: $expected, got: $actual)"
    fi
}

# Isolate from the real install
export LLAMA_DIR="${TEST_ROOT}/llama.cpp"
export LLAMA_SERVER="${LLAMA_DIR}/build/bin/llama-server"
export LLAMA_BENCH="${LLAMA_DIR}/build/bin/llama-bench"
mkdir -p "${LLAMA_DIR}/build/bin"

# shellcheck source=../lib/llama-build.sh
source "${SCRIPT_DIR}/lib/llama-build.sh"

echo ""
info "Test workspace: ${TEST_ROOT}"
echo ""

# ── 1. Sourcing without LLAMA_BENCH defaults it ──────────────────────────────
info "=== LLAMA_BENCH default ==="
default_bench="$(
    env -u LLAMA_BENCH \
        LLAMA_DIR="${TEST_ROOT}/defaultdir" \
        bash -c 'source "'"${SCRIPT_DIR}"'/lib/llama-build.sh"; printf "%s" "${LLAMA_BENCH}"'
)"
assert_eq "unset LLAMA_BENCH defaults under set -u" \
    "${TEST_ROOT}/defaultdir/build/bin/llama-bench" "$default_bench"

# ── 2. llama_binary_is_current needs both binaries ───────────────────────────
info "=== llama_binary_is_current: both binaries required ==="
git -C "${LLAMA_DIR}" init -q
git -C "${LLAMA_DIR}" config user.email "test@test.local"
git -C "${LLAMA_DIR}" config user.name "test"
echo "x" > "${LLAMA_DIR}/README"
git -C "${LLAMA_DIR}" add README
git -C "${LLAMA_DIR}" commit -q -m "test"
HEAD="$(llama_local_head)"
llama_write_stamp "cpu" "$HEAD"

touch "${LLAMA_SERVER}"; chmod +x "${LLAMA_SERVER}"

if llama_binary_is_current "cpu"; then
    fail "server-only build with a matching stamp must NOT be current"
else
    pass "server-only build + matching stamp → rebuild needed (llama-bench missing)"
fi

touch "${LLAMA_BENCH}"; chmod +x "${LLAMA_BENCH}"

if llama_binary_is_current "cpu"; then
    pass "server + bench + matching stamp → current"
else
    fail "server + bench + matching stamp should be current"
fi

# A non-executable llama-bench is not a build product.
chmod -x "${LLAMA_BENCH}"
if llama_binary_is_current "cpu"; then
    fail "non-executable llama-bench must not count as current"
else
    pass "non-executable llama-bench → rebuild needed"
fi
chmod +x "${LLAMA_BENCH}"

# llama_needs_rebuild mirrors it.
if llama_needs_rebuild "cpu"; then
    fail "llama_needs_rebuild should be false when both binaries are current"
else
    pass "llama_needs_rebuild false when both binaries are current"
fi
rm -f "${LLAMA_BENCH}"
if llama_needs_rebuild "cpu"; then
    pass "llama_needs_rebuild true when llama-bench is gone"
else
    fail "llama_needs_rebuild should be true when llama-bench is missing"
fi
touch "${LLAMA_BENCH}"; chmod +x "${LLAMA_BENCH}"
# The throwaway tree has no llama.cpp sources, so skip llm-serve's own patch
# re-apply; this test is about the build *invocation*, not the patches.
llama_apply_local_patches() { return 0; }

# ── 3. Stub toolchains: the build invocation names both targets ──────────────
info "=== Build invocation names llama-server and llama-bench ==="
stub_dir="${TEST_ROOT}/stub-bin"
mkdir -p "${stub_dir}" "${TEST_ROOT}/logs"

cat > "${stub_dir}/cmake" <<EOF
#!/usr/bin/env bash
echo "cmake \$*" >> "${TEST_ROOT}/logs/argv.log"
EOF

cat > "${stub_dir}/make" <<'EOF'
#!/usr/bin/env bash
echo "make $*" >> "${LOG_DIR}/argv.log"
touch "${LLAMA_SERVER}" "${LLAMA_BENCH}"
chmod +x "${LLAMA_SERVER}" "${LLAMA_BENCH}"
EOF

cat > "${stub_dir}/ninja" <<'EOF'
#!/usr/bin/env bash
echo "ninja $*" >> "${LOG_DIR}/argv.log"
touch "${LLAMA_SERVER}" "${LLAMA_BENCH}"
chmod +x "${LLAMA_SERVER}" "${LLAMA_BENCH}"
EOF

chmod +x "${stub_dir}/cmake" "${stub_dir}/make" "${stub_dir}/ninja"
export LOG_DIR="${TEST_ROOT}/logs"
export PATH="${stub_dir}:${PATH}"

rm -f "${LLAMA_SERVER}" "${LLAMA_BENCH}" "${TEST_ROOT}/logs/argv.log"

if llama_build_server cpu 1; then
    pass "llama_build_server succeeds with stub toolchains"
else
    fail "llama_build_server failed with stub toolchains"
fi

argv="$(cat "${TEST_ROOT}/logs/argv.log" 2>/dev/null || true)"
info "argv: ${argv}"

for bin in "${LLAMA_SERVER}" "${LLAMA_BENCH}"; do
    if [[ -x "$bin" ]]; then
        pass "binary produced: $(basename "$bin")"
    else
        fail "binary missing: $bin"
    fi
done

build_line="$(printf '%s\n' "$argv" | grep -vE '^cmake ' | head -1)"
if [[ -n "$build_line" ]]; then
    for t in llama-server llama-bench; do
        case " ${build_line} " in
            *" ${t} "*) pass "build invocation names target ${t}" ;;
            *) fail "build invocation missing target ${t}: ${build_line}" ;;
        esac
    done
else
    fail "no build invocation was logged"
fi

# Stamp is written only after both binaries exist.
llama_read_stamp
assert_eq "stamp type after build" "cpu" "$_LLAMA_STAMP_TYPE"
assert_eq "stamp commit after build" "$HEAD" "$_LLAMA_STAMP_COMMIT"

# ── 4. A build that fails to produce llama-bench must fail ───────────────────
info "=== Missing llama-bench fails the build ==="
cat > "${stub_dir}/ninja" <<'EOF'
#!/usr/bin/env bash
echo "ninja $*" >> "${LOG_DIR}/argv.log"
touch "${LLAMA_SERVER}"
chmod +x "${LLAMA_SERVER}"
EOF
chmod +x "${stub_dir}/ninja"

rm -f "${LLAMA_SERVER}" "${LLAMA_BENCH}"
if llama_build_server cpu 1; then
    fail "build without llama-bench must return nonzero"
else
    pass "build without llama-bench returns nonzero"
fi
if [[ -f "$(llama_stamp_file)" ]]; then
    llama_read_stamp
    if [[ "$_LLAMA_STAMP_COMMIT" == "$HEAD" ]]; then
        fail "stamp must not be re-asserted by a failed build"
    else
        pass "stamp not overwritten by a failed build"
    fi
else
    pass "no stamp written by a failed build"
fi

echo ""
echo "────────────────────────────────────────"
if (( FAIL == 0 )); then
    echo -e "${GREEN}All ${PASS} checks passed.${NC}"
    exit 0
else
    echo -e "${RED}${FAIL} of $((PASS + FAIL)) checks failed.${NC}"
    exit 1
fi
