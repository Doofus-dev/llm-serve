#!/usr/bin/env bash
# Shared TUI virtualenv helpers for setup.sh and llm-serve.
# Source this file; do not execute directly.
#
# Requires callers to set SCRIPT_DIR to the repo root before sourcing,
# or pass the repo root as an argument to the high-level helpers.

TUI_MIN_PY_MAJOR=3
TUI_MIN_PY_MINOR=9

tui_venv_dir() {
    echo "${1:-${SCRIPT_DIR}}/.venv"
}

tui_requirements_file() {
    echo "${1:-${SCRIPT_DIR}}/tui/requirements.txt"
}

tui_python_version_ok() {
    python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (${TUI_MIN_PY_MAJOR}, ${TUI_MIN_PY_MINOR}) else 1)" 2>/dev/null
}

tui_python_version() {
    python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true
}

# Debian/Ubuntu ship python3 without ensurepip; python3-venv provides both.
tui_venv_module_ok() {
    python3 -c "import venv, ensurepip" 2>/dev/null
}

tui_venv_healthy() {
    local venv="${1:-$(tui_venv_dir)}"
    [[ -x "${venv}/bin/python" ]] && "${venv}/bin/python" -c "import pip" 2>/dev/null
}

tui_deps_ok() {
    local venv="${1:-$(tui_venv_dir)}"
    [[ -x "${venv}/bin/python" ]] && "${venv}/bin/python" -c "import textual, httpx" 2>/dev/null
}

# Recreate the venv. Cleans up a half-created directory if venv fails
# (common on Ubuntu when python3-venv is missing).
tui_recreate_venv() {
    local venv="${1:-$(tui_venv_dir)}"
    rm -rf "${venv}"
    if ! python3 -m venv "${venv}"; then
        rm -rf "${venv}"
        return 1
    fi
    tui_venv_healthy "${venv}"
}

tui_pip_install() {
    local root="${1:-${SCRIPT_DIR}}"
    local venv
    venv="$(tui_venv_dir "${root}")"
    local reqs
    reqs="$(tui_requirements_file "${root}")"
    [[ -x "${venv}/bin/pip" ]] || return 1
    [[ -f "${reqs}" ]] || return 1
    "${venv}/bin/pip" install -q --upgrade pip
    "${venv}/bin/pip" install -q -r "${reqs}"
}

# Create/repair the venv and install TUI deps if they are missing.
# Prints a one-line status to stdout when it actually installs something.
tui_ensure_runtime() {
    local root="${1:-${SCRIPT_DIR}}"
    local venv
    venv="$(tui_venv_dir "${root}")"

    if ! tui_venv_healthy "${venv}"; then
        echo "Creating Python virtualenv and installing TUI dependencies..."
        tui_recreate_venv "${venv}" || return 1
        tui_pip_install "${root}" || return 1
    elif ! tui_deps_ok "${venv}"; then
        echo "Installing TUI Python packages..."
        tui_pip_install "${root}" || return 1
    fi
    tui_deps_ok "${venv}"
}
