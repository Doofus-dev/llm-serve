# Machine-wide llm-serve data dir (registry, presets, GGUFs, logs).
# Sourced by setup.sh. Python tui.paths.shared_data_home must stay in sync.
llm_serve_data_dir() {
    if [[ -n "${LLM_SERVE_DATA:-}" ]]; then
        printf '%s\n' "${LLM_SERVE_DATA}"
    elif [[ -n "${XDG_DATA_HOME:-}" ]]; then
        printf '%s\n' "${XDG_DATA_HOME}/llm-serve"
    else
        printf '%s\n' "${HOME}/.local/share/llm-serve"
    fi
}
