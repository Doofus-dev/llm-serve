"""Editor widget types and fixed-choice parameter values."""

from __future__ import annotations

from typing import Any

# Shown in config/launch but not in the model/preset editor UI.
HIDDEN_EDITOR_PARAMS = frozenset({"file"})

CACHE_TYPE_VALUES = (
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
)

# param -> (label, stored value)
PARAM_SELECT_OPTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "cache_k": tuple((value, value) for value in CACHE_TYPE_VALUES),
    "cache_v": tuple((value, value) for value in CACHE_TYPE_VALUES),
    "jinja": (("on", "on"), ("off", "off")),
    "cache_prompt": (("on", "on"), ("off", "off")),
    "cont_batching": (("on", "on"), ("off", "off")),
    "metrics": (("on", "on"), ("off", "off")),
    "flash_attn": (("on", "on"), ("off", "off"), ("auto", "auto")),
    "thinking": (("on", "on"), ("off", "off")),
    "reasoning_format": (
        ("auto", "auto"),
        ("none", "none"),
        ("deepseek", "deepseek"),
        ("deepseek-legacy", "deepseek-legacy"),
    ),
    "mtp": (("Off", "0"), ("On", "1")),
}

# Empty stored value maps to Select.NULL (server default — flag omitted).
PARAM_ALLOW_BLANK = frozenset({"thinking", "reasoning_format"})

PRESERVED_EDITOR_PARAMS = frozenset({"total_layers", "file"})


def fmt_locked_value(value: object) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def is_select_param(param: str) -> bool:
    return param in PARAM_SELECT_OPTIONS


def select_options(param: str) -> list[tuple[str, str]]:
    return list(PARAM_SELECT_OPTIONS[param])


def _normalize_stored(value: Any) -> str:
    if value is None or value is False:
        return ""
    return str(value).strip()


def _is_blank_select_value(selected: object) -> bool:
    try:
        from textual.widgets._select import NULL, NoSelection
    except ImportError:
        return selected in (None, "")
    return selected is NULL or isinstance(selected, NoSelection)


def select_initial_value(param: str, stored: Any) -> str | None:
    """Map stored config to a Select value, or None for blank/default."""
    text = _normalize_stored(stored)
    if param in PARAM_ALLOW_BLANK and not text:
        return None
    for _label, value in PARAM_SELECT_OPTIONS[param]:
        if value == text:
            return value
    # Fall back to first option when config has an unexpected value.
    return PARAM_SELECT_OPTIONS[param][0][1]


def select_to_stored(param: str, selected: object) -> str:
    """Map Select.value back to models.json / preset override string."""
    if _is_blank_select_value(selected):
        return ""
    if selected in (None, ""):
        return ""
    text = str(selected)
    if param in PARAM_ALLOW_BLANK and text in ("", "None", "False"):
        return ""
    return text


def coerce_text_param_value(val: str) -> Any:
    """Parse free-text editor fields (unchanged from prior behavior)."""
    val = val.strip()
    if val and val.lstrip("-").replace(".", "").isdigit():
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val
    return val


def read_field_value(param: str, widget: Any) -> Any:
    """Read a ParamInput or ParamSelect widget as a stored param value."""
    from textual.widgets import Select

    if isinstance(widget, Select):
        return select_to_stored(param, widget.value)
    return coerce_text_param_value(widget.value)
