"""Dashboard key bindings and help text."""

from __future__ import annotations

from textual.binding import Binding


_PRESET_HOTKEYS = (
    Binding("1", "activate_preset(1)", "Preset 1", show=False),
    Binding("2", "activate_preset(2)", "Preset 2", show=False),
    Binding("3", "activate_preset(3)", "Preset 3", show=False),
    Binding("4", "activate_preset(4)", "Preset 4", show=False),
    Binding("5", "activate_preset(5)", "Preset 5", show=False),
)


def build_app_bindings(
    *,
    remote_on: bool = False,
    log_label: str = "TRACE",
    info_label: str = "Info",
) -> list[Binding]:
    """Dashboard footer bindings grouped by function."""
    return [
        # Run server
        Binding("l", "launch", "Launch"),
        Binding("s", "stop", "Stop"),
        # Selected model / preset / alias
        Binding("e", "edit", "Edit"),
        Binding("p", "pick_quant", "Quant"),
        Binding("n", "new", "New"),
        Binding("d", "delete", "Delete"),
        # Next launch + log panel
        Binding("r", "toggle_remote", "Remote ON" if remote_on else "Remote OFF"),
        Binding("v", "cycle_log_verbosity", f"Log {log_label}"),
        Binding("o", "toggle_log_source", info_label),
        # App
        Binding("h", "open_hub", "Hub"),
        Binding("t", "change_theme", "Theme"),
        Binding("f1", "help", "Help"),
        Binding("q", "quit", "Quit"),
        *_PRESET_HOTKEYS,
    ]


SELECTION_ACTIONS = frozenset({"launch", "edit", "pick_quant", "new", "delete"})


def selection_supports_action(
    action: str,
    kind: str | None,
    *,
    models_section: bool = False,
    log_section: bool = False,
) -> bool:
    """Whether a selection-scoped footer action applies to the highlighted item."""
    if log_section and action in SELECTION_ACTIONS:
        return False
    if action == "edit":
        return kind in {"model", "preset", "alias"}
    if action == "pick_quant":
        return models_section and kind in {"model", "preset"}
    if action == "launch":
        return kind in {"model", "preset", "alias"}
    return True


HELP_TEXT = """\
Navigation
  Tab       Models ↔ Aliases
  ↑↓        move selection
  1-5       activate preset (or pin it on an alias)

Run (models or aliases pane)
  L         launch
  S         stop

Models pane (model or preset selected)
  E         edit profile / preset
  P         pick quant
  N         new preset
  D         delete

Aliases pane
  E         rename alias
  Esc       cancel rename dialog
  N         new alias
  D         delete alias
  ←→        change target model

Next launch / logs
  R         remote on/off (next launch)
  V         log verbosity (next launch)
  O         extra info lines on/off

App
  H         Hub
  T         theme
  F1        this help
  Q         quit
"""
