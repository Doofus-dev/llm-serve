"""Hub-style outline buttons for dialog and editor actions."""

from __future__ import annotations

from textual.containers import Horizontal

# Outline buttons with rounded corners and a dark surface fill.
# Overrides Textual's default beveled (-style-default) button chrome app-wide.
ACTION_BUTTON_CSS = """
ActionBar {
    height: auto;
    min-height: 3;
    align: left middle;
}

Button,
Button.-style-default,
Button.-style-flat {
    height: 3;
    min-width: 0;
    width: auto;
    padding: 0 1;
    margin: 0 1 0 0;
    border: round $primary;
    border-top: round $primary;
    border-bottom: round $primary;
    background: $surface;
    color: $text;
    text-style: none;
}

Button:hover,
Button:focus,
Button.-style-default:hover,
Button.-style-default:focus,
Button.-style-flat:hover,
Button.-style-flat:focus {
    background: $surface-darken-1;
    border: round $primary;
    border-top: round $primary;
    border-bottom: round $primary;
}

Button.-primary,
Button.-style-default.-primary,
Button.-style-flat.-primary {
    border: round $primary;
    border-top: round $primary;
    border-bottom: round $primary;
    color: $primary;
    background: $surface;
}

Button.-primary:hover,
Button.-primary:focus,
Button.-style-default.-primary:hover,
Button.-style-default.-primary:focus,
Button.-style-flat.-primary:hover,
Button.-style-flat.-primary:focus {
    background: $surface-darken-1;
    border: round $primary;
    border-top: round $primary;
    border-bottom: round $primary;
    color: $primary;
}

Button.-success,
Button.-style-default.-success,
Button.-style-flat.-success {
    border: round $success;
    border-top: round $success;
    border-bottom: round $success;
    color: $success;
    background: $surface;
}

Button.-success:hover,
Button.-success:focus,
Button.-style-default.-success:hover,
Button.-style-default.-success:focus,
Button.-style-flat.-success:hover,
Button.-style-flat.-success:focus {
    background: $surface-darken-1;
    border: round $success;
    border-top: round $success;
    border-bottom: round $success;
    color: $success;
}

Button.-error,
Button.-style-default.-error,
Button.-style-flat.-error {
    border: round $error;
    border-top: round $error;
    border-bottom: round $error;
    color: $error;
    background: $surface;
}

Button.-error:hover,
Button.-error:focus,
Button.-style-default.-error:hover,
Button.-style-default.-error:focus,
Button.-style-flat.-error:hover,
Button.-style-flat.-error:focus {
    background: $surface-darken-1;
    border: round $error;
    border-top: round $error;
    border-bottom: round $error;
    color: $error;
}

Button:disabled,
Button.-style-default:disabled,
Button.-style-flat:disabled {
    opacity: 0.45;
    background: $surface;
}
"""


class ActionBar(Horizontal):
    """Horizontal row of outline action buttons."""

    DEFAULT_CSS = """
    ActionBar {
        height: auto;
        min-height: 3;
        align: left middle;
    }
    """
