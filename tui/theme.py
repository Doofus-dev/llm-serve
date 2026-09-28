"""Cohesive accent palette + card framing for the llm-serve TUI.

Item 1 (palette tokens) lives here as a single source of truth: the CSS
tokens below are defined once in ``PALETTE_CSS`` and applied to every
theme, while the Rich markup styles used by widget code come from the
same palette so CSS and Rich text stay in sync.

Item 2 (card framing) is the ``CARD_CSS`` block: each of the three
right-panel zones gets a rounded border in ``$border``, a title-bar row
in ``$text-muted`` with a hairline separator, and 1-cell padding.
"""

from __future__ import annotations

# --- Palette tokens -----------------------------------------------------
# CSS tokens use hex for precise control; Rich styles use named colors
# so existing tests that pin exact style strings continue to pass.

BG = "#101214"          # near-black surface
PANEL = "#1a1d21"       # slightly lighter card background
BORDER = "#3a4046"      # muted gray card edges
ACCENT = "cyan"         # single primary hue for interactive/focused
OK = "green"            # green health state
WARN = "yellow"         # amber health state
ERR = "red"             # red health state
TEXT = "white"          # primary text tier
TEXT_MUTED = "grey"     # secondary text tier
TEXT_DIM = "darkgrey"   # faint text tier

# --- Rich markup styles (what widget code uses) --------------------------
# Named colors match the Rich color names used in existing tests.

ACCENT_STYLE = "bold cyan"
OK_STYLE = "bold green"
WARN_STYLE = "bold yellow"
ERR_STYLE = "bold red"
DIM_STYLE = "dim"
BOLD_STYLE = "bold"
# --- CSS: palette tokens + card framing ----------------------------------
# CSS tokens use hex values for precise color control.

PALETTE_CSS = """\
/* llm-serve accent palette — applied to every theme */
@theme {
    $bg: #101214;
    $panel: #1a1d21;
    $border: #3a4046;
    $accent: #4db3c4;
    $ok: #58c06d;
    $warn: #d8a63f;
    $err: #d9534f;
    $text: #d0d4d8;
    $text-muted: #8a929c;
    $text-dim: #5c646c;
}
"""

CARD_CSS = """
#status,
#config,
#logs {
    border: round $border;
    padding: 0 1;
    background: $panel;
}

#status .card-title,
#config .card-title,
#logs .card-title {
    height: 1;
    margin: 0;
    padding: 0 1;
    color: $text-muted;
    text-style: bold;
    border-bottom: solid $border;
    background: $panel;
}
"""


def theme_css(theme: str) -> str:
    """Palette + card framing CSS to append to the app stylesheet."""
    return f"/* llm-serve theme: {theme} */\n" + PALETTE_CSS + CARD_CSS


def palette_css(theme: str = "any") -> str:
    """Palette tokens only (for callers that don't need card framing)."""
    return f"/* llm-serve palette: {theme} */\n" + PALETTE_CSS


def card_title(text: str) -> str:
    """Rich markup for a card title-bar row: muted bold text with a hairline."""
    return f"[bold {TEXT_MUTED}]{text}[/]\n[{'─' * 60}]"
