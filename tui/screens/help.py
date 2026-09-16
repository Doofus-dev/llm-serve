"""F1 help modal — keys, panels, and tips."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static


HELP_BINDINGS = [
    Binding("escape", "close_help", "Close"),
    Binding("f1", "close_help", "Close"),
]


class HelpScreen(ModalScreen[None]):
    """Full-screen help overlay with keys, panel descriptions, and tips."""

    CSS = """\
    HelpScreen {
        align: center middle;
    }
    #help-modal {
        width: 90%;
        height: 85%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    #help-modal .help-title {
        height: 1;
        margin: 0;
        padding: 0 1;
        color: $foreground;
        text-style: bold;
        border-bottom: solid $border;
        background: $panel;
    }
    #help-scroll {
        height: 1fr;
        overflow-y: auto;
        padding: 1 1;
    }
    .help-section {
        color: $accent;
        text-style: bold;
        margin-top: 2;
        margin-bottom: 0;
    }
    .help-section:first-child {
        margin-top: 0;
    }
    .help-row {
        color: $foreground;
        margin-left: 2;
    }
    .help-key {
        color: $primary;
    }
    .help-desc {
        color: $text-muted;
    }
    .help-tip {
        color: $text-muted;
        margin-left: 2;
    }
    """

    BINDINGS = HELP_BINDINGS

    def compose(self) -> ComposeResult:
        with Vertical(id="help-modal"):
            yield Label("llm-serve — Help", classes="help-title")
            with Vertical(id="help-scroll"):
                # --- Keys section ---
                yield Label("Keys", classes="help-section")
                yield Static(
                    "Navigation\n"
                    "  Tab        Models ↔ Aliases\n"
                    "  ↑↓         Move selection\n"
                    "  1-5        Activate preset (or pin it on an alias)",
                    classes="help-row",
                )
                yield Static(
                    "Run (models or aliases pane)\n"
                    "  L          Launch server\n"
                    "  S          Stop server",
                    classes="help-row",
                )
                yield Static(
                    "Models pane (model or preset selected)\n"
                    "  E          Edit profile / preset\n"
                    "  P          Pick quantization\n"
                    "  N          New preset\n"
                    "  D          Delete",
                    classes="help-row",
                )
                yield Static(
                    "Aliases pane\n"
                    "  E          Rename alias\n"
                    "  Esc        Cancel rename dialog\n"
                    "  N          New alias\n"
                    "  D          Delete alias\n"
                    "  ←→         Change target model",
                    classes="help-row",
                )
                yield Static(
                    "Next launch / logs\n"
                    "  R          Remote on/off (next launch)\n"
                    "  V          Log verbosity (next launch)\n"
                    "  O          Extra info lines on/off",
                    classes="help-row",
                )
                yield Static(
                    "App\n"
                    "  H          Hub (browse/download models)\n"
                    "  T          Cycle theme\n"
                    "  F1         This help\n"
                    "  Q          Quit",
                    classes="help-row",
                )

                # --- Panels section ---
                yield Label("Panels", classes="help-section")
                yield Static(
                    "Status bar (top right)\n"
                    "  Server state, model name, VRAM gauge with percentage\n"
                    "  and health label. Green = healthy, yellow = warning,\n"
                    "  red = critical.",
                    classes="help-row",
                )
                yield Static(
                    "Throughput chart (below status)\n"
                    "  Tokens/sec over time with baseline comparison line.\n"
                    "  The filled area shows current throughput; the dashed\n"
                    "  line is your recorded baseline for that model.",
                    classes="help-row",
                )
                yield Static(
                    "Log panel (bottom)\n"
                    "  Server output in real time. Cycle verbosity with V:\n"
                    "  TRACE → INFO → WARN → ERROR. Toggle extra info lines\n"
                    "  with O (context length, cache hits, etc.).",
                    classes="help-row",
                )
                yield Static(
                    "Model nav (left, top)\n"
                    "  List of models and presets. Green dot = has baseline,\n"
                    "  dim dot = no baseline yet. Select a model or preset\n"
                    "  to launch it.",
                    classes="help-row",
                )
                yield Static(
                    "Alias nav (left, bottom)\n"
                    "  Named launch shortcuts pointing at a model + preset.\n"
                    "  Use ←→ to change the target model for an alias.",
                    classes="help-row",
                )

                # --- Tips section ---
                yield Label("Tips", classes="help-section")
                yield Static(
                    "• Switch presets with 1-5 while a model is selected —\n"
                    "  the server restarts with the new preset's parameters.",
                    classes="help-tip",
                )
                yield Static(
                    "• Press L to launch, S to stop. The status bar shows\n"
                    "  live VRAM usage and throughput as tokens stream.",
                    classes="help-tip",
                )
                yield Static(
                    "• Press E on a model or preset to edit its parameters.\n"
                    "  Use F1 inside the editor for parameter help.",
                    classes="help-tip",
                )
                yield Static(
                    "• Press H to open the Hub and browse/download models\n"
                    "  from Hugging Face.",
                    classes="help-tip",
                )
                yield Static(
                    "• Remote mode (R) launches the server on a remote host\n"
                    "  instead of locally. Configure in settings.",
                    classes="help-tip",
                )

    def action_close_help(self) -> None:
        self.dismiss(None)
