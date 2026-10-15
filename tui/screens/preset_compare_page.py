"""Side-by-side preset metrics comparison (key x)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static
from rich.text import Text

from tui.data.baselines import load_baselines
from tui.data.compare_metrics import (
    compare_presets_metrics,
    format_metrics_table,
    resolve_baseline,
)
from tui.data.models_json import ModelConfig, Registry
from tui.data.presets import (
    Preset,
    PresetStore,
    get_preset,
    list_presets_for_quant,
)
from tui.screens.compare import format_diff_summary
from tui.widgets.action_bar import ActionBar
from tui.widgets.config import build_preset_column


@dataclass
class PresetCompareSide:
    model: str
    quant: str
    slot: int
    identity: dict[str, Any]
    preset: Preset
    role: str = "selected"

    @property
    def title(self) -> str:
        return f"{self.role}  [{self.slot}] {self.preset.name}"


def identity_for_quant(cfg: ModelConfig, quant: str) -> dict[str, Any]:
    """Copy a model's identity with file/quant pointed at this preset's quant."""
    identity = dict(cfg.params)
    identity["active_quant"] = quant
    quants = identity.get("quants")
    if isinstance(quants, dict):
        entry = quants.get(quant) or {}
        if isinstance(entry, dict) and entry.get("file"):
            identity["file"] = str(entry["file"])
    return identity


class PresetComparePage(ModalScreen[None]):
    """Compare two presets' identity, measured metrics, and param diff."""

    BINDINGS = [
        Binding("escape", "close_page", "Back"),
        Binding("up", "cycle_preset(-1)", "Previous"),
        Binding("down", "cycle_preset(1)", "Next"),
    ]

    CSS = """\
    PresetComparePage {
        background: $surface;
        align: center middle;
    }
    #pcp-root {
        height: 100%;
        width: 100%;
        padding: 0 1;
    }
    #pcp-title {
        height: auto;
        color: $accent;
        padding: 0 1;
        border-bottom: solid $border;
    }
    #pcp-status {
        height: auto;
        min-height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #pcp-config-title {
        height: auto;
        color: $accent;
        text-style: bold;
        border: round $border;
        background: $panel;
    }
    #pcp-config-grid {
        height: auto;
        padding: 0 1;
        background: $panel;
    }
    #pcp-metrics {
        height: auto;
        max-height: 12;
        padding: 0 1;
        color: $foreground;
        border: round $border;
        background: $panel;
    }
    #pcp-diff {
        height: auto;
        max-height: 10;
        padding: 0 1;
        border: round $border;
        background: $panel;
    }
    #pcp-help {
        height: auto;
        color: $text-muted;
        min-height: 1;
    }
    """

    def __init__(
        self,
        *,
        left: PresetCompareSide,
        right: PresetCompareSide,
        models_dir: Path | None,
        baselines_path: Path | None = None,
        store: PresetStore | None = None,
        registry: Registry | None = None,
        live: Any | None = None,
        history: list[float] | None = None,
        gpu_name: str = "",
    ) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.models_dir = models_dir
        self.baselines_path = baselines_path
        self.store = store
        self.registry = registry
        self.live = live
        self.history = history
        self.gpu_name = gpu_name

    def compose(self) -> ComposeResult:
        with Vertical(id="pcp-root"):
            yield Label("", id="pcp-title")
            yield Static("", id="pcp-status")
            yield Static("", id="pcp-config-title")
            yield Static("", id="pcp-config-grid")
            yield Static("", id="pcp-metrics")
            yield Static("", id="pcp-diff")
            yield Static(
                "[dim]Esc back · ↑↓ flip preset · green = better · red = worse[/]",
                id="pcp-help",
            )
            with ActionBar(id="pcp-actions"):
                yield Button("Back", id="pcp-back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_measured_columns)

    def on_key(self, event: Key) -> None:
        """Cycle presets with arrow keys regardless of widget focus."""
        if event.key in {"up", "down"}:
            direction = 1 if event.key == "down" else -1
            self.action_cycle_preset(direction)
            event.stop()
            event.prevent_default()

    def refresh_measured_columns(self) -> None:
        """Reload baselines / presets and redraw grids, metrics, and diff."""
        if not self.is_mounted:
            return
        self._refresh_sides()
        self._fill_content()

    def _refresh_sides(self) -> None:
        if self.store is None:
            return
        for side in (self.left, self.right):
            preset = get_preset(self.store, side.model, side.quant, side.slot)
            if preset is not None:
                side.preset = preset
            if self.registry is not None and side.model in self.registry.models:
                side.identity = identity_for_quant(
                    self.registry.models[side.model], side.quant
                )

    def _right_options(self) -> list[int]:
        """Comparable right-side slots: all presets of the same model/quant."""
        if self.store is None:
            return []
        return sorted(
            list_presets_for_quant(self.store, self.left.model, self.left.quant)
        )

    def _fill_content(self) -> None:
        try:
            title = self.query_one("#pcp-title", Label)
            status = self.query_one("#pcp-status", Static)
            config_title = self.query_one("#pcp-config-title", Static)
            config_grid = self.query_one("#pcp-config-grid", Static)
            metrics = self.query_one("#pcp-metrics", Static)
            diff = self.query_one("#pcp-diff", Static)
        except NoMatches:
            return

        left_display = str(self.left.identity.get("display") or self.left.model)
        title.update(
            f"[bold]Compare presets[/]  [bold cyan]{self.left.preset.name}[/]  vs  "
            f"[bold cyan]{self.right.preset.name}[/]  [dim]{left_display}/{self.left.quant}[/]"
        )
        if self.left.model != self.right.model:
            status.update(
                "[dim]different models — metrics not directly comparable[/]"
            )
        else:
            status.update("")

        config_title.update(
            Text(
                f"{self.left.title}  vs  {self.right.title}",
                style="bold cyan",
            )
        )
        config_grid.update(
            build_preset_column(
                (self.left.identity, self.left.preset),
                (self.right.identity, self.right.preset),
                self.models_dir,
            )
        )

        baselines = (
            load_baselines(self.baselines_path) if self.baselines_path else []
        )
        left_run = resolve_baseline(
            self.left.model,
            self.left.quant,
            self.left.slot,
            self.models_dir,
            baselines,
            identity=self.left.identity,
            preset=self.left.preset,
            gpu_name=self.gpu_name,
            live=self.live if self.left.role == "running" else None,
            history=self.history if self.left.role == "running" else None,
        )
        right_run = resolve_baseline(
            self.right.model,
            self.right.quant,
            self.right.slot,
            self.models_dir,
            baselines,
            identity=self.right.identity,
            preset=self.right.preset,
            gpu_name=self.gpu_name,
            live=self.live if self.right.role == "running" else None,
            history=self.history if self.right.role == "running" else None,
        )
        metrics.update(
            format_metrics_table(
                compare_presets_metrics(left_run, right_run),
                left_label=self.left.role,
                right_label=self.right.role,
            )
        )
        diff.update(
            format_diff_summary(
                self.left.preset.name,
                self.right.preset.name,
                self.left.preset.params,
                self.right.preset.params,
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pcp-back":
            self.action_close_page()

    def action_cycle_preset(self, direction: int) -> None:
        """Flip the right side to the next/previous preset of the same quant."""
        if self.store is None:
            return
        options = self._right_options()
        if not options:
            self.notify(
                "No other presets to compare against", severity="warning"
            )
            return
        current = next(
            (i for i, slot in enumerate(options) if slot == self.right.slot), None
        )
        if current is None:
            current = 0
        index = (current + direction) % len(options)
        slot = options[index]
        preset = get_preset(self.store, self.left.model, self.left.quant, slot)
        if preset is None:
            return
        self.right.slot = slot
        self.right.preset = preset
        self.refresh_measured_columns()

    def action_close_page(self) -> None:
        self.dismiss(None)
