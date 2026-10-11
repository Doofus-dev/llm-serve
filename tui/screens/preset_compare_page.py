"""Side-by-side preset metrics comparison (key x)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
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
from tui.data.presets import Preset, PresetStore, get_preset
from tui.screens.compare import format_diff_summary
from tui.widgets.action_bar import ActionBar
from tui.widgets.config import build_preset_grid


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
    #pcp-columns {
        height: 1fr;
        width: 1fr;
    }
    .pcp-side {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        border: round $border;
        background: $panel;
    }
    .pcp-side-title {
        height: auto;
        color: $accent;
        text-style: bold;
    }
    .pcp-side-grid {
        height: auto;
    }
    #pcp-metrics {
        height: auto;
        max-height: 12;
        padding: 0 1;
        color: $foreground;
        border: round $border;
        background: $panel;
    }
    #pcp-diff-wrap {
        height: auto;
        max-height: 10;
    }
    #pcp-diff {
        height: auto;
        padding: 0 1;
        color: $foreground;
        border: round $border;
        background: $panel;
    }
    #pcp-help {
        height: auto;
        color: $text-muted;
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
            with Horizontal(id="pcp-columns"):
                with VerticalScroll(classes="pcp-side", id="pcp-left"):
                    yield Static("", classes="pcp-side-title", id="pcp-left-title")
                    yield Static("", classes="pcp-side-grid", id="pcp-left-grid")
                with VerticalScroll(classes="pcp-side", id="pcp-right"):
                    yield Static("", classes="pcp-side-title", id="pcp-right-title")
                    yield Static("", classes="pcp-side-grid", id="pcp-right-grid")
            yield Static("", id="pcp-metrics")
            with VerticalScroll(id="pcp-diff-wrap"):
                yield Static("", id="pcp-diff")
            yield Static(
                "[dim]Esc back · green = better · red = worse[/]",
                id="pcp-help",
            )
            with ActionBar(id="pcp-actions"):
                yield Button("Back", id="pcp-back")

    def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_measured_columns)

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

    def _fill_content(self) -> None:
        try:
            title = self.query_one("#pcp-title", Label)
            status = self.query_one("#pcp-status", Static)
            left_title = self.query_one("#pcp-left-title", Static)
            right_title = self.query_one("#pcp-right-title", Static)
            left_grid = self.query_one("#pcp-left-grid", Static)
            right_grid = self.query_one("#pcp-right-grid", Static)
            metrics = self.query_one("#pcp-metrics", Static)
            diff = self.query_one("#pcp-diff", Static)
        except NoMatches:
            return

        left_display = str(self.left.identity.get("display") or self.left.model)
        right_display = str(self.right.identity.get("display") or self.right.model)
        title.update(
            f"[bold]Compare presets[/]  {left_display}/{self.left.quant}  vs  "
            f"{right_display}/{self.right.quant}"
        )
        if self.left.model != self.right.model:
            status.update(
                "[dim]different models — metrics not directly comparable[/]"
            )
        else:
            status.update("")

        left_title.update(Text(self.left.title, style="bold cyan"))
        right_title.update(Text(self.right.title, style="bold cyan"))
        left_grid.update(
            build_preset_grid(self.left.identity, self.left.preset, self.models_dir)
        )
        right_grid.update(
            build_preset_grid(self.right.identity, self.right.preset, self.models_dir)
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

    def action_close_page(self) -> None:
        self.dismiss(None)
