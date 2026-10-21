"""Side-by-side preset metrics comparison (key x)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
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
from tui.data.models_json import ModelConfig, Registry, quant_entry_rel_paths
from tui.data.presets import (
    Preset,
    PresetStore,
    get_preset,
    list_presets_for_quant,
)
from tui.widgets.action_bar import ActionBar
from tui.widgets.config import build_preset_column, build_preset_column_wide


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
        if isinstance(entry, dict):
            rels = quant_entry_rel_paths(entry)
            if rels:
                identity["file"] = rels[0]
    return identity


class PresetComparePage(ModalScreen[None]):
    """Compare two presets' measured metrics and colored config values."""

    BINDINGS = [
        Binding("escape", "close_page", "Back"),
        Binding("up", "cycle_preset(-1)", "Previous"),
        Binding("down", "cycle_preset(1)", "Next"),
    ]

    CSS = """\
    PresetComparePage {
        background: $surface;
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
    #pcp-scroll {
        height: 1fr;
        width: 100%;
        overflow-y: auto;
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
        padding: 1 2;
        color: $foreground;
        border: round $border;
        background: $panel;
        margin-bottom: 1;
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
            with ScrollableContainer(id="pcp-scroll"):
                yield Static("", id="pcp-status")
                yield Static("", id="pcp-metrics")
                yield Static("", id="pcp-config-title")
                yield Static("", id="pcp-config-grid")
            yield Static(
                "[dim]Esc back · ↑↓ cycle selected preset · running stays · green = better · red = worse[/]",
                id="pcp-help",
            )
            with ActionBar(id="pcp-actions"):
                yield Button("Back", id="pcp-back")

    def on_mount(self) -> None:
        self.query_one("#pcp-scroll", ScrollableContainer).can_focus = False
        self.call_after_refresh(self.refresh_measured_columns)

    def on_key(self, event: Key) -> None:
        """Cycle presets with arrow keys regardless of widget focus."""
        if event.key in {"up", "down"}:
            direction = 1 if event.key == "down" else -1
            self.action_cycle_preset(direction)
            event.stop()
            event.prevent_default()

    def refresh_measured_columns(self) -> None:
        """Reload baselines / presets and redraw metrics and config."""
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

    def _cycle_side(self) -> PresetCompareSide:
        """The side arrows move. Running stays pinned; otherwise cycle the right."""
        if self.left.role == "running":
            return self.right
        if self.right.role == "running":
            return self.left
        return self.right

    def _cycle_options(self) -> list[int]:
        """Slots on the cycling side's model/quant."""
        if self.store is None:
            return []
        side = self._cycle_side()
        return sorted(list_presets_for_quant(self.store, side.model, side.quant))

    def _fill_content(self) -> None:
        try:
            title = self.query_one("#pcp-title", Label)
            status = self.query_one("#pcp-status", Static)
            metrics = self.query_one("#pcp-metrics", Static)
            config_title = self.query_one("#pcp-config-title", Static)
            config_grid = self.query_one("#pcp-config-grid", Static)
        except NoMatches:
            return

        left_display = str(self.left.identity.get("display") or self.left.model)
        right_display = str(self.right.identity.get("display") or self.right.model)
        if self.left.model == self.right.model and self.left.quant == self.right.quant:
            scope = f"[dim]{left_display}/{self.left.quant}[/]"
        else:
            scope = (
                f"[dim]{left_display}/{self.left.quant}[/]  vs  "
                f"[dim]{right_display}/{self.right.quant}[/]"
            )
        title.update(
            f"[bold]Compare presets[/]  [bold cyan]{self.left.preset.name}[/]  vs  "
            f"[bold cyan]{self.right.preset.name}[/]  {scope}"
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
        # Use two-column layout when terminal is wide enough
        width = self.app.size.width
        if width >= 100:
            config_grid.update(
                build_preset_column_wide(
                    (self.left.identity, self.left.preset),
                    (self.right.identity, self.right.preset),
                    self.models_dir,
                    width,
                )
            )
        else:
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
                left_label=self.left.title,
                right_label=self.right.title,
            )
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pcp-back":
            self.action_close_page()

    def action_cycle_preset(self, direction: int) -> None:
        """Flip the selected (or idle-right) side; keep a running preset pinned."""
        if self.store is None:
            return
        side = self._cycle_side()
        options = self._cycle_options()
        if not options:
            self.notify(
                "No other presets to compare against", severity="warning"
            )
            return
        current = next(
            (i for i, slot in enumerate(options) if slot == side.slot), None
        )
        if current is None:
            current = 0
        index = (current + direction) % len(options)
        slot = options[index]
        preset = get_preset(self.store, side.model, side.quant, slot)
        if preset is None:
            return
        side.slot = slot
        side.preset = preset
        self.refresh_measured_columns()

    def action_close_page(self) -> None:
        self.dismiss(None)
