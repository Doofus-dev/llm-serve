"""Modal quant picker for a model profile — full-width Hub-style file table."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Select, Static

from tui.data.gpu import GPUStats, query_gpu
from tui.data.hf import HubFile, list_repo_ggufs
from tui.data.models_json import ModelConfig, Registry, merge_repo_catalog, save_registry
from tui.data.quant_table import QuantFileRow, build_quant_file_rows, fmt_downloaded, fmt_size
from tui.data.vram import fmt_memory_mb


class QuantPickerScreen(ModalScreen[str | None]):
    """Pick a quant; returns quant_id or None. Undownloaded picks trigger download via callback."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("left", "context_prev", "Prev context"),
        ("right", "context_next", "Next context"),
        ("[", "offload_prev", "Less GPU"),
        ("]", "offload_next", "More GPU"),
    ]

    CSS = """
    QuantPickerScreen {
        align: center middle;
    }

    #quant-picker {
        width: 100%;
        height: 100%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }

    #quant-picker-title {
        height: 1;
        color: $accent;
    }

    #quant-picker-status {
        height: 1;
    }

    #quant-context-controls {
        height: 3;
        min-height: 3;
    }

    #quant-context-controls Horizontal {
        height: 3;
        align: left middle;
    }

    #quant-context-controls Select {
        height: 3;
        width: 26;
        border: none;
        background: transparent;
        color: $accent;
    }

    #quant-context-controls Label {
        width: auto;
        height: 1;
        margin: 0 1 0 0;
    }

    #quant-table {
        height: 1fr;
        min-height: 10;
    }

    #quant-picker-actions {
        height: 3;
        layout: horizontal;
        align: left middle;
    }

    #quant-picker-actions Button {
        margin-right: 1;
    }

    #quant-picker-help {
        height: 2;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        model_name: str,
        cfg: ModelConfig,
        registry: Registry,
        models_dir: Path,
        registry_path: Path,
        *,
        baselines_path: Path | None = None,
        on_download: Callable[[str, str, int], None] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.cfg = cfg
        self.registry = registry
        self.models_dir = models_dir
        self.registry_path = registry_path
        self.baselines_path = baselines_path
        self.on_download = on_download
        self._rows: list[QuantFileRow] = []
        self._files: list[HubFile] = []
        self._load_id = 0
        self.gpu = GPUStats()
        self.context_tokens = 65_536
        self.context_options: list[int] = [32_768, 65_536, 131_072, 262_144]
        self.offload_ratio = 1.0
        self.offload_options = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._syncing_controls = False

    def compose(self) -> ComposeResult:
        with Vertical(id="quant-picker"):
            yield Label("", id="quant-picker-title")
            yield Static("[dim]Loading quants…[/]", id="quant-picker-status")
            with Horizontal(id="quant-context-controls"):
                yield Label("Context:")
                yield Select(
                    [(self._fmt_context(v), v) for v in self.context_options],
                    value=self.context_tokens,
                    id="context-select",
                    allow_blank=False,
                )
                yield Label("Offload:")
                yield Select(
                    [(self._fmt_offload(r), r) for r in self.offload_options],
                    value=self.offload_ratio,
                    id="offload-select",
                    allow_blank=False,
                )
                yield Static("", id="hardware-summary")
            yield DataTable(id="quant-table", cursor_type="row", zebra_stripes=True)
            yield Static(
                "[dim]←→ context · [ ] offload · Enter select · "
                "● on disk · — download on select · "
                "[green]●[/] fit · [yellow]⚠[/] tight · [red]●[/] too large[/]",
                id="quant-picker-help",
            )
            with Horizontal(id="quant-picker-actions"):
                yield Button("Select", variant="primary", id="select")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#quant-picker-title", Label).update(
            f"[bold]Quant — {self.cfg.display}[/]  [dim]({self.model_name})[/]"
        )
        self.gpu = query_gpu()
        try:
            model_ctx = int(self.cfg.params.get("ctx", 65_536))
            if model_ctx >= 32_768 and model_ctx not in self.context_options:
                self.context_options = sorted(set(self.context_options + [model_ctx]))
            self.context_tokens = min(model_ctx, max(self.context_options))
        except (TypeError, ValueError):
            pass
        self._render_estimate_controls()
        self._load_id += 1
        self._fetch_files(self._load_id)

    @staticmethod
    def _fmt_context(tokens: int) -> str:
        return f"{tokens // 1024}K" if tokens < 1_048_576 else f"{tokens // 1_048_576}M"

    @staticmethod
    def _fmt_offload(ratio: float) -> str:
        if ratio <= 0:
            return "CPU"
        return f"{int(ratio * 100)}%"

    def _render_estimate_controls(self) -> None:
        self._syncing_controls = True
        try:
            ctx_select = self.query_one("#context-select", Select)
            ctx_select.set_options(
                [(self._fmt_context(v), v) for v in self.context_options]
            )
            if self.context_tokens in self.context_options:
                ctx_select.value = self.context_tokens
            offload_select = self.query_one("#offload-select", Select)
            offload_select.set_options(
                [(self._fmt_offload(r), r) for r in self.offload_options]
            )
            offload_select.value = self.offload_ratio
        finally:
            self._syncing_controls = False
        if self.gpu.vram_total_mb:
            free = max(0.0, self.gpu.vram_total_mb - self.gpu.vram_used_mb)
            pool = "unified pool" if self.gpu.unified else "VRAM"
            summary = (
                f"{self.gpu.name}: {fmt_memory_mb(self.gpu.vram_total_mb)} {pool} · "
                f"{fmt_memory_mb(free)} free"
            )
        else:
            summary = "GPU VRAM unavailable"
        self.query_one("#hardware-summary", Static).update(summary)

    def _refresh_table(self) -> None:
        source = self.cfg.params.get("source")
        author = ""
        if isinstance(source, dict):
            author = str(source.get("author") or "")

        self._rows = build_quant_file_rows(
            self._files,
            gpu=self.gpu,
            context_tokens=self.context_tokens,
            offload_ratio=self.offload_ratio,
            baselines_path=self.baselines_path,
            models_dir=self.models_dir,
            author=author,
        )

        table = self.query_one("#quant-table", DataTable)
        table.clear(columns=True)
        table.add_column("Disk", width=4, key="disk")
        table.add_column("Quant / file", width=28, key="file")
        table.add_column("File size", width=9, key="size")
        table.add_column("Est. VRAM", width=16, key="vram")
        table.add_column("Act. VRAM", width=9, key="act_vram")
        table.add_column("Est. t/s", width=9, key="speed")
        table.add_column("Act. t/s", width=8, key="act_speed")

        active = self.cfg.active_quant
        cursor_row = 0
        for idx, row in enumerate(self._rows):
            disk = fmt_downloaded(row.downloaded)
            if row.downloaded:
                disk = f"[green]{disk}[/]"
            table.add_row(
                disk,
                row.path,
                fmt_size(row.size),
                row.vram_cell,
                row.act_vram,
                row.est_tps,
                row.act_tps,
                key=row.quant_id,
            )
            if row.quant_id == active:
                cursor_row = idx

        status = self.query_one("#quant-picker-status", Static)
        if self._rows:
            status.update(f"[dim]{len(self._rows)} quants · sorted by estimated VRAM[/]")
            table.move_cursor(row=cursor_row)
        else:
            status.update("[dim]No quants found for this repo[/]")

    @work(thread=True)
    def _fetch_files(self, load_id: int) -> None:
        params = dict(self.cfg.params)
        source = params.get("source")
        repo = source.get("repo") if isinstance(source, dict) else None
        author = str(source.get("author") or "") if isinstance(source, dict) else ""

        remote_files: list[HubFile] = []
        error: str | None = None
        if repo:
            remote_files, error = list_repo_ggufs(repo)
            if remote_files:
                merge_repo_catalog(
                    params, [(f.path, f.size) for f in remote_files]
                )
                self.cfg.params["quants"] = params["quants"]
                save_registry(self.registry_path, self.registry)

        # Ensure local-only quants appear even if Hub list fails
        quants = params.get("quants") or {}
        seen_paths = {f.path for f in remote_files}
        for qid, entry in quants.items():
            if not isinstance(entry, dict):
                continue
            fn = str(entry.get("filename") or qid)
            if fn in seen_paths:
                continue
            size = 0
            if author:
                from tui.data.quant_table import local_file_size

                size = local_file_size(self.models_dir, author, fn)
            remote_files.append(HubFile(path=fn, size=size))
            seen_paths.add(fn)

        self.app.call_from_thread(self._finish_fetch, load_id, remote_files, error)

    def _finish_fetch(
        self, load_id: int, files: list[HubFile], error: str | None
    ) -> None:
        if load_id != self._load_id:
            return
        self._files = files
        if error and not files:
            self.query_one("#quant-picker-status", Static).update(
                f"[bold $error]{error}[/]"
            )
        self._refresh_table()
        self.query_one("#quant-table", DataTable).focus()

    def _change_context(self, delta: int) -> None:
        if self.context_tokens not in self.context_options:
            self.context_options = sorted(set(self.context_options + [self.context_tokens]))
        index = self.context_options.index(self.context_tokens)
        new_index = max(0, min(len(self.context_options) - 1, index + delta))
        self.context_tokens = self.context_options[new_index]
        self._render_estimate_controls()
        self._refresh_table()

    def _change_offload(self, delta: int) -> None:
        index = self.offload_options.index(self.offload_ratio)
        new_index = max(0, min(len(self.offload_options) - 1, index + delta))
        self.offload_ratio = self.offload_options[new_index]
        self._render_estimate_controls()
        self._refresh_table()

    def action_context_prev(self) -> None:
        self._change_context(-1)

    def action_context_next(self) -> None:
        self._change_context(1)

    def action_offload_prev(self) -> None:
        self._change_offload(-1)

    def action_offload_next(self) -> None:
        self._change_offload(1)

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._syncing_controls:
            return
        if event.select.id == "context-select" and isinstance(event.value, int):
            self.context_tokens = event.value
            self._refresh_table()
        elif event.select.id == "offload-select" and isinstance(event.value, (int, float)):
            self.offload_ratio = float(event.value)
            self._refresh_table()

    def _selected_row(self) -> QuantFileRow | None:
        table = self.query_one("#quant-table", DataTable)
        if table.cursor_row is None or not self._rows:
            return None
        row_key = table.get_row_at(table.cursor_row)[1]
        # row[1] is path after disk column
        path = str(row_key)
        for item in self._rows:
            if item.path == path:
                return item
        return None

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self._confirm()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._confirm()

    def _confirm(self) -> None:
        picked = self._selected_row()
        if not picked:
            self.notify("Select a quant", severity="warning")
            return
        if picked.downloaded:
            self.dismiss(picked.quant_id)
            return
        if self.on_download:
            self.on_download(self.model_name, picked.path, picked.size)
            self.dismiss(None)
        else:
            self.notify("Download handler unavailable", severity="error")
