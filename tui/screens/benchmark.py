"""Benchmark page: run llama-bench on catalogued GGUFs and compare hosts."""

from __future__ import annotations

import asyncio

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Label, Select, Static

from tui.data.benchmarks import (
    DEFAULT_N_GEN,
    DEFAULT_N_PROMPT,
    DEFAULT_REPETITIONS,
    BenchConfig,
    BenchRow,
    BenchTarget,
    HardwareFingerprint,
    LlamaBenchError,
    LlamaBenchMissing,
    catalogued_ggufs,
    collect_fingerprint,
    format_speed,
    hostnames_in,
    load_benchmarks,
    record_benchmark,
    require_llama_bench,
    row_from_run,
    rows_for_table,
    run_llama_bench,
)
from tui.data.gpu import GPUStats
from tui.data.models_json import Registry
from tui.paths import AppPaths
from tui.widgets.action_bar import ACTION_BUTTON_CSS, ActionBar

RESULT_COLUMNS = (
    "Host",
    "Model",
    "Quant",
    "Prompt tokens",
    "Gen tokens",
    "Repeats",
    "Prompt t/s",
    "Gen t/s",
    "gpu_layers",
    "threads",
    "backend",
    "GPU",
    "When",
)

PICKER_COLUMNS = ("Pick", "Model", "Quant", "File", "Status")

ALL_HOSTS = "__all__"


class BenchmarkScreen(Screen):
    """Pick local GGUFs, run llama-bench off the UI thread, compare hosts."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("space", "toggle_pick", "Toggle", show=False),
        Binding("a", "pick_all", "All", show=False),
        Binding("r", "run_bench", "Run selected"),
        Binding("ctrl+r", "run_all", "Run all"),
    ]

    CSS = """
    BenchmarkScreen {
        align: center middle;
    }
"""
    CSS += ACTION_BUTTON_CSS
    CSS += """
    #bench-panel {
        width: 100%;
        height: 100%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }

    #bench-title {
        height: 1;
        color: $accent;
        content-align: left middle;
    }

    #bench-hardware {
        height: auto;
        min-height: 1;
        color: $text-muted;
    }

    #bench-status {
        height: auto;
        min-height: 1;
    }

    #bench-knobs {
        height: auto;
        min-height: 3;
        align: left middle;
    }

    #bench-knobs Label {
        width: auto;
        height: 3;
        content-align: left middle;
        margin: 0 1 0 0;
    }

    #bench-knobs Input {
        width: 10;
        height: 3;
        margin: 0 2 0 0;
        border: round $accent;
        background: transparent;
        color: $foreground;
        padding: 0 1;
    }

    #bench-host-filter {
        width: 28;
        height: 3;
        margin: 0 1 0 0;
        border: none;
        background: transparent;
    }

    #bench-picker {
        height: 1fr;
        min-height: 6;
        border: round $accent;
    }

    #bench-results {
        height: 1fr;
        min-height: 6;
        border: round $accent;
    }

    #bench-help {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        registry: Registry,
        paths: AppPaths,
        gpu: GPUStats | None = None,
        fingerprint: HardwareFingerprint | None = None,
        initial_model: str | None = None,
        runner=None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.paths = paths
        self.gpu = gpu
        self.fingerprint = fingerprint or collect_fingerprint(gpu)
        self.initial_model = initial_model
        self.runner = runner
        self.targets: list[BenchTarget] = []
        self.picked: set[str] = set()
        self.status_by_key: dict[str, str] = {}
        self.rows: list[BenchRow] = []
        self._bench_active = False
        self._host_filter = ALL_HOSTS

    def compose(self) -> ComposeResult:
        with Vertical(id="bench-panel"):
            yield Label("[bold]Benchmark[/]  offline Prompt t/s and Gen t/s", id="bench-title")
            yield Static("", id="bench-hardware")
            yield Static("", id="bench-status")
            with Horizontal(id="bench-knobs"):
                yield Label("Prompt tokens")
                yield Input(str(DEFAULT_N_PROMPT), id="bench-pp")
                yield Label("Gen tokens")
                yield Input(str(DEFAULT_N_GEN), id="bench-n")
                yield Label("Repeats")
                yield Input(str(DEFAULT_REPETITIONS), id="bench-r")
                yield Label("Host")
                yield Select(
                    [("All hosts", ALL_HOSTS)],
                    value=ALL_HOSTS,
                    allow_blank=False,
                    id="bench-host-filter",
                )
            with ActionBar():
                yield Button("Run selected", variant="success", id="run")
                yield Button("Run all", id="run-all")
                yield Button("Close", id="close")
            yield Label("Local GGUFs", classes="help-section")
            yield DataTable(id="bench-picker", cursor_type="row")
            yield Label("Results", classes="help-section")
            yield DataTable(id="bench-results", cursor_type="row")
            yield Static(
                "Space: pick  R: run selected (cursor if none)  Ctrl+R: run all one at a time  Esc: close  —  "
                "Same Prompt/Gen/Repeats on this host rewrites the last run. Change a knob to keep both.",
                id="bench-help",
            )

    def on_mount(self) -> None:
        self.query_one("#bench-hardware", Static).update(
            f"This machine: {self.fingerprint.label()}"
        )
        picker = self.query_one("#bench-picker", DataTable)
        results = self.query_one("#bench-results", DataTable)
        picker.zebra_stripes = True
        results.zebra_stripes = True
        self._reload_catalog()
        if self.initial_model:
            for target in self.targets:
                if target.model == self.initial_model:
                    self.picked.add(target.key)
        self._render_picker()
        self._reload_results()
        self._warn_missing_binary()

    def _warn_missing_binary(self) -> None:
        try:
            require_llama_bench(self.paths.llama_bench)
        except LlamaBenchMissing as exc:
            self._set_status(str(exc), error=True)

    def _set_status(self, message: str, *, error: bool = False) -> None:
        markup = f"[red]{message}[/]" if error else message
        self.query_one("#bench-status", Static).update(markup)

    def _reload_catalog(self) -> None:
        self.targets = catalogued_ggufs(self.registry, self.paths.models_dir)

    def _read_config(self, *, silent: bool = False) -> BenchConfig | None:
        def parse(widget_id: str, default: int, label: str) -> int | None:
            raw = self.query_one(widget_id, Input).value.strip() or str(default)
            try:
                value = int(raw)
            except ValueError:
                if not silent:
                    self.notify(f"{label} must be an integer", severity="error")
                return None
            if value < 0:
                if not silent:
                    self.notify(f"{label} must be ≥ 0", severity="error")
                return None
            return value

        n_prompt = parse("#bench-pp", DEFAULT_N_PROMPT, "Prompt tokens")
        n_gen = parse("#bench-n", DEFAULT_N_GEN, "Gen tokens")
        reps = parse("#bench-r", DEFAULT_REPETITIONS, "Repeats")
        if n_prompt is None or n_gen is None or reps is None:
            return None
        if n_prompt <= 0 and n_gen <= 0:
            if not silent:
                self.notify("Need Prompt tokens or Gen tokens > 0", severity="error")
            return None
        if reps <= 0:
            if not silent:
                self.notify("Repeats must be ≥ 1", severity="error")
            return None
        return BenchConfig(n_prompt=n_prompt, n_gen=n_gen, repetitions=reps)

    def _render_picker(self) -> None:
        table = self.query_one("#bench-picker", DataTable)
        cursor = table.cursor_row
        table.clear(columns=True)
        for label in PICKER_COLUMNS:
            table.add_column(label, key=label.lower())
        if not self.targets:
            self._set_status("No local GGUFs in the model catalog")
            return
        for target in self.targets:
            picked = "●" if target.key in self.picked else "○"
            status = self.status_by_key.get(target.key, "ready")
            table.add_row(
                picked,
                target.display or target.model,
                target.quant,
                target.file_rel,
                status,
                key=target.key,
            )
        if table.row_count and 0 <= cursor < table.row_count:
            table.move_cursor(row=cursor)

    def _reload_results(self) -> None:
        self.rows = load_benchmarks(self.paths.benchmarks_json)
        self._refresh_host_filter()
        self._render_results()

    def _refresh_host_filter(self) -> None:
        select = self.query_one("#bench-host-filter", Select)
        options = [("All hosts", ALL_HOSTS)]
        this = self.fingerprint.hostname
        if this:
            options.append((f"{this} (this machine)", this))
        for name in hostnames_in(self.rows):
            if name != this:
                options.append((name, name))
        current = self._host_filter
        if current not in {value for _label, value in options}:
            current = ALL_HOSTS
            self._host_filter = ALL_HOSTS
        select.set_options(options)
        select.value = current

    def _render_results(self) -> None:
        table = self.query_one("#bench-results", DataTable)
        table.clear(columns=True)
        for label in RESULT_COLUMNS:
            table.add_column(label, key=label.lower().replace(" ", "_"))
        host_filter = "" if self._host_filter == ALL_HOSTS else self._host_filter
        current = self._read_config(silent=True)
        ordered = rows_for_table(
            self.rows,
            this_hostname=self.fingerprint.hostname,
            host_filter=host_filter,
            current=current,
        )
        prev_group: tuple | None = None
        for row in ordered:
            group = (row.model, row.quant, row.n_prompt, row.n_gen, row.repetitions)
            host = row.hostname
            if host == self.fingerprint.hostname:
                host = f"{host} *"
            model = row.display or row.model
            quant = row.quant
            if prev_group == group:
                model = ""
                quant = ""
            table.add_row(
                host,
                model,
                quant,
                str(row.n_prompt),
                str(row.n_gen),
                str(row.repetitions),
                format_speed(row.pp_tps, row.pp_stddev),
                format_speed(row.tg_tps, row.tg_stddev),
                "—" if row.n_gpu_layers is None else str(row.n_gpu_layers),
                "—" if row.n_threads is None else str(row.n_threads),
                row.backend or "—",
                row.gpu_name or "—",
                row.updated_at or "—",
                key=f"{row.hostname}:{row.model}:{row.quant}:{row.n_prompt}:{row.n_gen}:{row.repetitions}:{row.n_depth}",
            )
            prev_group = group

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "bench-host-filter":
            return
        value = event.value
        if value in (None, Select.BLANK):
            return
        self._host_filter = str(value)
        self._render_results()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.action_close()
            return
        if event.button.id == "run":
            self.action_run_bench()
            return
        if event.button.id == "run-all":
            self.action_run_all()

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_toggle_pick(self) -> None:
        table = self.query_one("#bench-picker", DataTable)
        if not table.row_count:
            return
        index = table.cursor_row
        if index < 0 or index >= len(self.targets):
            return
        key = self.targets[index].key
        if key in self.picked:
            self.picked.remove(key)
        else:
            self.picked.add(key)
        self._render_picker()

    def action_pick_all(self) -> None:
        self.picked = {target.key for target in self.targets}
        self._render_picker()

    def _cursor_target(self) -> BenchTarget | None:
        table = self.query_one("#bench-picker", DataTable)
        if not table.row_count:
            return None
        index = table.cursor_row
        if index < 0 or index >= len(self.targets):
            return None
        return self.targets[index]

    def action_run_all(self) -> None:
        self._start_run(list(self.targets), label="all local GGUFs")

    def action_run_bench(self) -> None:
        selected = [target for target in self.targets if target.key in self.picked]
        if not selected:
            cursor = self._cursor_target()
            if cursor is None:
                self.notify("Pick a GGUF or put the cursor on one row", severity="warning")
                return
            selected = [cursor]
            self.picked.add(cursor.key)
            self._render_picker()
        self._start_run(selected, label="selected")

    def _start_run(self, selected: list[BenchTarget], *, label: str) -> None:
        if self._bench_active:
            self.notify("Benchmark already running", severity="warning")
            return
        if not selected:
            self.notify("No local GGUFs to benchmark", severity="warning")
            return
        config = self._read_config()
        if config is None:
            return
        try:
            require_llama_bench(self.paths.llama_bench)
        except LlamaBenchMissing as exc:
            for target in selected:
                self.status_by_key[target.key] = "failed"
            self._render_picker()
            self._set_status(str(exc), error=True)
            self.notify(str(exc), severity="error", timeout=8)
            return
        names = ", ".join(f"{t.display} {t.quant}" for t in selected[:4])
        extra = "" if len(selected) <= 4 else f" +{len(selected) - 4} more"
        self._set_status(
            f"Queue {len(selected)} · {label} · one at a time · "
            f"Prompt {config.n_prompt} · Gen {config.n_gen} · Repeats {config.repetitions} · {names}{extra}"
        )
        self._run_worker(selected, config)

    @work(exclusive=True)
    async def _run_worker(self, selected: list[BenchTarget], config: BenchConfig) -> None:
        self._bench_active = True
        try:
            self.query_one("#run", Button).disabled = True
        except (QueryError, NoMatches):
            pass
        try:
            total = len(selected)
            for index, target in enumerate(selected, start=1):
                self.status_by_key[target.key] = f"running {index}/{total}"
                self._render_picker()
                self._set_status(
                    f"Running {index}/{total} · {target.display} {target.quant} · one at a time · "
                    f"Prompt {config.n_prompt} · Gen {config.n_gen} · Repeats {config.repetitions}"
                )
                try:
                    measurement = await asyncio.to_thread(
                        run_llama_bench,
                        binary=self.paths.llama_bench,
                        model_paths=target.paths,
                        config=config,
                        runner=self.runner,
                    )
                    row = row_from_run(
                        target=target,
                        measurement=measurement,
                        fingerprint=self.fingerprint,
                        config=config,
                    )
                    replaced = await asyncio.to_thread(
                        record_benchmark, self.paths.benchmarks_json, row
                    )
                    self.status_by_key[target.key] = "done"
                    self._reload_results()
                    verb = "replaced last run" if replaced else "saved new run"
                    self._set_status(
                        f"{index}/{total} {verb} · {target.display} {target.quant} · "
                        f"Prompt {row.n_prompt} · Gen {row.n_gen} · Repeats {row.repetitions}"
                    )
                except LlamaBenchError as exc:
                    self.status_by_key[target.key] = "failed"
                    self._render_picker()
                    self._set_status(str(exc), error=True)
                    self.notify(str(exc)[:200], severity="error", timeout=8)
            self._render_picker()
        finally:
            self._bench_active = False
            try:
                self.query_one("#run", Button).disabled = False
            except (QueryError, NoMatches):
                pass
