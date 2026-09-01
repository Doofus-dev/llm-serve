"""Hugging Face Hub browse and download screen."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Input, Label, Select, Static
from textual.worker import Worker, WorkerState

from tui.data.hf import (
    HF_INSTALL_HINT,
    AuthStatus,
    DownloadPlan,
    HubFile,
    HubRepo,
    auth_login,
    auth_whoami,
    build_download_plan,
    build_source_metadata,
    download_files,
    fmt_size,
    hf_available,
    local_download_bytes,
    list_gguf_repos,
    list_repo_ggufs,
)
from tui.data.gguf import read_gguf_architecture
from tui.data.models_json import (
    Registry,
    create_downloaded_model,
    find_model_by_repo,
    load_registry,
)
from tui.data.quant import family_display
from tui.data.quant_table import build_quant_file_rows
from tui.data.settings import TUISettings, remember_hf_author, save_settings
from tui.data.gpu import GPUStats, query_gpu
from tui.data.vram import fmt_memory_mb


class HFLoginDialog(ModalScreen[tuple[bool, str] | None]):
    """Prompt for a Hugging Face token."""

    def compose(self) -> ComposeResult:
        with Vertical(id="hub-login-dialog"):
            yield Label("[bold]Hugging Face Login[/bold]")
            yield Label("Paste a token from huggingface.co/settings/tokens")
            yield Input(password=True, placeholder="hf_...", id="token")
            with Horizontal():
                yield Button("Login", variant="success", id="login")
                yield Button("Cancel", variant="default", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        token = self.query_one("#token", Input).value.strip()
        ok, message = auth_login(token)
        self.dismiss((ok, message))


class DownloadProfileDialog(ModalScreen[tuple[str, str] | None]):
    """Name + clone-from for a downloaded model profile."""

    def __init__(self, registry: Registry, default_name: str):
        super().__init__()
        self.registry = registry
        self.default_name = default_name

    def compose(self) -> ComposeResult:
        options = [(name, name) for name in self.registry.models.keys()]
        with Vertical(id="hub-download-dialog"):
            yield Label("[bold]Register Downloaded Model[/bold]")
            yield Label("Display name:")
            yield Input(value=self.default_name, placeholder="Qwen 3.8", id="name")
            yield Label("Clone server params from:")
            yield Select(options, id="clone_from", value=options[0][1] if options else Select.BLANK)
            with Horizontal():
                yield Button("Create", variant="success", id="create")
                yield Button("Cancel", variant="default", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        name = self.query_one("#name", Input).value.strip()
        clone_from = self.query_one("#clone_from", Select).value
        if not name:
            self.app.notify("Name cannot be empty", severity="error")
            return
        if clone_from in (None, Select.BLANK):
            self.app.notify("Select a model to clone params from", severity="error")
            return
        self.dismiss((name, str(clone_from)))


class HubTable(DataTable):
    """Row list that yields left/right to the Hub context/offload steppers."""

    def action_cursor_left(self) -> None:
        screen = self.screen
        if getattr(screen, "mode", None) == "files":
            screen.action_context_prev()
            return
        super().action_cursor_left()

    def action_cursor_right(self) -> None:
        screen = self.screen
        if getattr(screen, "mode", None) == "files":
            screen.action_context_next()
            return
        super().action_cursor_right()


class HubScreen(Screen):
    """Browse trending GGUF repos, pick a file, and download into models/<author>/."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("b", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "focus_filters", "Focus filters"),
        Binding("left", "context_prev", "Previous context"),
        Binding("right", "context_next", "Next context"),
        Binding("[", "offload_prev", "Fewer GPU layers"),
        Binding("]", "offload_next", "More GPU layers"),
    ]

    CSS = """
    HubScreen {
        align: center middle;
    }

    #hub-panel {
        width: 100%;
        max-width: 120;
        height: 100%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }

    #hub-title {
        height: 1;
        color: $accent;
        content-align: left middle;
    }

    #hub-status {
        height: 2;
        content-align: left middle;
    }

    #hub-filters Horizontal {
        height: 3;
        align: left middle;
    }

    #hub-filters {
        height: 6;
        min-height: 6;
    }

    #hub-filters Button {
        height: 3;
        min-width: 0;
        width: auto;
        padding: 0 1;
        margin: 0 1 0 0;
        border: round $primary;
        background: transparent;
        color: $text;
    }

    #hub-filters Button:hover,
    #hub-filters Button:focus {
        background: transparent;
    }

    #hub-filters Button.-success {
        border: round $success;
        color: $success;
    }

    #hub-filters Button.-primary {
        border: round $primary;
        color: $primary;
    }

    #hub-filters #select {
        min-width: 0;
    }

    #hub-table {
        height: 1fr;
        min-height: 3;
        border: round $accent;
    }

    #context-controls {
        height: 3;
        align: left middle;
    }

    #context-label,
    #offload-label {
        width: auto;
        height: 3;
        content-align: left middle;
        margin: 0 1 0 0;
    }

    #offload-label {
        margin: 0 1 0 2;
    }

    #context-select,
    #offload-select {
        height: 3;
        width: 14;
        border: none;
        background: transparent;
        color: $accent;
    }

    #hardware-summary {
        width: 1fr;
        height: 3;
        content-align: left middle;
        margin: 0 1;
    }

    #hub-help {
        height: 1;
        min-height: 1;
    }

    #hub-filters Input {
        height: 3;
        min-width: 12;
        width: 1fr;
        border: round $accent;
    }

    #hub-filters Select {
        height: 3;
        width: 26;
        border: none;
        background: transparent;
        color: $accent;
    }

    #hub-filters .field-label {
        width: auto;
        height: 1;
        content-align: left middle;
        margin: 0 1 0 0;
    }

    #hub-login-dialog, #hub-download-dialog {
        width: 70;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1 2;
        align: center middle;
    }
    """

    def __init__(
        self,
        *,
        registry: Registry,
        settings: TUISettings,
        settings_path: Path,
        models_json_path: Path,
        models_dir: Path,
        baselines_path: Path | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.settings = settings
        self.settings_path = settings_path
        self.models_json_path = models_json_path
        self.models_dir = models_dir
        self.baselines_path = baselines_path
        self.on_complete = on_complete
        self.auth_status = AuthStatus(logged_in=False)
        self.repos: list[HubRepo] = []
        self.files: list[HubFile] = []
        self.selected_repo: HubRepo | None = None
        self.mode = "repos"
        self._downloading = False
        self._repo_load_id = 0
        self._file_load_id = 0
        self.gpu = GPUStats()
        self.context_tokens = 65_536
        self.context_options: list[int] = []
        self.offload_ratio = 1.0
        self.offload_options = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._syncing_controls = False

    def compose(self) -> ComposeResult:
        with Vertical(id="hub-panel"):
            yield Static("", id="hub-status")
            yield Label(
                "[bold]Browse Hugging Face GGUF models[/]  "
                "[dim]Tab: fields · ↑↓: list · Enter: open/download · "
                "B: back · Esc: close[/]",
                id="hub-title",
            )
            with Vertical(id="hub-filters"):
                with Horizontal():
                    yield Select([], prompt="Previous authors", id="author_preset", allow_blank=True)
                    yield Input(placeholder="Author (e.g. bartowski)", id="author_filter")
                    yield Input(placeholder="Model name (e.g. Qwen3.6)", id="search_filter")
                with Horizontal():
                    yield Button("Search", variant="primary", id="search")
                    yield Button("Login", id="login")
                    yield Button("Back", id="back")
                    yield Button("Open / download", variant="success", id="select")
                    yield Button("Close", id="close")
            with Horizontal(id="context-controls"):
                yield Label("Context:", id="context-label")
                yield Select(
                    [(self._fmt_context(value), value) for value in (32_768, 65_536)],
                    value=65_536,
                    prompt="Context",
                    id="context-select",
                    allow_blank=False,
                )
                yield Label("Offload:", id="offload-label")
                yield Select(
                    [(self._fmt_offload(ratio), ratio) for ratio in self.offload_options],
                    value=self.offload_ratio,
                    prompt="Offload",
                    id="offload-select",
                    allow_blank=False,
                )
                yield Static("", id="hardware-summary")
            yield HubTable(id="hub-table", cursor_type="row")
            yield Static(
                "[dim]Tab: fields · ←→ context · [ ] offload · Enter open · F filters · "
                "Act. after a local run · "
                "[green]●[/] fit · [yellow]⚠[/] tight · [red]●[/] too large[/]",
                id="hub-help",
            )

    def on_mount(self) -> None:
        self._refresh_author_preset_options()
        self._update_auth_status()
        self.gpu = query_gpu()
        self._update_context_options()
        self.query_one("#context-controls").display = False
        self.query_one("#hub-table", DataTable).focus()
        self._request_repos("", "")

    def _refresh_author_preset_options(self) -> None:
        select = self.query_one("#author_preset", Select)
        options = [(author, author) for author in self.settings.hf_authors]
        select.set_options(options)

    def _set_status(self, message: str) -> None:
        self.query_one("#hub-status", Static).update(message)

    def _update_auth_status(self) -> None:
        if self._downloading:
            return
        if not hf_available():
            self.auth_status = AuthStatus(logged_in=False)
            self._set_status(f"[bold $error]hf CLI not installed[/] — {HF_INSTALL_HINT}")
            return
        self.auth_status = auth_whoami()
        if self.auth_status.logged_in:
            self._set_status(f"[bold $success]Logged in[/] as {self.auth_status.name}")
        else:
            self._set_status(
                "[bold $warning]Not logged in[/] — public repos only; press Login for gated models"
            )

    def _set_mode_repos(self) -> None:
        self.mode = "repos"
        self.selected_repo = None
        self.files = []
        self.query_one("#context-controls").display = False
        self.query_one("#back", Button).disabled = True
        self.query_one("#select", Button).label = "Open / download"
        self._render_repo_table()
        self.query_one("#hub-table", DataTable).focus()

    def _set_mode_files(self, repo: HubRepo) -> None:
        self._repo_load_id += 1
        self.mode = "files"
        self.selected_repo = repo
        self.query_one("#context-controls").display = True
        self.query_one("#back", Button).disabled = False
        self.query_one("#select", Button).label = "Download file"
        self._update_context_options(repo.context_length)
        self._file_load_id += 1
        self._load_files(repo, self._file_load_id)

    def _update_context_options(self, model_max: int | None = None) -> None:
        """Set doubling context stops, capped at the model's limit."""
        maximum = model_max or 65_536
        stops = [32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576]
        self.context_options = [value for value in stops if value <= maximum]
        if maximum >= 32_768 and maximum not in self.context_options:
            self.context_options.append(maximum)
        if not self.context_options:
            self.context_options = [maximum]
        self.context_tokens = min(65_536, self.context_options[-1])
        self._render_estimate_controls()
        if self.mode == "files":
            self._render_file_table()

    def _render_estimate_controls(self) -> None:
        self._syncing_controls = True
        try:
            context_select = self.query_one("#context-select", Select)
            context_select.set_options(
                [(self._fmt_context(value), value) for value in self.context_options]
            )
            if self.context_tokens in self.context_options:
                context_select.value = self.context_tokens
            offload_select = self.query_one("#offload-select", Select)
            offload_select.set_options(
                [(self._fmt_offload(ratio), ratio) for ratio in self.offload_options]
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

    @staticmethod
    def _fmt_context(tokens: int) -> str:
        return f"{tokens // 1024}K" if tokens < 1_048_576 else f"{tokens // 1_048_576}M"

    @staticmethod
    def _fmt_offload(ratio: float) -> str:
        if ratio <= 0:
            return "CPU"
        return f"{int(ratio * 100)}%"

    def _change_context(self, delta: int) -> None:
        index = self.context_options.index(self.context_tokens)
        new_index = max(0, min(len(self.context_options) - 1, index + delta))
        self.context_tokens = self.context_options[new_index]
        self._refresh_estimates()

    def _change_offload(self, delta: int) -> None:
        index = self.offload_options.index(self.offload_ratio)
        new_index = max(0, min(len(self.offload_options) - 1, index + delta))
        self.offload_ratio = self.offload_options[new_index]
        self._refresh_estimates()

    def _refresh_estimates(self) -> None:
        self._render_estimate_controls()
        if self.mode == "files":
            self._render_file_table()

    def _render_repo_table(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        table.clear(columns=True)
        table.add_column("Repo / author", width=34, key="repo")
        table.add_column("Size", width=11, key="size")
        table.add_column("Downloads", width=12, key="downloads")
        for repo in self.repos:
            table.add_row(
                repo.id,
                fmt_size(repo.size),
                str(repo.downloads),
                key=repo.id,
            )

    def _render_file_table(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        table.clear(columns=True)
        table.add_column("Quant / file", width=24, key="file")
        table.add_column("File size", width=9, key="size")
        table.add_column("Est. VRAM", width=16, key="vram")
        table.add_column("Act. VRAM", width=9, key="act_vram")
        table.add_column("Est. t/s", width=9, key="speed")
        table.add_column("Act. t/s", width=8, key="act_speed")
        rows = build_quant_file_rows(
            self.files,
            gpu=self.gpu,
            context_tokens=self.context_tokens,
            offload_ratio=self.offload_ratio,
            baselines_path=self.baselines_path,
            models_dir=self.models_dir,
            author=self.selected_repo.author if self.selected_repo else "",
        )
        for row in rows:
            table.add_row(
                row.path,
                fmt_size(row.size),
                row.vram_cell,
                row.act_vram,
                row.est_tps,
                row.act_tps,
                key=row.path,
            )

    def _log(self, message: str) -> None:
        if self._downloading:
            return
        self._set_status(message)

    def _request_repos(self, author: str, search: str) -> None:
        if self._downloading:
            self.notify("Wait for the download to finish", severity="warning")
            return
        self._repo_load_id += 1
        self._set_status("[dim]Loading GGUF repos…[/]")
        self._load_repos(author, search, self._repo_load_id)

    @work(thread=True)
    def _load_repos(self, author: str, search: str, load_id: int) -> None:
        repos, error = list_gguf_repos(author=author, search=search)
        self.app.call_from_thread(self._finish_load_repos, load_id, repos, error)

    def _finish_load_repos(
        self, load_id: int, repos: list[HubRepo], error: str | None
    ) -> None:
        if load_id != self._repo_load_id or self._downloading:
            return
        if error:
            self._log(f"[bold $error]{error}[/]")
            return
        self.repos = repos
        self._set_mode_repos()
        self._log(f"Loaded {len(repos)} GGUF repos")

    @work(thread=True)
    def _load_files(self, repo: HubRepo, load_id: int) -> None:
        self.app.call_from_thread(self._log, f"[dim]Listing GGUF files in {repo.id}…[/]")
        files, error = list_repo_ggufs(repo.id)
        self.app.call_from_thread(self._finish_load_files, load_id, files, error)

    def _finish_load_files(
        self, load_id: int, files: list[HubFile], error: str | None
    ) -> None:
        if load_id != self._file_load_id or self._downloading:
            return
        if error:
            self._log(f"[bold $error]{error}[/]")
            return
        self.files = files
        self._render_file_table()
        self._log(f"Found {len(files)} GGUF files")
        self.query_one("#hub-table", DataTable).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.action_close()
        elif event.button.id == "search":
            self._request_repos(
                self.query_one("#author_filter", Input).value.strip(),
                self.query_one("#search_filter", Input).value.strip(),
            )
        elif event.button.id == "login":
            self._prompt_login()
        elif event.button.id == "back":
            self.action_back()
        elif event.button.id == "select":
            self._handle_select()

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._syncing_controls:
            return
        if event.select.id == "context-select":
            if event.value not in (None, Select.BLANK):
                self.context_tokens = int(event.value)
                if self.mode == "files":
                    self._render_file_table()
            return
        if event.select.id == "offload-select":
            if event.value not in (None, Select.BLANK):
                self.offload_ratio = float(event.value)
                if self.mode == "files":
                    self._render_file_table()
            return
        if event.select.id != "author_preset":
            return
        value = event.value
        if value not in (None, Select.BLANK):
            self.query_one("#author_filter", Input).value = str(value)
            self._request_repos(
                str(value),
                self.query_one("#search_filter", Input).value.strip(),
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"author_filter", "search_filter"}:
            self._request_repos(
                self.query_one("#author_filter", Input).value.strip(),
                self.query_one("#search_filter", Input).value.strip(),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on the focused list row advances to the next step."""
        if event.data_table.id == "hub-table":
            self._handle_select()

    def _prompt_login(self) -> None:
        def handle_result(result: tuple[bool, str] | None) -> None:
            if result is None:
                return
            ok, message = result
            if ok:
                self.notify(message)
                self._update_auth_status()
            else:
                self.notify(message, severity="error")

        self.app.push_screen(HFLoginDialog(), handle_result)

    def _handle_select(self) -> None:
        if self._downloading:
            self.notify("A download is already running", severity="warning")
            return
        table = self.query_one("#hub-table", DataTable)
        if table.cursor_row is None:
            self.notify("Select a row first", severity="warning")
            return
        row_key = table.get_row_at(table.cursor_row)[0]
        if self.mode == "repos":
            repo = next((r for r in self.repos if r.id == row_key), None)
            if repo is None:
                self.notify("Repo not found", severity="error")
                return
            self._set_mode_files(repo)
            return

        if not self.selected_repo:
            self.notify("No repo selected", severity="error")
            return
        filename = str(row_key)
        existing = find_model_by_repo(self.registry, self.selected_repo.id)
        if existing:
            cfg = self.registry.models[existing]
            self._start_download((cfg.display, existing), filename)
            return
        default_name = self._default_profile_name(self.selected_repo, filename)
        self.app.push_screen(
            DownloadProfileDialog(self.registry, default_name),
            lambda result: self._start_download(result, filename),
        )

    @staticmethod
    def _default_profile_name(repo: HubRepo, filename: str) -> str:
        return family_display(repo.id, filename)

    def _start_download(self, profile: tuple[str, str] | None, filename: str) -> None:
        if profile is None or not self.selected_repo:
            return
        if self._downloading:
            self.notify("A download is already running", severity="warning")
            return
        display_name, clone_from = profile
        all_names = [f.path for f in self.files]
        plan = build_download_plan(
            self.selected_repo.id,
            filename,
            self.models_dir,
            all_ggufs=all_names,
        )
        expected = sum(item.size for item in self.files if item.path in plan.filenames)

        from tui.app import LLMServeApp

        app = self.app
        if isinstance(app, LLMServeApp):

            def on_done() -> None:
                self._downloading = False
                self.registry = load_registry(self.models_json_path, models_dir=self.models_dir)
                self._set_status(f"[bold $success]Registered[/] {filename}")
                if self.on_complete:
                    self.on_complete()

            def on_fail(message: str) -> None:
                self._downloading = False
                self._set_status(f"[bold $error]Download failed[/] {message[:140]}")

            started = app.start_model_download(
                plan=plan,
                filename=filename,
                expected_bytes=expected,
                clone_from=clone_from,
                display=display_name,
                on_complete=on_done,
                on_error=on_fail,
            )
            if not started:
                return
            self._downloading = True
            self._set_status(
                f"[bold $warning]DOWNLOADING[/] {filename}  — runs in background (close Hub anytime)"
            )
            self.notify("Download started in background", timeout=6)
            return

        self._downloading = True
        self._set_status(
            f"[bold $warning]DOWNLOADING[/] {filename}  starting…  — leave Hub open"
        )
        self.notify(
            f"Downloading {filename} — stay on this Hub screen until it finishes",
            timeout=8,
        )
        self._download_worker(plan, display_name, clone_from, filename, expected)

    def _format_download_status(
        self,
        filename: str,
        plan: DownloadPlan,
        expected: int,
        cli_line: str,
        elapsed_s: int,
    ) -> str:
        used = local_download_bytes(plan)
        if expected > 0:
            pct = min(99.9, 100.0 * used / expected) if used else 0.0
            size = f"{fmt_size(used)} / {fmt_size(expected)} ({pct:.0f}%)"
        elif used:
            size = fmt_size(used)
        else:
            size = "starting…"
        extra = f"  {cli_line[:70]}" if cli_line else ""
        return (
            f"[bold $warning]DOWNLOADING[/] {filename}  {size}  {elapsed_s}s"
            f"{extra}  — leave Hub open"
        )

    @work(exclusive=True)
    async def _download_worker(
        self,
        plan: DownloadPlan,
        profile_name: str,
        clone_from: str,
        filename: str,
        expected: int,
    ) -> None:
        self._downloading = True
        cli_line = ""
        started = asyncio.get_running_loop().time()

        def on_line(line: str) -> None:
            nonlocal cli_line
            text = line.strip()
            if text:
                cli_line = text

        try:
            ok, message = False, "Download cancelled"
            download_task = asyncio.create_task(download_files(plan, on_line=on_line))
            try:
                while not download_task.done():
                    elapsed = int(asyncio.get_running_loop().time() - started)
                    self._set_status(
                        self._format_download_status(
                            filename, plan, expected, cli_line, elapsed
                        )
                    )
                    await asyncio.sleep(0.4)
                ok, message = download_task.result()
            finally:
                if not download_task.done():
                    download_task.cancel()
                    try:
                        await download_task
                    except (asyncio.CancelledError, Exception):
                        pass
            if not ok:
                self.notify(message[:200], severity="error")
                self._set_status(f"[bold $error]Download failed[/] {message[:140]}")
                return

            base_params = dict(self.registry.models[clone_from].params)
            source = build_source_metadata(plan, filename)
            gguf_path = plan.local_dir / filename
            create_downloaded_model(
                self.models_json_path,
                profile_name,
                base_params,
                file_rel=plan.relative_file,
                source=source,
                gguf_path=gguf_path,
                models_dir=self.models_dir,
            )
            self.registry = load_registry(self.models_json_path, models_dir=self.models_dir)
            if remember_hf_author(self.settings, plan.author):
                save_settings(self.settings_path, self.settings)
                self._refresh_author_preset_options()

            self.notify(f"Downloaded and registered {profile_name}")
            extra = ""
            info = read_gguf_architecture(gguf_path)
            if info.block_count is not None:
                extra = f"  layers={info.block_count} ({info.architecture or 'gguf'})"
            self._set_status(
                f"[bold $success]Registered {profile_name}[/]  file={plan.relative_file}{extra}"
            )
            if self.on_complete:
                self.on_complete()
        except Exception as exc:
            self.notify(str(exc)[:200], severity="error")
            self._set_status(f"[bold $error]Download failed[/] {exc}")
        finally:
            self._downloading = False

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "_download_worker":
            return
        busy = event.state in (WorkerState.PENDING, WorkerState.RUNNING)
        for button_id in ("search", "select", "back", "login", "close"):
            try:
                self.query_one(f"#{button_id}", Button).disabled = busy
            except Exception:
                pass

    def action_refresh(self) -> None:
        if self._downloading:
            self.notify("Wait for the download to finish", severity="warning")
            return
        if self.mode == "repos":
            self._request_repos(
                self.query_one("#author_filter", Input).value.strip(),
                self.query_one("#search_filter", Input).value.strip(),
            )
        elif self.selected_repo:
            self._file_load_id += 1
            self._load_files(self.selected_repo, self._file_load_id)

    def action_back(self) -> None:
        """Return from the quant list to the repo list."""
        if self._downloading:
            self.notify("Wait for the download to finish — leave Hub open", severity="warning")
            return
        if self.mode == "files":
            self._set_mode_repos()
        else:
            self.notify("Already at the main Hub page")

    def action_focus_filters(self) -> None:
        self.query_one("#author_filter", Input).focus()

    def action_context_prev(self) -> None:
        self._change_context(-1)

    def action_context_next(self) -> None:
        self._change_context(1)

    def action_offload_prev(self) -> None:
        self._change_offload(-1)

    def action_offload_next(self) -> None:
        self._change_offload(1)

    def action_close(self) -> None:
        from tui.app import LLMServeApp

        if self._downloading and not (
            isinstance(self.app, LLMServeApp) and self.app.download_manager.busy
        ):
            self.notify("Wait for the download to finish — leave Hub open", severity="warning")
            return
        self.dismiss(None)
