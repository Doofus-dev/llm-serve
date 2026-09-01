"""Hugging Face Hub browse and download screen."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Input, Label, Select, Static
from textual.worker import Worker, WorkerState
from rich.text import Text

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
    list_gguf_repos,
    list_repo_ggufs,
)
from tui.data.models_json import Registry, create_downloaded_model, load_registry
from tui.data.settings import TUISettings, remember_hf_author, save_settings
from tui.data.gpu import GPUStats, query_gpu
from tui.data.vram import classify_vram, estimate_vram_mb, fmt_memory_mb, status_symbol


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
            yield Label("Profile name:")
            yield Input(value=self.default_name, placeholder="my-model", id="name")
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
        if name in self.registry.models:
            self.app.notify(f"Model '{name}' already exists", severity="error")
            return
        if clone_from in (None, Select.BLANK):
            self.app.notify("Select a model to clone params from", severity="error")
            return
        self.dismiss((name, str(clone_from)))


class HubScreen(Screen):
    """Browse trending GGUF repos, pick a file, and download into models/<author>/."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("b", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "focus_filters", "Focus filters"),
        Binding("left", "context_prev", "Previous context"),
        Binding("right", "context_next", "Next context"),
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
        height: 1;
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
        height: 1;
        align: left middle;
    }

    #context-label {
        width: auto;
        height: 1;
        content-align: left middle;
        margin: 0 1 0 0;
    }

    #context-value {
        width: 8;
        height: 1;
        content-align: center middle;
    }

    #context-controls Button {
        width: 3;
        min-width: 3;
        height: 1;
        min-height: 1;
        padding: 0;
        margin: 0;
        border: none;
        background: transparent;
        color: $primary;
    }

    #hardware-summary {
        width: 1fr;
        height: 1;
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
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self.settings = settings
        self.settings_path = settings_path
        self.models_json_path = models_json_path
        self.models_dir = models_dir
        self.on_complete = on_complete
        self.auth_status = AuthStatus(logged_in=False)
        self.repos: list[HubRepo] = []
        self.files: list[HubFile] = []
        self.selected_repo: HubRepo | None = None
        self.mode = "repos"
        self._busy = False
        self.gpu = GPUStats()
        self.context_tokens = 65_536
        self.context_options: list[int] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="hub-panel"):
            yield Static("", id="hub-status")
            yield Label(
                "[bold]Browse Hugging Face GGUF models[/]  "
                "[dim]Tab: controls · ↑↓: list · Enter: open/download · "
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
                yield Button("◀", id="context-prev")
                yield Static("64K", id="context-value")
                yield Button("▶", id="context-next")
                yield Static("", id="hardware-summary")
            yield DataTable(id="hub-table", cursor_type="row")
            yield Static(
                "[dim]Enter open · F filters · "
                "[green]●[/] fit · [yellow]⚠[/] tight · [red]●[/] too large[/]",
                id="hub-help",
            )

    def on_mount(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        self._refresh_author_preset_options()
        self._update_auth_status()
        self.gpu = query_gpu()
        self._update_context_options()
        self.query_one("#context-controls").display = False
        self.query_one("#hub-table", DataTable).focus()
        self._load_repos("", "")

    def _refresh_author_preset_options(self) -> None:
        select = self.query_one("#author_preset", Select)
        options = [(author, author) for author in self.settings.hf_authors]
        select.set_options(options)

    def _update_auth_status(self) -> None:
        if not hf_available():
            self.auth_status = AuthStatus(logged_in=False)
            self.query_one("#hub-status", Static).update(
                f"[bold $error]hf CLI not installed[/] — {HF_INSTALL_HINT}"
            )
            return
        self.auth_status = auth_whoami()
        if self.auth_status.logged_in:
            self.query_one("#hub-status", Static).update(
                f"[bold $success]Logged in[/] as {self.auth_status.name}"
            )
        else:
            self.query_one("#hub-status", Static).update(
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
        self.mode = "files"
        self.selected_repo = repo
        self.query_one("#context-controls").display = True
        self.query_one("#back", Button).disabled = False
        self.query_one("#select", Button).label = "Download file"
        self._update_context_options(repo.context_length)
        self._load_files(repo)

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
        self._render_context_controls()
        if self.mode == "files":
            self._render_file_table()

    def _render_context_controls(self) -> None:
        index = self.context_options.index(self.context_tokens)
        self.query_one("#context-value", Static).update(self._fmt_context(self.context_tokens))
        self.query_one("#context-prev", Button).disabled = index == 0
        self.query_one("#context-next", Button).disabled = index == len(self.context_options) - 1
        if self.gpu.vram_total_mb:
            available = self.gpu.vram_total_mb - self.gpu.vram_used_mb
            summary = (
                f"{self.gpu.name}: {fmt_memory_mb(available)} available "
                f"(first GPU)"
            )
        else:
            summary = "GPU VRAM unavailable"
        self.query_one("#hardware-summary", Static).update(summary)

    @staticmethod
    def _fmt_context(tokens: int) -> str:
        return f"{tokens // 1024}K" if tokens < 1_048_576 else f"{tokens // 1_048_576}M"

    def _change_context(self, delta: int) -> None:
        index = self.context_options.index(self.context_tokens)
        new_index = max(0, min(len(self.context_options) - 1, index + delta))
        self.context_tokens = self.context_options[new_index]
        self._render_context_controls()
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
        table.add_column("Quant / file", width=34, key="file")
        table.add_column("File size", width=11, key="size")
        table.add_column("Est. VRAM", width=22, key="vram")
        estimates = [
            (item, classify_vram(estimate_vram_mb(item.size, self.context_tokens), self.gpu))
            for item in self.files
        ]
        estimates.sort(key=lambda pair: pair[1].total_mb, reverse=True)
        for item, estimate in estimates:
            if estimate.percent_available is None:
                fit = "?"
            else:
                fit = f"{estimate.percent_available:.0f}%"
            vram_cell = Text(f"{fmt_memory_mb(estimate.total_mb)} · {fit} ")
            vram_cell.append_text(status_symbol(estimate.status))
            table.add_row(
                item.path,
                fmt_size(item.size),
                vram_cell,
                key=item.path,
            )

    def _log(self, message: str) -> None:
        # Retained as a worker callback hook; the Hub no longer has a log panel.
        return

    @work(thread=True)
    def _load_repos(self, author: str, search: str) -> None:
        if self._busy:
            return
        self._busy = True
        self.app.call_from_thread(self._log, "[dim]Loading GGUF repos...[/]")
        repos, error = list_gguf_repos(author=author, search=search)
        self.app.call_from_thread(self._finish_load_repos, repos, error)

    def _finish_load_repos(self, repos: list[HubRepo], error: str | None) -> None:
        self._busy = False
        if error:
            self._log(f"[bold $error]{error}[/]")
            return
        self.repos = repos
        self._set_mode_repos()
        self._log(f"Loaded {len(repos)} GGUF repos")

    @work(thread=True)
    def _load_files(self, repo: HubRepo) -> None:
        self._busy = True
        self.app.call_from_thread(self._log, f"[dim]Listing files in {repo.id}...[/]")
        files, error = list_repo_ggufs(repo.id)
        self.app.call_from_thread(self._finish_load_files, files, error)

    def _finish_load_files(self, files: list[HubFile], error: str | None) -> None:
        self._busy = False
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
            self._load_repos(
                self.query_one("#author_filter", Input).value.strip(),
                self.query_one("#search_filter", Input).value.strip(),
            )
        elif event.button.id == "login":
            self._prompt_login()
        elif event.button.id == "back":
            self._set_mode_repos()
        elif event.button.id == "select":
            self._handle_select()
        elif event.button.id == "context-prev":
            self._change_context(-1)
        elif event.button.id == "context-next":
            self._change_context(1)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "author_preset":
            return
        value = event.value
        if value not in (None, Select.BLANK):
            self.query_one("#author_filter", Input).value = str(value)
            self._load_repos(
                str(value),
                self.query_one("#search_filter", Input).value.strip(),
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"author_filter", "search_filter"}:
            self._load_repos(
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
        default_name = self._default_profile_name(self.selected_repo, filename)
        self.app.push_screen(
            DownloadProfileDialog(self.registry, default_name),
            lambda result: self._start_download(result, filename),
        )

    @staticmethod
    def _default_profile_name(repo: HubRepo, filename: str) -> str:
        stem = Path(filename).stem.lower()
        author = repo.author.lower()
        return f"{author}-{stem}"[:48].strip("-")

    def _start_download(self, profile: tuple[str, str] | None, filename: str) -> None:
        if profile is None or not self.selected_repo:
            return
        profile_name, clone_from = profile
        all_names = [f.path for f in self.files]
        plan = build_download_plan(
            self.selected_repo.id,
            filename,
            self.models_dir,
            all_ggufs=all_names,
        )
        self._download_worker(plan, profile_name, clone_from, filename)

    @work(exclusive=True)
    async def _download_worker(
        self,
        plan: DownloadPlan,
        profile_name: str,
        clone_from: str,
        filename: str,
    ) -> None:
        files_label = ", ".join(plan.filenames)
        self._log(f"[bold]Downloading[/] {plan.repo_id} → {plan.local_dir}/")
        self._log(f"Files: {files_label}")

        def on_line(line: str) -> None:
            self.app.call_from_thread(self._log, line)

        ok, message = await download_files(plan, on_line=on_line)
        if not ok:
            self.notify(message[:200], severity="error")
            self._log(f"[bold $error]{message}[/]")
            return

        base_params = dict(self.registry.models[clone_from].params)
        source = build_source_metadata(plan, filename)
        create_downloaded_model(
            self.models_json_path,
            profile_name,
            base_params,
            file_rel=plan.relative_file,
            source=source,
        )
        self.registry = load_registry(self.models_json_path)
        if remember_hf_author(self.settings, plan.author):
            save_settings(self.settings_path, self.settings)
            self.app.call_from_thread(self._refresh_author_preset_options)

        self.app.call_from_thread(
            self.notify,
            f"Downloaded and registered {profile_name}",
        )
        self._log(f"[bold $success]Registered profile {profile_name}[/]  file={plan.relative_file}")
        if self.on_complete:
            self.app.call_from_thread(self.on_complete)

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
        if self.mode == "repos":
            self._load_repos(
                self.query_one("#author_filter", Input).value.strip(),
                self.query_one("#search_filter", Input).value.strip(),
            )
        elif self.selected_repo:
            self._load_files(self.selected_repo)

    def action_back(self) -> None:
        """Return from the quant list to the repo list."""
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

    def action_close(self) -> None:
        self.dismiss(None)
