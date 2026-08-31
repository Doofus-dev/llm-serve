"""Hugging Face Hub browse and download screen."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Input, Label, RichLog, Select, Static
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
    list_gguf_repos,
    list_repo_ggufs,
)
from tui.data.models_json import Registry, create_downloaded_model, load_registry
from tui.data.settings import TUISettings, remember_hf_author, save_settings


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
        Binding("r", "refresh", "Refresh"),
        Binding("f", "focus_filters", "Focus filters"),
    ]

    CSS = """
    HubScreen {
        align: center middle;
    }

    #hub-panel {
        width: 95%;
        height: 100%;
        border: thick $primary;
        background: $surface;
        padding: 0 1;
    }

    #hub-filters Horizontal {
        height: 3;
        align: left middle;
    }

    #hub-table {
        height: 1fr;
        min-height: 5;
        border: solid $accent;
    }

    #hub-log {
        height: 4;
        border: solid $accent;
    }

    #hub-filters Input {
        height: 3;
        min-width: 12;
        width: 1fr;
    }

    #hub-filters Select {
        height: 3;
        width: 20;
    }

    #hub-filters .field-label {
        width: auto;
        height: 3;
        content-align: left middle;
        margin: 0 1 0 0;
    }

    #hub-login-dialog, #hub-download-dialog {
        width: 70;
        height: auto;
        background: $surface;
        border: thick $primary;
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

    def compose(self) -> ComposeResult:
        with Vertical(id="hub-panel"):
            yield Static("", id="hub-status")
            yield Label(
                "[bold]Browse Hugging Face GGUF models[/]  "
                "[dim]Tab: controls · ↑↓: list · Enter: open/download[/]",
                id="hub-title",
            )
            with Vertical(id="hub-filters"):
                with Horizontal():
                    yield Select([], id="author_preset", allow_blank=True)
                    yield Input(placeholder="Author (e.g. bartowski)", id="author_filter")
                    yield Input(placeholder="Model name (e.g. Qwen3.6)", id="search_filter")
                with Horizontal():
                    yield Button("Search", variant="primary", id="search")
                    yield Button("Login", id="login")
                    yield Button("Back", id="back")
                    yield Button("Open / download", variant="success", id="select")
                    yield Button("Close", id="close")
            yield DataTable(id="hub-table", cursor_type="row")
            yield RichLog(id="hub-log", highlight=True, markup=True)
            yield Static(
                "[dim]Leave filters blank for trending. Focus the list and press Enter "
                "on a repo, then Enter on a GGUF file to download it.[/]",
                id="hub-help",
            )

    def on_mount(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        table.add_columns("Repo / File", "Downloads", "Size", "Extra")
        self._refresh_author_preset_options()
        self._update_auth_status()
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
        self.query_one("#back", Button).disabled = True
        self.query_one("#select", Button).label = "Open / download"
        self._render_repo_table()
        self.query_one("#hub-table", DataTable).focus()

    def _set_mode_files(self, repo: HubRepo) -> None:
        self.mode = "files"
        self.selected_repo = repo
        self.query_one("#back", Button).disabled = False
        self.query_one("#select", Button).label = "Download file"
        self._load_files(repo)

    def _render_repo_table(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        table.clear(columns=False)
        for repo in self.repos:
            extra = f"trend {repo.trending_score}" if repo.trending_score is not None else ""
            table.add_row(repo.id, str(repo.downloads), fmt_size(repo.size), extra, key=repo.id)

    def _render_file_table(self) -> None:
        table = self.query_one("#hub-table", DataTable)
        table.clear(columns=False)
        for item in self.files:
            table.add_row(item.path, "", fmt_size(item.size), "", key=item.path)

    def _log(self, message: str) -> None:
        self.query_one("#hub-log", RichLog).write(message)

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

    def action_focus_filters(self) -> None:
        self.query_one("#author_filter", Input).focus()

    def action_close(self) -> None:
        self.dismiss(None)
