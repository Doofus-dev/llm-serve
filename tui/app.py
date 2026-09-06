"""llm-serve TUI — config editor, hub, and live dashboard."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.events import Focus
from textual.widgets import Footer, Header, Label

from tui.bindings import HELP_TEXT, SELECTION_ACTIONS, build_app_bindings, selection_supports_action
from tui.data.baselines import RunBaseline, record_baseline
from tui.data.downloads import DownloadJob, DownloadManager
from tui.data.gpu import query_gpu
from tui.data.hf import build_download_plan, build_source_metadata, fmt_size
from tui.data.models_json import (
    AliasTarget,
    add_or_update_quant,
    clamp_preset_contexts,
    delete_model,
    load_registry,
    save_registry,
    set_active_quant,
    sync_gguf_architecture,
    unshared_model_file_paths,
    update_model,
)
from tui.data.pidfile import read_pid_file
from tui.data.presets import (
    MAX_PRESETS_PER_MODEL,
    clear_active_preset,
    delete_all_presets_for_model,
    delete_preset,
    get_active_slot,
    get_preset,
    load_presets,
    merge_identity_and_preset,
    next_free_slot,
    save_presets,
    set_active_preset,
    set_preset,
)
from tui.data.quant import quant_from_filename
from tui.data.settings import (
    cycle_log_verbosity,
    load_settings,
    log_verbosity_label,
    save_settings,
)
from tui.data.stats import ServerClient
from tui.data.throughput_history import (
    ThroughputHistory,
    ThroughputReader,
    baseline_speed,
    sample_tps_for_history,
)
from tui.launch import LaunchError, launch_background, prepare_launch, stop_server
from tui.paths import METRICS_HISTORY_SAMPLES, METRICS_POLL_INTERVAL, AppPaths, default_paths
from tui.screens.editors import (
    ConfirmDialog,
    CreateAliasDialog,
    EditAliasDialog,
    ParamFocused,
    ParamHelpPanel,
    ParamInput,
    ParamSelect,
    PresetEditor,
    ProfileEditor,
)
from tui.screens.hub import HubScreen
from tui.screens.quant_picker import QuantPickerScreen
from tui.widgets.config import ConfigPanel
from tui.widgets.download_bar import DownloadBar
from tui.widgets.log_panel import LogPanel
from tui.widgets.nav import AliasNav, ModelNav
from tui.widgets.status import StatusPanel


class LLMServeApp(App):
    TITLE = "llm-serve"
    CSS_PATH = Path(__file__).with_name("app.tcss")
    BINDINGS = build_app_bindings()

    def __init__(self, paths: AppPaths | None = None):
        super().__init__()
        self.paths = paths or default_paths()
        self.registry = load_registry(self.paths.models_json, models_dir=self.paths.models_dir)
        self.preset_store = load_presets(self.paths.presets_json)
        self.settings = load_settings(self.paths.settings_json)
        self.download_manager = DownloadManager()
        self.client: ServerClient | None = None
        self.remote_launch: bool = self.settings.remote_launch
        self.log_verbosity: int = self.settings.log_verbosity
        self._launch_time: float | None = None
        self._editor_mode: bool = False
        self._editor_widget: ProfileEditor | PresetEditor | None = None
        self._help_panel: ParamHelpPanel | None = None
        self._help_visible: bool = False
        self._focused_param: str | None = None
        self._gen_history = ThroughputHistory(max_samples=METRICS_HISTORY_SAMPLES)
        self._throughput_reader = ThroughputReader()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-body"):
            yield DownloadBar(id="download-bar")
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    yield Label("MODELS", classes="nav-heading")
                    yield ModelNav(self.registry, self.preset_store, self.paths.models_dir)
                    yield Label("ALIASES  Tab · 1-5 pin", classes="nav-heading")
                    yield AliasNav(self.registry)
                with Vertical(id="right"):
                    yield StatusPanel(id="status")
                    yield ConfigPanel(id="config")
            yield LogPanel(id="logs")
        yield Footer()

    def _selection_kind(self) -> str | None:
        data = self._selected_data()
        return data[0] if data else None

    def _models_section_focused(self) -> bool:
        return isinstance(self.focused, ModelNav)

    def _log_panel_focused(self) -> bool:
        return isinstance(self.focused, LogPanel)

    def _action_available(self, action: str) -> bool:
        if not self.is_mounted:
            return False
        try:
            return selection_supports_action(
                action,
                self._selection_kind(),
                models_section=self._models_section_focused(),
                log_section=self._log_panel_focused(),
            )
        except (QueryError, NoMatches):
            return False

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in SELECTION_ACTIONS:
            return False if self._editor_mode else self._action_available(action)
        return True

    def on_focus(self, event: Focus) -> None:
        if isinstance(event.widget, (ModelNav, AliasNav, LogPanel)):
            self.refresh_bindings()

    def _update_footer(self) -> None:
        """Refresh footer bindings for editor mode, labels, and selection."""
        if self._editor_mode:
            self._bindings = BindingsMap(
                [
                    Binding("ctrl+s", "save_edit", "Save"),
                    Binding("escape", "cancel_edit", "Cancel"),
                ]
            )
        else:
            info_label = "Info"
            if self.is_mounted:
                try:
                    info_label = self._info_log_label()
                except (QueryError, NoMatches):
                    pass
            self._bindings = BindingsMap(
                build_app_bindings(
                    remote_on=self.remote_launch,
                    log_label=log_verbosity_label(self.log_verbosity),
                    info_label=info_label,
                )
            )
        if self.is_mounted:
            self.refresh_bindings()

    def on_mount(self) -> None:
        saved_theme = self.settings.theme
        if saved_theme and saved_theme in self.available_themes:
            self.theme = saved_theme
        nav = self.query_one(ModelNav)
        nav.focus()
        cfg = self.query_one(ConfigPanel)
        cfg.models_dir = self.paths.models_dir
        cfg.registry = self.registry
        cfg.preset_store = self.preset_store
        first = next(iter(self.registry.models), None)
        if first:
            cfg.selected = first
        self._refresh_pid()
        self.set_interval(METRICS_POLL_INTERVAL, self._poll_metrics)
        self.set_interval(5.0, self._poll_gpu)
        self.set_interval(3.0, self._poll_log)
        self.query_one(LogPanel).poll_file(self.paths.log_file)
        self.query_one(StatusPanel).next_remote = self.remote_launch
        self.query_one(StatusPanel).next_log_verbosity = self.log_verbosity
        self._update_footer()
        self.download_manager.subscribe(self._on_download_state)
        self._reload_registry(notify_gguf=True)

    def _on_download_state(self, state) -> None:
        """Apply download progress on the UI thread."""

        def apply() -> None:
            try:
                self.query_one("#download-bar", DownloadBar).apply_state(state)
            except (QueryError, NoMatches):
                return

        self.call_later(apply)

    @work(exclusive=True)
    async def _process_download_queue(self) -> None:
        while True:
            job = self.download_manager.pop_next()
            if job is None:
                return
            ok, message = await self.download_manager.run(job)
            if ok:
                if job.on_success:
                    job.on_success()
            elif job.on_error:
                job.on_error(message)

    def _model_active_quant(self, model_name: str) -> str:
        cfg = self.registry.models.get(model_name)
        if not cfg:
            return "LOCAL"
        if cfg.active_quant:
            return cfg.active_quant
        return quant_from_filename(cfg.file)

    def _switch_quant(self, model_name: str, quant_id: str) -> None:
        if set_active_quant(self.paths.models_json, model_name, quant_id, models_dir=self.paths.models_dir):
            self._reload_registry()
            self.notify(f"Switched to quant {quant_id}")
        else:
            self.notify(f"Quant {quant_id} not available", severity="error")

    def start_model_download(
        self,
        *,
        plan,
        filename: str,
        expected_bytes: int,
        clone_from: str,
        display: str | None = None,
        hf_context: int | None = None,
        on_complete=None,
        on_error=None,
    ) -> bool:
        repo = plan.repo_id

        def _after_download() -> None:
            from tui.data.gguf import read_gguf_architecture

            base_params = dict(self.registry.models[clone_from].params)
            source = build_source_metadata(plan, filename)
            gguf_path = plan.local_dir / filename
            slug, quant_id = add_or_update_quant(
                self.paths.models_json,
                repo_id=repo,
                filename=filename,
                file_rel=plan.relative_file,
                source=source,
                base_params=base_params,
                gguf_path=gguf_path,
                models_dir=self.paths.models_dir,
                display=display,
                hf_context=hf_context,
            )
            self._reload_registry()
            extra = ""
            info = read_gguf_architecture(gguf_path)
            if info.block_count is not None:
                extra = f" layers={info.block_count}"
            self.notify(f"Registered {slug} quant {quant_id}{extra}")
            if on_complete:
                on_complete()

        def _on_error(message: str) -> None:
            self.notify(message[:200], severity="error")
            if on_error:
                on_error(message)

        job = DownloadJob(
            plan=plan,
            filename=filename,
            expected_bytes=expected_bytes,
            clone_from=clone_from,
            display=display,
            on_success=_after_download,
            on_error=_on_error,
        )
        result = self.download_manager.enqueue(job)
        if result == "duplicate":
            self.notify(f"{filename} is already downloading or queued", severity="warning")
            return False
        if result == "queued":
            waiting = self.download_manager.queue_size
            self.notify(f"Queued {filename} ({waiting} waiting)", timeout=4)
        else:
            self.notify(f"Downloading {filename}…", timeout=4)
        self._process_download_queue()
        return True

    def action_pick_quant(self) -> None:
        model_name = self._selected_model()
        if not model_name:
            self.notify("Select a model first", severity="warning")
            return
        if model_name not in self.registry.models:
            return
        cfg = self.registry.models[model_name]

        def on_download(model: str, filename: str, expected_bytes: int = 0) -> None:
            source = cfg.params.get("source")
            if not isinstance(source, dict) or not source.get("repo"):
                self.notify("No Hub repo for this model", severity="error")
                return
            clone_from = model_name if model_name in self.registry.models else next(iter(self.registry.models))
            plan = build_download_plan(source["repo"], filename, self.paths.models_dir)
            self.start_model_download(
                plan=plan,
                filename=filename,
                expected_bytes=expected_bytes,
                clone_from=clone_from,
                display=cfg.display,
            )

        def handle(result: str | None) -> None:
            if result:
                self._switch_quant(model_name, result)

        preferred_ctx = None
        params = self._effective_model_params(model_name) or {}
        try:
            ctx = int(float(str(params.get("ctx") or 0)))
        except (TypeError, ValueError):
            ctx = 0
        if ctx > 0:
            preferred_ctx = ctx

        self.push_screen(
            QuantPickerScreen(
                model_name,
                cfg,
                self.registry,
                self.paths.models_dir,
                self.paths.models_json,
                baselines_path=self.paths.baselines_json,
                preferred_ctx=preferred_ctx,
                on_download=on_download,
            ),
            handle,
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if not isinstance(event.option_list, (ModelNav, AliasNav)):
            return
        data = event.option_list.selected_data
        if data and data[0] == "preset":
            _, model_name, quant, slot = data
            set_active_preset(self.preset_store, model_name, quant, slot)
            save_presets(self.paths.presets_json, self.preset_store)
            self._reload_registry()

    def watch_theme(self, theme_name: str) -> None:
        """Persist theme choice and refresh panels that use theme colors."""
        if theme_name and theme_name in self.available_themes:
            self.settings.theme = theme_name
            save_settings(self.paths.settings_json, self.settings)
        self.call_after_refresh(self._refresh_themed_widgets)

    def _refresh_themed_widgets(self) -> None:
        """Re-render custom panels so theme markup/CSS picks up the new palette."""
        if self._editor_mode:
            return
        for selector in (StatusPanel, ConfigPanel):
            try:
                self.query_one(selector).refresh()
            except (QueryError, NoMatches):
                continue

    def _refresh_pid(self) -> None:
        info = read_pid_file(self.paths.pid_file)
        if self._editor_mode:
            return
        try:
            panel = self.query_one(StatusPanel)
        except (QueryError, NoMatches):
            return
        was_alive = panel.pid_info.alive if panel.pid_info else False
        alive = info.alive if info else False
        prev = panel.pid_info
        switched = bool(
            alive
            and info
            and prev
            and (info.pid != prev.pid or info.model != prev.model)
        )
        if not alive or switched:
            self._gen_history.clear()
            self._throughput_reader.clear()
            panel.gen_tps_history = []
            panel.live_throughput = None
            panel.metrics = None
            panel.props = None
        panel.pid_info = info if alive else None
        running_key = None
        if alive and info:
            running_alias = self.registry.aliases.get(info.model)
            running_key = running_alias.model if running_alias else info.model
            if running_key not in self.registry.models:
                running_key = next(
                    (
                        key
                        for key, model in self.registry.models.items()
                        if model.display == info.model
                    ),
                    None,
                )
        panel.model_display = (
            self.registry.models[running_key].display
            if running_key in self.registry.models
            else None
        )
        panel.preset_display = None
        if running_key in self.registry.models:
            quant = info.quant or self._model_active_quant(running_key)
            slot = info.preset_slot
            if slot is None:
                slot = get_active_slot(self.preset_store, running_key, quant)
            if slot is not None:
                running_preset = get_preset(self.preset_store, running_key, quant, slot)
                if running_preset:
                    panel.preset_display = running_preset.name
        if alive and not was_alive:
            self._launch_time = time.time()
        if alive and info:
            if self.client is None or self.client.base != f"http://127.0.0.1:{info.port}":
                if self.client:
                    asyncio.ensure_future(self.client.close())
                self.client = ServerClient("127.0.0.1", info.port)
        panel.uptime = (time.time() - self._launch_time) if (alive and self._launch_time) else 0.0

    async def _poll_metrics(self) -> None:
        self._refresh_pid()
        try:
            panel = self.query_one(StatusPanel)
        except (QueryError, NoMatches):
            return
        if self.client and panel.pid_info:
            metrics, slots = await asyncio.gather(
                self.client.metrics(),
                self.client.slots(),
            )
            panel.metrics = metrics
            if metrics:
                live = self._throughput_reader.update(metrics, slots)
                panel.live_throughput = live
                self._gen_history.push(sample_tps_for_history(live))
                panel.gen_tps_history = self._gen_history.samples
            if panel.props is None:
                panel.props = await self.client.props()
            self._record_baseline(panel)

    def _poll_gpu(self) -> None:
        try:
            panel = self.query_one(StatusPanel)
        except (QueryError, NoMatches):
            return
        panel.gpu = query_gpu()
        self._record_baseline(panel)

    def _poll_log(self) -> None:
        try:
            self.query_one(LogPanel).poll_file(self.paths.log_file)
        except (QueryError, NoMatches):
            return

    def _effective_model_params(self, model_name: str) -> dict | None:
        if model_name not in self.registry.models:
            return None
        identity = dict(self.registry.models[model_name].params)
        quant = self._model_active_quant(model_name)
        slot = get_active_slot(self.preset_store, model_name, quant)
        if slot is not None:
            preset = get_preset(self.preset_store, model_name, quant, slot)
            if preset:
                return merge_identity_and_preset(identity, preset)
        return identity

    def _record_baseline(self, panel: StatusPanel) -> None:
        """Persist observed VRAM / tok/s for the running model on this GPU."""
        info = panel.pid_info
        gpu = panel.gpu
        if not info or not info.alive or not gpu or not gpu.available:
            return
        if gpu.vram_used_mb <= 0:
            return
        params = self._effective_model_params(info.model)
        if not params:
            return

        file_rel = str(params.get("file", ""))
        path = self.paths.models_dir / file_rel
        try:
            file_size = path.stat().st_size if path.is_file() else 0
        except OSError:
            file_size = 0

        gen_tps, prompt_tps, tokens = baseline_speed(
            panel.live_throughput,
            panel.gen_tps_history,
        )
        metrics = panel.metrics
        if prompt_tps is None and metrics and metrics.avg_prompt_tps > 0:
            prompt_tps = metrics.avg_prompt_tps

        def as_int(value: object, default: int = 0) -> int:
            try:
                return int(float(str(value)))
            except (TypeError, ValueError):
                return default

        ctx = as_int(params.get("ctx"), 0)
        if ctx <= 0:
            return
        try:
            record_baseline(
                self.paths.baselines_json,
                RunBaseline(
                    model=info.model,
                    file=file_rel,
                    file_size=file_size,
                    gpu_name=gpu.name,
                    ctx=ctx,
                    gpu_layers=as_int(params.get("gpu_layers"), 99),
                    total_layers=as_int(params.get("total_layers"), 0),
                    cache_k=str(params.get("cache_k") or "q4_0"),
                    cache_v=str(params.get("cache_v") or "q4_0"),
                    vram_used_mb=gpu.vram_used_mb,
                    gen_tps=gen_tps,
                    prompt_tps=prompt_tps,
                    tokens_predicted=tokens,
                ),
            )
        except OSError:
            return

    def _reload_registry(self, *, notify_gguf: bool = False) -> None:
        """Reload models.json and presets.json, refresh UI."""
        changes = sync_gguf_architecture(self.paths.models_json, self.paths.models_dir)
        clamped = clamp_preset_contexts(self.paths.models_json, self.paths.presets_json, models_dir=self.paths.models_dir)
        self.registry = load_registry(self.paths.models_json, models_dir=self.paths.models_dir)
        self.preset_store = load_presets(self.paths.presets_json)
        nav = self.query_one(ModelNav)
        nav.registry = self.registry
        nav.preset_store = self.preset_store
        nav.refresh_cards()
        aliases = self.query_one(AliasNav)
        aliases.registry = self.registry
        aliases.refresh_aliases()
        cfg = self.query_one(ConfigPanel)
        cfg.registry = self.registry
        cfg.preset_store = self.preset_store
        cfg.models_dir = self.paths.models_dir
        if notify_gguf and changes:
            bits = ", ".join(
                f"{name} {field} {old}→{new}" for name, field, old, new in changes
            )
            self.notify(f"Updated from GGUF: {bits}")
        if clamped:
            bits = ", ".join(
                f"{model}/{quant}[{slot}] {old}→{new}"
                for model, quant, slot, old, new in clamped[:3]
            )
            extra = f" (+{len(clamped) - 3} more)" if len(clamped) > 3 else ""
            self.notify(f"Capped preset context: {bits}{extra}", severity="warning")
        self._update_footer()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if not isinstance(event.option_list, (ModelNav, AliasNav)):
            return
        cfg = self.query_one(ConfigPanel)
        data = event.option_list.selected_data
        if data and data[0] in ("model", "preset"):
            cfg.selected = data[1]
        elif data and data[0] == "alias":
            target = self.registry.aliases.get(data[1])
            cfg.selected = target.model if target else None
        cfg.registry = self.registry
        cfg.preset_store = self.preset_store
        self._update_footer()

    def _selected_model(self) -> str | None:
        data = self._selected_data()
        if not data:
            return None
        if data[0] == "alias":
            target = self.registry.aliases.get(data[1])
            return target.model if target else None
        if data[0] in ("model", "preset"):
            return data[1]
        return None

    def _selected_data(self) -> tuple | None:
        if isinstance(self.focused, AliasNav):
            return self.query_one(AliasNav).selected_data
        return self.query_one(ModelNav).selected_data

    def action_launch(self) -> None:
        selected = self._selected_data()
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return

        if model not in self.registry.models:
            self.notify(f"Model '{model}' not found", severity="error")
            return

        cfg = self.registry.models[model]
        alias_name = selected[1] if selected and selected[0] == "alias" else None
        alias_target = self.registry.aliases.get(alias_name) if alias_name else None
        quant = (
            alias_target.quant
            if alias_target and alias_target.quant is not None
            else self._model_active_quant(model)
        )
        slot = (
            alias_target.preset_slot
            if alias_target and alias_target.preset_slot is not None
            else get_active_slot(self.preset_store, model, quant)
        )
        if slot is None:
            self.notify(
                f"No active preset for {cfg.display} ({quant}). Create and apply one first.",
                severity="error",
            )
            return

        preset = get_preset(self.preset_store, model, quant, slot)
        if not preset:
            self.notify(
                f"Active preset slot {slot} is missing for {cfg.display} ({quant})",
                severity="error",
            )
            return

        launch_name = alias_name or cfg.display
        access = "remote" if self.remote_launch else "local"
        self.notify(
            f"Launching {launch_name} ({quant}) preset [{slot}] {preset.name} — {access}"
        )
        self._launch_worker(launch_name)

    @work(exclusive=True)
    async def _launch_worker(self, launch_name: str) -> None:
        def _run():
            plan = prepare_launch(
                launch_name,
                remote=self.remote_launch,
                log_verbosity=self.log_verbosity,
                paths=self.paths,
                registry=self.registry,
            )
            launch_background(plan, paths=self.paths)
            return plan

        try:
            await asyncio.to_thread(_run)
        except LaunchError as exc:
            self.notify(f"Launch failed: {str(exc)[:200]}", severity="error")
            return
        self.notify(f"Launched {launch_name}")
        self._reload_registry()
        self._refresh_pid()

    def _info_log_label(self) -> str:
        try:
            showing = self.query_one(LogPanel)._show_info
        except (QueryError, NoMatches):
            showing = False
        return "Info ON" if showing else "Info"

    def action_toggle_log_source(self) -> None:
        if self._editor_mode:
            return
        self.query_one(LogPanel).toggle_info()
        self._update_footer()

    def action_toggle_remote(self) -> None:
        """Toggle Meshnet/LAN binding for the next launch."""
        self.remote_launch = not self.remote_launch
        self.settings.remote_launch = self.remote_launch
        save_settings(self.paths.settings_json, self.settings)
        self.query_one(StatusPanel).next_remote = self.remote_launch
        self._update_footer()

    def action_cycle_log_verbosity(self) -> None:
        """Cycle llama.cpp log verbosity for the next launch (info/trace/debug)."""
        if self._editor_mode:
            return
        self.log_verbosity = cycle_log_verbosity(self.log_verbosity)
        self.settings.log_verbosity = self.log_verbosity
        save_settings(self.paths.settings_json, self.settings)
        self.query_one(StatusPanel).next_log_verbosity = self.log_verbosity
        self._update_footer()

    def action_stop(self) -> None:
        self._stop_worker()

    @work(exclusive=True)
    async def _stop_worker(self) -> None:
        try:
            message = await asyncio.to_thread(lambda: stop_server(paths=self.paths))
        except LaunchError as exc:
            self.notify(f"Stop error: {exc}", severity="error")
            return
        self.notify(message)
        self._refresh_pid()

    def on_param_focused(self, event: ParamFocused) -> None:
        self._focused_param = event.param
        if self._help_panel and self._help_visible:
            self._help_panel.show_param(event.param)

    def _enter_param_editor(self, editor: ProfileEditor | PresetEditor) -> None:
        """Show model/preset editor with optional F2 help panel."""
        self.query_one("#status").display = False
        self.query_one("#config").display = False
        self.query_one("#logs").display = False

        editor.id = "editor-scroll"
        help_panel = ParamHelpPanel(id="param-help")

        right = self.query_one("#right")
        right.mount(editor)
        right.mount(help_panel)

        self._editor_widget = editor
        self._help_panel = help_panel
        self._help_visible = False
        self._focused_param = None
        self._editor_mode = True
        self._update_footer()

    def _current_focused_param(self) -> str | None:
        """Read the param name from whichever field is focused in the editor."""
        focused = self.focused
        if isinstance(focused, (ParamInput, ParamSelect)):
            return focused.param_name
        return self._focused_param

    def action_toggle_param_help(self) -> None:
        if not self._editor_mode or not self._help_panel:
            return
        self._help_visible = not self._help_visible
        right = self.query_one("#right")
        if self._help_visible:
            right.add_class("help-open")
            param = self._current_focused_param()
            if param:
                self._focused_param = param
            self._help_panel.show_param(self._focused_param)
        else:
            right.remove_class("help-open")
        self._update_footer()

    def action_edit(self) -> None:
        if self._editor_mode:
            self.notify("Already in edit mode", severity="warning")
            return
        
        data = self._selected_data()
        if data is None:
            self.notify("Select a model, preset, or alias first", severity="warning")
            return

        # Check if it's a preset
        if data[0] == "preset":
            _, model_name, quant, slot = data
            self._edit_preset(model_name, quant, slot)
            return
        
        # Check if it's an alias
        if data[0] == "alias":
            alias_name = data[1]

            def handle_rename(new_name: str | None) -> None:
                if not new_name or new_name == alias_name:
                    return
                target = self.registry.aliases.pop(alias_name)
                self.registry.aliases[new_name] = target
                save_registry(self.paths.models_json, self.registry)
                self._reload_registry()
                nav = self.query_one(AliasNav)
                nav.focus()
                for index in range(nav.option_count):
                    option = nav.get_option_at_index(index)
                    if nav._option_data.get(option.id or "") == ("alias", new_name):
                        nav.highlighted = index
                        break
                self.notify(f"Renamed alias {alias_name} → {new_name}")

            self.push_screen(EditAliasDialog(self.registry, alias_name), handle_rename)
            return
        
        # Otherwise it's a model
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return
        if model not in self.registry.models:
            self.notify(f"Model '{model}' not found", severity="error")
            return

        self._reload_registry()
        cfg = self.registry.models[model]
        
        def on_save(new_params: dict) -> None:
            update_model(self.paths.models_json, model, new_params)
            self._reload_registry()
            self._exit_editor()
            self.notify(f"Saved profile {cfg.display}")

        def on_cancel() -> None:
            self._exit_editor()
            self.notify("Edit cancelled")

        editor = ProfileEditor(model, cfg.params, self.registry, on_save, on_cancel)
        self._enter_param_editor(editor)

    def _edit_preset(self, model_name: str, quant: str, slot: int) -> None:
        """Edit a preset."""
        self._reload_registry()
        identity = self.registry.models[model_name].params
        preset = get_preset(self.preset_store, model_name, quant, slot)
        is_new = preset is None

        def on_save(name: str, params: dict) -> None:
            set_preset(self.preset_store, model_name, quant, slot, name, params)
            if is_new or get_active_slot(self.preset_store, model_name, quant) is None:
                set_active_preset(self.preset_store, model_name, quant, slot)
            save_presets(self.paths.presets_json, self.preset_store)
            self._reload_registry()
            self._exit_editor()
            self.notify(f"Saved preset {model_name}/{quant} [{slot}] {name}")

        def on_cancel() -> None:
            self._exit_editor()
            self.notify("Edit cancelled")

        editor = PresetEditor(
            model_name,
            slot,
            preset,
            identity,
            is_new=is_new,
            models_dir=self.paths.models_dir,
            on_save=on_save,
            on_cancel=on_cancel,
        )
        self._enter_param_editor(editor)

    def cycle_selected_alias(self, direction: int) -> None:
        """Immediately point the selected alias at the adjacent model."""
        nav = self.query_one(AliasNav)
        data = nav.selected_data
        if not data or data[0] != "alias":
            return
        alias_name = data[1]
        model_names = list(self.registry.models)
        if not model_names:
            self.notify("No models are available", severity="warning")
            return
        current_target = self.registry.aliases.get(alias_name)
        current = current_target.model if current_target else None
        try:
            index = model_names.index(current)
        except ValueError:
            index = 0
        target = model_names[(index + direction) % len(model_names)]
        if target == current:
            return
        self.registry.aliases[alias_name] = AliasTarget(model=target)
        save_registry(self.paths.models_json, self.registry)
        self._reload_registry()

    def _exit_editor(self) -> None:
        """Exit editor mode and restore normal view."""
        if self._help_panel:
            self._help_panel.remove()
            self._help_panel = None
        if self._editor_widget:
            self._editor_widget.remove()
            self._editor_widget = None
        right = self.query_one("#right")
        right.remove_class("help-open")
        self._help_visible = False
        self._focused_param = None
        self.query_one("#status").display = True
        self.query_one("#config").display = True
        self.query_one("#logs").display = True
        self._editor_mode = False
        self._update_footer()
        self.query_one(ModelNav).focus()

    def action_save_edit(self) -> None:
        """Save current editor (called by Ctrl+S binding)."""
        if self._editor_widget:
            self._editor_widget.save()

    def action_cancel_edit(self) -> None:
        """Cancel current editor (called by Esc binding)."""
        if self._editor_widget:
            self._exit_editor()
            self.notify("Edit cancelled")

    def action_new(self) -> None:
        if self._editor_mode:
            self.notify("Close the editor first", severity="warning")
            return
        data = self._selected_data()
        if data and data[0] == "alias":
            def handle_alias_result(result: tuple[str, str] | None) -> None:
                if result is not None:
                    name, target = result
                    self.registry.aliases[name] = AliasTarget(model=target)
                    save_registry(self.paths.models_json, self.registry)
                    self._reload_registry()
                    self.notify(f"Created alias {name} → {target}")

            self.push_screen(CreateAliasDialog(self.registry), handle_alias_result)
            return

        model = self._selected_model()
        if not model:
            self.notify("Select a model first to create a preset", severity="warning")
            return
        self._new_preset_for_model(model)

    def _new_preset_for_model(self, model_name: str) -> None:
        """Open the preset editor on the next free slot for this model's active quant."""
        if model_name not in self.registry.models:
            self.notify(f"Model '{model_name}' not found", severity="error")
            return
        quant = self._model_active_quant(model_name)
        slot = next_free_slot(self.preset_store, model_name, quant)
        if slot is None:
            self.notify(
                f"{model_name}/{quant} already has {MAX_PRESETS_PER_MODEL} presets",
                severity="warning",
            )
            return
        self._edit_preset(model_name, quant, slot)

    def action_delete(self) -> None:
        data = self._selected_data()
        if data is None:
            self.notify("Select a model, preset, or alias first", severity="warning")
            return
        
        # Check if it's a preset
        if data[0] == "preset":
            _, model_name, quant, slot = data
            preset = get_preset(self.preset_store, model_name, quant, slot)
            if not preset:
                self.notify("Preset not found", severity="error")
                return

            pid_info = read_pid_file(self.paths.pid_file)
            if pid_info and pid_info.alive and pid_info.model == model_name:
                if get_active_slot(self.preset_store, model_name, quant) == slot:
                    self.notify("Cannot delete preset while server is running with it", severity="error")
                    return

            msg = f"Delete preset '{preset.name}' (slot {slot}) for {model_name}/{quant}?"

            def handle_preset_confirm(confirmed: bool) -> None:
                if confirmed:
                    delete_preset(self.preset_store, model_name, quant, slot)
                    if get_active_slot(self.preset_store, model_name, quant) == slot:
                        clear_active_preset(self.preset_store, model_name, quant)
                    save_presets(self.paths.presets_json, self.preset_store)
                    self._reload_registry()
                    self.notify(f"Deleted preset {model_name}/{quant} [{slot}]")

            self.push_screen(ConfirmDialog(msg), handle_preset_confirm)
            return
        
        # Check if it's an alias
        if data[0] == "alias":
            alias_name = data[1]
            msg = f"Delete alias '{alias_name}'?"
            
            def handle_alias_confirm(confirmed: bool) -> None:
                if confirmed:
                    del self.registry.aliases[alias_name]
                    save_registry(self.paths.models_json, self.registry)
                    self._reload_registry()
                    self.notify(f"Deleted alias {alias_name}")
            
            self.push_screen(ConfirmDialog(msg), handle_alias_confirm)
            return
        
        # Otherwise it's a model
        model = self._selected_model()
        if not model:
            self.notify("Select a model first", severity="warning")
            return
        pid_info = read_pid_file(self.paths.pid_file)
        if pid_info and pid_info.alive and pid_info.model == model:
            self.notify("Stop this model before deleting it", severity="error")
            return
        
        # Check for aliases pointing to this model
        aliases_using = [
            alias
            for alias, target in self.registry.aliases.items()
            if target.model == model
        ]
        files_to_delete = unshared_model_file_paths(self.registry, model, self.paths.models_dir)
        bytes_to_free = sum(path.stat().st_size for path in files_to_delete)
        display_name = self.registry.models[model].display
        msg = f"Delete model '{display_name}' and all of its presets?"
        if files_to_delete:
            msg += (
                f"\n\nThis also permanently deletes {len(files_to_delete)} GGUF "
                f"file{'s' if len(files_to_delete) != 1 else ''} "
                f"({fmt_size(bytes_to_free)}) from disk."
            )
        else:
            msg += "\n\nNo unshared GGUF files are stored on disk."
        if aliases_using:
            msg += f"\n\nAliases also removed: {', '.join(aliases_using)}"
        
        def handle_model_confirm(confirmed: bool) -> None:
            if confirmed:
                try:
                    for path in files_to_delete:
                        path.unlink()
                except OSError as exc:
                    self.notify(f"Could not delete model file: {exc}", severity="error")
                    return
                delete_model(self.paths.models_json, model)
                delete_all_presets_for_model(self.preset_store, model)
                save_presets(self.paths.presets_json, self.preset_store)
                self._reload_registry()
                freed = f" and freed {fmt_size(bytes_to_free)}" if files_to_delete else ""
                self.notify(f"Deleted {display_name}{freed}")
        
        self.push_screen(ConfirmDialog(msg), handle_model_confirm)

    def action_activate_preset(self, slot: int) -> None:
        """Activate a family default, or pin/unpin a preset on an alias."""
        selected = self._selected_data()
        model_name = self._selected_model()
        if not model_name:
            self.notify("Select a model card first", severity="warning")
            return
        if selected and selected[0] == "alias":
            alias_name = selected[1]
            target = self.registry.aliases[alias_name]
            if target.preset_slot == slot:
                target.quant = None
                target.preset_slot = None
                save_registry(self.paths.models_json, self.registry)
                self._reload_registry()
                return
        quant = self._model_active_quant(model_name)
        preset = get_preset(self.preset_store, model_name, quant, slot)
        if preset is None:
            self.notify(
                f"No preset [{slot}] for {self.registry.models[model_name].display}",
                severity="warning",
            )
            return
        if selected and selected[0] == "alias":
            alias_name = selected[1]
            target = self.registry.aliases[alias_name]
            target.quant = quant
            target.preset_slot = slot
            save_registry(self.paths.models_json, self.registry)
            self._reload_registry()
            return
        set_active_preset(self.preset_store, model_name, quant, slot)
        save_presets(self.paths.presets_json, self.preset_store)
        self._reload_registry()

    def action_open_hub(self) -> None:
        if self._editor_mode:
            self.notify("Close the editor first", severity="warning")
            return

        def on_complete() -> None:
            self.registry = load_registry(self.paths.models_json, models_dir=self.paths.models_dir)
            self._reload_registry()

        self.push_screen(
            HubScreen(
                registry=self.registry,
                settings=self.settings,
                settings_path=self.paths.settings_json,
                models_json_path=self.paths.models_json,
                models_dir=self.paths.models_dir,
                baselines_path=self.paths.baselines_json,
                on_complete=on_complete,
            )
        )

    def action_help(self) -> None:
        self.notify(HELP_TEXT, title="Help", timeout=15)

    def notify(self, message: str, *, title: str = "", severity: str = "information",
               timeout: float | None = None, **kwargs):
        if timeout is not None:
            try:
                return super().notify(message, title=title, severity=severity,
                                      timeout=timeout, **kwargs)
            except TypeError:
                pass
        return super().notify(message, title=title, severity=severity, **kwargs)


def main() -> None:
    app = LLMServeApp()
    app.run()


if __name__ == "__main__":
    main()
