"""Translated llama.cpp event feed."""

from __future__ import annotations

from pathlib import Path

from textual.geometry import Size
from textual.widgets import RichLog

from tui.data.server_log import LogEvent, LogTailer, is_unformatted_event, render_event
from tui.paths import MAX_LOG_EVENTS


class LogPanel(RichLog):
    """Translated llama.cpp events plus consecutive-run counters for leftover lines."""

    def __init__(self, **kwargs):
        super().__init__(
            markup=True,
            wrap=True,
            highlight=False,
            max_lines=400,
            auto_scroll=False,
            **kwargs,
        )
        self._tailer = LogTailer()
        self._events: list[LogEvent] = []
        self._show_info = False
        self._empty_shown = False
        self._last_strip_count = 0

    def _should_follow(self) -> bool:
        if self.is_vertical_scrollbar_grabbed:
            return False
        return self.is_vertical_scroll_end

    def _is_visible(self, event: LogEvent) -> bool:
        return self._show_info or not is_unformatted_event(event)

    def poll_file(self, path: Path) -> None:
        try:
            exists = path.exists()
            new_events, full_reload = self._tailer.poll(path)
        except OSError as e:
            self._tailer.reset()
            self._events.clear()
            self._last_strip_count = 0
            self.clear()
            self.write(f"[$error]log read error: {e}[/]", scroll_end=True)
            self._empty_shown = False
            return

        if not exists:
            if not self._empty_shown:
                self.clear()
                self._events.clear()
                self._last_strip_count = 0
                self.write("[dim]no log file yet[/]", scroll_end=True)
                self._empty_shown = True
            return

        if full_reload:
            self._events = list(new_events[-MAX_LOG_EVENTS:])
            self._rerender(follow=True)
            return

        if not new_events:
            if not self._events and not self._empty_shown:
                self._show_empty()
            return

        follow = self._should_follow()
        replaced = False
        appended: list[LogEvent] = []
        for event in new_events:
            if event.replace_last and self._events:
                self._events[-1] = event
                replaced = True
            else:
                self._events.append(event)
                appended.append(event)
        trimmed = False
        if len(self._events) > MAX_LOG_EVENTS:
            self._events = self._events[-MAX_LOG_EVENTS:]
            trimmed = True

        if trimmed:
            self._rerender(follow=follow)
            return
        if replaced and appended:
            self._rerender(follow=follow)
            return
        if replaced:
            last = self._events[-1]
            if self._is_visible(last):
                self._rewrite_last(follow=follow)
            return
        self._empty_shown = False
        for event in appended:
            written = self._write_event(event, follow=follow)
            if written:
                self._last_strip_count = written

    def _show_empty(self) -> None:
        self.clear()
        self._last_strip_count = 0
        self.write("[dim]no events yet — press L to launch[/]", scroll_end=True)
        self._empty_shown = True

    def _write_event(self, event: LogEvent, *, follow: bool) -> int:
        if not self._is_visible(event):
            return 0
        before = len(self.lines)
        self.write(render_event(event), scroll_end=follow)
        return max(0, len(self.lines) - before)

    def _drop_last_strips(self, count: int) -> None:
        if count <= 0 or count > len(self.lines):
            return
        del self.lines[-count:]
        self.virtual_size = Size(self._widest_line_width, len(self.lines))
        self.refresh()

    def _rewrite_last(self, *, follow: bool) -> None:
        if not self._events:
            self._show_empty()
            return
        self._empty_shown = False
        if self._last_strip_count:
            self._drop_last_strips(self._last_strip_count)
        self._last_strip_count = self._write_event(self._events[-1], follow=follow)

    def _rerender(self, *, follow: bool | None = None) -> None:
        if follow is None:
            follow = self._should_follow()
        pinned_y = self.scroll_y
        self.clear()
        self._last_strip_count = 0
        if not self._events:
            self._show_empty()
            return
        self._empty_shown = False
        last_written = 0
        for event in self._events:
            written = self._write_event(event, follow=False)
            if written:
                last_written = written
        self._last_strip_count = last_written
        if follow:
            self.scroll_end(animate=False, immediate=True, x_axis=False)
        else:
            self.scroll_to(y=pinned_y, animate=False, immediate=True)

    def toggle_info(self) -> bool:
        self._show_info = not self._show_info
        self._rerender()
        return self._show_info
