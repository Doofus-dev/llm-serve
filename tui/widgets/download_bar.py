"""Progress strip shown while a Hub download is running."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar


class DownloadBar(Vertical):
    """Progress strip shown only while a Hub download is running."""

    DEFAULT_CSS = """
    DownloadBar {
        height: 0;
        overflow: hidden;
        padding: 0;
        border: none;
    }

    DownloadBar.visible {
        height: 3;
        padding: 0 1;
        border-bottom: solid $warning;
        background: $surface-darken-1;
    }

    DownloadBar #download-label {
        height: 1;
        color: $warning;
    }

    DownloadBar ProgressBar {
        height: 1;
        width: 1fr;
    }
    """

    def apply_state(self, state) -> None:
        """Update visibility and progress from DownloadState."""
        if state.running:
            self.add_class("visible")
            label = self.query_one("#download-label", Label)
            label.update(state.status_line or f"Downloading {state.filename}…")
            bar = self.query_one("#download-progress", ProgressBar)
            total = state.progress_total
            if total:
                bar.update(total=float(total), progress=float(state.used_bytes))
            else:
                bar.update(total=None, progress=0.0)
        else:
            self.remove_class("visible")
            self.query_one("#download-progress", ProgressBar).update(total=None)

    def compose(self) -> ComposeResult:
        yield Label("", id="download-label")
        yield ProgressBar(total=None, id="download-progress", show_eta=False)

