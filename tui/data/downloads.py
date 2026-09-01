"""Background Hugging Face downloads shared by Hub and quant picker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from tui.data.hf import DownloadPlan, download_files, fmt_size, local_download_bytes


@dataclass
class DownloadJob:
    plan: DownloadPlan
    filename: str
    expected_bytes: int
    model_slug: str | None = None
    clone_from: str | None = None
    display: str | None = None
    on_success: Callable[[], None] | None = None
    on_error: Callable[[str], None] | None = None


@dataclass
class DownloadState:
    running: bool = False
    filename: str = ""
    status_line: str = ""
    used_bytes: int = 0
    expected_bytes: int = 0
    elapsed_s: int = 0
    cli_line: str = ""

    @property
    def progress_pct(self) -> float | None:
        if self.expected_bytes > 0 and self.used_bytes >= 0:
            return min(100.0, 100.0 * self.used_bytes / self.expected_bytes)
        return None

    @property
    def progress_total(self) -> int | None:
        return self.expected_bytes if self.expected_bytes > 0 else None


class DownloadManager:
    def __init__(self) -> None:
        self.state = DownloadState()
        self._task: asyncio.Task | None = None
        self._listeners: list[Callable[[DownloadState], None]] = []

    def subscribe(self, listener: Callable[[DownloadState], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self.state)

    @property
    def busy(self) -> bool:
        return self.state.running

    def format_status(self) -> str:
        st = self.state
        if not st.running:
            return ""
        if st.expected_bytes > 0:
            pct = st.progress_pct or 0.0
            size = f"{fmt_size(st.used_bytes)} / {fmt_size(st.expected_bytes)} ({pct:.0f}%)"
        elif st.used_bytes:
            size = fmt_size(st.used_bytes)
        else:
            size = "starting…"
        extra = f"  {st.cli_line[:60]}" if st.cli_line else ""
        return f"Downloading {st.filename}  {size}  {st.elapsed_s}s{extra}"

    async def run(self, job: DownloadJob) -> tuple[bool, str]:
        if self.state.running:
            return False, "Another download is already running"

        self.state = DownloadState(
            running=True,
            filename=job.filename,
            expected_bytes=job.expected_bytes,
        )
        self.state.status_line = self.format_status()
        self._notify()

        started = asyncio.get_running_loop().time()
        cli_line = ""

        def on_line(line: str) -> None:
            nonlocal cli_line
            text = line.strip()
            if text:
                cli_line = text
                self.state.cli_line = text

        try:
            download_task = asyncio.create_task(download_files(job.plan, on_line=on_line))
            while not download_task.done():
                self.state.used_bytes = local_download_bytes(job.plan)
                self.state.elapsed_s = int(asyncio.get_running_loop().time() - started)
                self.state.status_line = self.format_status()
                self._notify()
                await asyncio.sleep(0.25)
            ok, message = download_task.result()
            self.state.used_bytes = local_download_bytes(job.plan)
            self.state.elapsed_s = int(asyncio.get_running_loop().time() - started)
            self.state.status_line = self.format_status()
            self._notify()
            return ok, message
        finally:
            self.state.running = False
            self.state.status_line = ""
            self._notify()
