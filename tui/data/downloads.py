"""Background Hugging Face downloads shared by Hub and quant picker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Literal

from tui.data.hf import DownloadPlan, download_files, fmt_size, local_download_bytes

EnqueueResult = Literal["started", "queued", "duplicate"]


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

    @property
    def key(self) -> str:
        return self.plan.relative_file or self.filename


@dataclass
class DownloadState:
    running: bool = False
    filename: str = ""
    status_line: str = ""
    used_bytes: int = 0
    expected_bytes: int = 0
    elapsed_s: int = 0
    cli_line: str = ""
    queued: tuple[str, ...] = ()

    @property
    def queued_count(self) -> int:
        return len(self.queued)

    @property
    def progress_pct(self) -> float | None:
        if self.expected_bytes > 0 and self.used_bytes >= 0:
            return min(100.0, 100.0 * self.used_bytes / self.expected_bytes)
        return None

    @property
    def progress_total(self) -> int | None:
        return self.expected_bytes if self.expected_bytes > 0 else None

    @property
    def active(self) -> bool:
        return self.running or bool(self.queued)


class DownloadManager:
    def __init__(self) -> None:
        self.state = DownloadState()
        self._queue: list[DownloadJob] = []
        self._current: DownloadJob | None = None
        self._listeners: list[Callable[[DownloadState], None]] = []

    def subscribe(self, listener: Callable[[DownloadState], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self.state)

    def _sync_queue(self) -> None:
        self.state.queued = tuple(job.filename for job in self._queue)

    @property
    def busy(self) -> bool:
        return self.state.running or bool(self._queue) or self._current is not None

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def has_job(self, key: str) -> bool:
        if self._current is not None and self._current.key == key:
            return True
        return any(job.key == key for job in self._queue)

    def enqueue(self, job: DownloadJob) -> EnqueueResult:
        """Add a job. Returns started, queued, or duplicate."""
        if self.has_job(job.key):
            return "duplicate"
        idle = not self.state.running and self._current is None and not self._queue
        self._queue.append(job)
        self._sync_queue()
        if self.state.running:
            self.state.status_line = self.format_status()
        self._notify()
        return "started" if idle else "queued"

    def pop_next(self) -> DownloadJob | None:
        if not self._queue:
            self._current = None
            self._sync_queue()
            self._notify()
            return None
        self._current = self._queue.pop(0)
        self._sync_queue()
        self._notify()
        return self._current

    def format_status(self) -> str:
        st = self.state
        if not st.running and not st.queued:
            return ""
        if st.expected_bytes > 0:
            pct = st.progress_pct or 0.0
            size = f"{fmt_size(st.used_bytes)} / {fmt_size(st.expected_bytes)} ({pct:.0f}%)"
        elif st.used_bytes:
            size = fmt_size(st.used_bytes)
        else:
            size = "starting…"
        extra = f"  {st.cli_line[:60]}" if st.cli_line else ""
        waiting = f"  · {st.queued_count} queued" if st.queued_count else ""
        name = st.filename or (st.queued[0] if st.queued else "download")
        verb = "Downloading" if st.running else "Queued"
        return f"{verb} {name}  {size}  {st.elapsed_s}s{extra}{waiting}"

    async def run(self, job: DownloadJob) -> tuple[bool, str]:
        if self.state.running:
            return False, "Another download is already running"

        self._current = job
        self.state = DownloadState(
            running=True,
            filename=job.filename,
            expected_bytes=job.expected_bytes,
            queued=tuple(queued.filename for queued in self._queue),
        )
        self.state.status_line = self.format_status()
        self._notify()

        started = asyncio.get_running_loop().time()

        def on_line(line: str) -> None:
            text = line.strip()
            if text:
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
            self._current = None
            self.state.running = False
            self._sync_queue()
            self.state.status_line = self.format_status() if self._queue else ""
            self._notify()
