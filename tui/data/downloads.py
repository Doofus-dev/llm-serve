"""Background Hugging Face downloads shared by Hub and quant picker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Literal

from tui.data.hf import DownloadPlan, download_files, fmt_size, local_download_bytes

EnqueueResult = Literal["started", "duplicate"]


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

    @property
    def label(self) -> str:
        if self.display:
            return f"{self.display} · {self.filename}"
        return self.filename


@dataclass
class DownloadProgress:
    key: str
    filename: str
    model_slug: str = ""
    display: str = ""
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
    def label(self) -> str:
        if self.display:
            return f"{self.display} · {self.filename}"
        return self.filename


@dataclass
class DownloadState:
    running: bool = False
    filename: str = ""
    model_slug: str = ""
    display: str = ""
    status_line: str = ""
    used_bytes: int = 0
    expected_bytes: int = 0
    elapsed_s: int = 0
    cli_line: str = ""
    queued: tuple[str, ...] = ()
    jobs: tuple[DownloadProgress, ...] = field(default_factory=tuple)

    @property
    def queued_count(self) -> int:
        return len(self.queued)

    @property
    def active_count(self) -> int:
        return len(self.jobs)

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
        return self.running or bool(self.queued) or bool(self.jobs)

    def progress_for(self, filename: str) -> DownloadProgress | None:
        for item in self.jobs:
            if item.filename == filename or item.key.endswith(f"/{filename}"):
                return item
        return None

    def is_transferring(self, filename: str) -> bool:
        return self.progress_for(filename) is not None


class DownloadManager:
    def __init__(self) -> None:
        self.state = DownloadState()
        self._active: dict[str, DownloadJob] = {}
        self._progress: dict[str, DownloadProgress] = {}
        self._listeners: list[Callable[[DownloadState], None]] = []

    def subscribe(self, listener: Callable[[DownloadState], None]) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[DownloadState], None]) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self.state)

    def _rebuild(self) -> None:
        jobs = tuple(self._progress.values())
        used = sum(item.used_bytes for item in jobs)
        expected = sum(item.expected_bytes for item in jobs)
        primary = jobs[0] if jobs else None
        self.state = DownloadState(
            running=bool(jobs),
            filename=primary.filename if primary else "",
            model_slug=primary.model_slug if primary else "",
            display=primary.display if primary else "",
            used_bytes=used,
            expected_bytes=expected,
            elapsed_s=max((item.elapsed_s for item in jobs), default=0),
            cli_line=primary.cli_line if primary else "",
            jobs=jobs,
        )
        self.state.status_line = self.format_status()

    @property
    def busy(self) -> bool:
        return bool(self._active)

    @property
    def queue_size(self) -> int:
        return max(0, len(self._active) - 1)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def has_job(self, key: str) -> bool:
        return key in self._active

    def has_filename(self, filename: str) -> bool:
        return any(job.filename == filename for job in self._active.values())

    def enqueue(self, job: DownloadJob) -> EnqueueResult:
        """Register a job so it can start immediately. Duplicates are ignored."""
        if self.has_job(job.key):
            return "duplicate"
        self._active[job.key] = job
        self._progress[job.key] = DownloadProgress(
            key=job.key,
            filename=job.filename,
            model_slug=job.model_slug or job.clone_from or "",
            display=job.display or "",
            expected_bytes=job.expected_bytes,
        )
        self._rebuild()
        self._notify()
        return "started"

    def format_status(self) -> str:
        st = self.state
        if not st.jobs:
            return ""
        parts: list[str] = []
        for item in st.jobs:
            if item.expected_bytes > 0:
                pct = item.progress_pct or 0.0
                size = (
                    f"{fmt_size(item.used_bytes)} / "
                    f"{fmt_size(item.expected_bytes)} ({pct:.0f}%)"
                )
            elif item.used_bytes:
                size = fmt_size(item.used_bytes)
            else:
                size = "starting…"
            extra = f"  {item.cli_line[:40]}" if item.cli_line else ""
            parts.append(f"{item.label}  {size}{extra}")
        if len(parts) == 1:
            return f"Downloading {parts[0]}  {st.elapsed_s}s"
        joined = "  ·  ".join(parts)
        return f"Downloading {len(parts)} files  {st.elapsed_s}s  ·  {joined}"

    async def run(self, job: DownloadJob) -> tuple[bool, str]:
        if job.key not in self._active:
            if self.enqueue(job) == "duplicate" and job.key not in self._active:
                return False, f"{job.filename} is already downloading"

        progress = self._progress.get(job.key)
        if progress is None:
            progress = DownloadProgress(
                key=job.key,
                filename=job.filename,
                model_slug=job.model_slug or job.clone_from or "",
                display=job.display or "",
                expected_bytes=job.expected_bytes,
            )
            self._progress[job.key] = progress
            self._rebuild()
            self._notify()

        started = asyncio.get_running_loop().time()

        def on_line(line: str) -> None:
            text = line.strip()
            if text:
                progress.cli_line = text

        try:
            download_task = asyncio.create_task(download_files(job.plan, on_line=on_line))
            while not download_task.done():
                progress.used_bytes = local_download_bytes(job.plan)
                progress.elapsed_s = int(asyncio.get_running_loop().time() - started)
                self._rebuild()
                self._notify()
                await asyncio.sleep(0.25)
            ok, message = download_task.result()
            progress.used_bytes = local_download_bytes(job.plan)
            progress.elapsed_s = int(asyncio.get_running_loop().time() - started)
            self._rebuild()
            self._notify()
            return ok, message
        finally:
            self._active.pop(job.key, None)
            self._progress.pop(job.key, None)
            self._rebuild()
            self._notify()
