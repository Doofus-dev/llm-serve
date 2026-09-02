"""Hugging Face Hub CLI wrapper for model browse and download."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HF_INSTALL_HINT = "Install: curl -LsSf https://hf.co/cli/install.sh | bash -s"

SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d+)-of-(?P<total>\d+)\.gguf$", re.IGNORECASE)


@dataclass(frozen=True)
class HubRepo:
    id: str
    author: str
    downloads: int
    likes: int
    trending_score: int | None = None
    size: int = 0
    context_length: int | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HubRepo:
        repo_id = str(data.get("id", ""))
        author = repo_id.split("/", 1)[0] if "/" in repo_id else repo_id
        return cls(
            id=repo_id,
            author=author,
            downloads=int(data.get("downloads") or 0),
            likes=int(data.get("likes") or 0),
            size=int(
                (data.get("gguf") or {}).get("total")
                or data.get("total_file_size")
                or 0
            ),
            trending_score=int(data["trending_score"]) if data.get("trending_score") is not None else None,
            context_length=(
                int((data.get("gguf") or {}).get("context_length"))
                if (data.get("gguf") or {}).get("context_length") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class HubFile:
    path: str
    size: int

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HubFile | None:
        path = str(data.get("path", ""))
        if not path.lower().endswith(".gguf"):
            return None
        size = int(data.get("size") or data.get("lfs", {}).get("size") or 0)
        return cls(path=path, size=size)


@dataclass(frozen=True)
class AuthStatus:
    logged_in: bool
    name: str | None = None
    orgs: list[str] | None = None


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    author: str
    filenames: list[str]
    local_dir: Path
    relative_file: str
    revision: str = "main"


def hf_available() -> bool:
    return shutil.which("hf") is not None


def repo_author(repo_id: str) -> str:
    return repo_id.split("/", 1)[0] if "/" in repo_id else repo_id


def build_download_plan(
    repo_id: str,
    filename: str,
    models_dir: Path,
    *,
    all_ggufs: list[str] | None = None,
    revision: str = "main",
) -> DownloadPlan:
    """Build local paths and file list for a Hub download."""
    author = repo_author(repo_id)
    local_dir = models_dir / author
    filenames = shard_filenames(filename, all_ggufs or [])
    relative_file = f"{author}/{filename}"
    return DownloadPlan(
        repo_id=repo_id,
        author=author,
        filenames=filenames,
        local_dir=local_dir,
        relative_file=relative_file,
        revision=revision,
    )


def shard_filenames(selected: str, all_ggufs: list[str]) -> list[str]:
    """If selected is a sharded GGUF, return all shards in the set."""
    match = SHARD_RE.match(selected)
    if not match:
        return [selected]

    prefix = match.group("prefix")
    total = match.group("total")
    pattern = re.compile(rf"^{re.escape(prefix)}-\d+-of-{total}\.gguf$", re.IGNORECASE)
    shards = sorted(f for f in all_ggufs if pattern.match(f))
    return shards or [selected]


def build_source_metadata(plan: DownloadPlan, filename: str) -> dict[str, str]:
    return {
        "hub": "huggingface",
        "repo": plan.repo_id,
        "filename": filename,
        "author": plan.author,
        "revision": plan.revision,
    }


def _run_hf(args: list[str], *, timeout: float | None = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["hf", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def take_hf_progress_lines(buffer: bytes) -> tuple[list[str], bytes]:
    """Split hf CLI output on newline or carriage return.

    Download progress (tqdm) rewrites the same line with ``\\r``, so
    ``readline()`` stays silent until the whole file finishes.
    """
    lines: list[str] = []
    start = 0
    for index, byte in enumerate(buffer):
        if byte not in (10, 13):
            continue
        piece = buffer[start:index]
        start = index + 1
        if piece:
            lines.append(piece.decode("utf-8", errors="replace"))
    return lines, buffer[start:]


async def _run_hf_async(
    args: list[str],
    *,
    timeout: float | None = None,
    on_line: Callable[[str], None] | None = None,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        "hf",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    chunks: list[str] = []
    leftover = b""
    assert proc.stdout is not None
    while True:
        block = await proc.stdout.read(512)
        if not block:
            break
        leftover += block
        lines, leftover = take_hf_progress_lines(leftover)
        for text in lines:
            chunks.append(text)
            if on_line:
                on_line(text)
    if leftover:
        text = leftover.decode("utf-8", errors="replace").strip()
        if text:
            chunks.append(text)
            if on_line:
                on_line(text)
    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    output = "\n".join(chunks)
    return rc, output, output


def auth_whoami() -> AuthStatus:
    if not hf_available():
        return AuthStatus(logged_in=False)
    result = _run_hf(["auth", "whoami", "--format", "json"], timeout=30)
    if result.returncode != 0:
        return AuthStatus(logged_in=False)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return AuthStatus(logged_in=False)
    if isinstance(data, dict):
        name = data.get("name") or data.get("fullname")
        orgs = data.get("orgs")
        if isinstance(orgs, list):
            org_names = [str(o.get("name", o)) if isinstance(o, dict) else str(o) for o in orgs]
        else:
            org_names = None
        return AuthStatus(logged_in=bool(name), name=str(name) if name else None, orgs=org_names)
    return AuthStatus(logged_in=False)


def auth_login(token: str) -> tuple[bool, str]:
    if not hf_available():
        return False, f"hf CLI not found. {HF_INSTALL_HINT}"
    token = token.strip()
    if not token:
        return False, "Token cannot be empty"
    result = _run_hf(["auth", "login", "--token", token], timeout=60)
    if result.returncode == 0:
        return True, "Logged in to Hugging Face"
    message = (result.stderr or result.stdout or "Login failed").strip()
    return False, message


def list_gguf_repos(
    *,
    author: str = "",
    search: str = "",
    limit: int = 30,
    sort: str = "trending_score",
) -> tuple[list[HubRepo], str | None]:
    if not hf_available():
        return [], f"hf CLI not found. {HF_INSTALL_HINT}"

    args = [
        "models", "list",
        "--filter", "gguf",
        "--sort", sort,
        "--limit", str(limit),
        "--expand", "gguf",
        "--format", "json",
    ]
    if author.strip():
        args.extend(["--author", author.strip()])
    if search.strip():
        args.extend(["--search", search.strip()])

    result = _run_hf(args, timeout=120)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "Failed to list models").strip()
        return [], message

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return [], f"Invalid JSON from hf: {exc}"

    if not isinstance(data, list):
        return [], "Unexpected response from hf models list"

    repos = [HubRepo.from_json(item) for item in data if isinstance(item, dict) and item.get("id")]
    return repos, None


def list_repo_ggufs(repo_id: str, *, revision: str = "main") -> tuple[list[HubFile], str | None]:
    if not hf_available():
        return [], f"hf CLI not found. {HF_INSTALL_HINT}"

    args = ["models", "list", repo_id, "--format", "json"]
    if revision:
        args.extend(["--revision", revision])

    result = _run_hf(args, timeout=120)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "Failed to list repo files").strip()
        return [], message

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return [], f"Invalid JSON from hf: {exc}"

    if not isinstance(data, list):
        return [], "Unexpected response from hf models list"

    files: list[HubFile] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        parsed = HubFile.from_json(item)
        if parsed is not None:
            files.append(parsed)
    files.sort(key=lambda f: f.path.lower())
    return files, None


async def download_files(
    plan: DownloadPlan,
    *,
    on_line: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    if not hf_available():
        return False, f"hf CLI not found. {HF_INSTALL_HINT}"

    plan.local_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "download",
        plan.repo_id,
        *plan.filenames,
        "--local-dir",
        str(plan.local_dir),
        "--revision",
        plan.revision,
    ]
    rc, output, _ = await _run_hf_async(args, on_line=on_line)
    if rc == 0:
        return True, output or "Download complete"
    return False, output or "Download failed"


def local_download_bytes(plan: DownloadPlan) -> int:
    """Bytes already on disk for this plan (finished or still arriving)."""
    total = 0
    cache_dir = plan.local_dir / ".cache" / "huggingface" / "download"
    cache_files: set[Path] = set()
    target_locks: list[Path] = []
    for name in plan.filenames:
        direct = plan.local_dir / name
        if direct.is_file():
            total += direct.stat().st_size
            continue
        incomplete = plan.local_dir / f"{name}.incomplete"
        if incomplete.is_file():
            total += incomplete.stat().st_size
            continue
        lock = cache_dir / f"{name}.lock"
        if not lock.is_file():
            continue
        target_locks.append(lock)

        # huggingface_hub names the transfer file from its ETag rather than
        # the requested filename. Completed metadata records that ETag on its
        # second line, which lets resumed downloads be matched exactly.
        metadata = cache_dir / f"{name}.metadata"
        if metadata.is_file():
            try:
                lines = metadata.read_text().splitlines()
            except OSError:
                lines = []
            if len(lines) >= 2 and lines[1]:
                candidate = cache_dir / f"{lines[1]}.incomplete"
                if candidate.is_file():
                    cache_files.add(candidate)

    if target_locks and cache_dir.is_dir():
        # Metadata is commonly written only after completion. During a fresh
        # transfer, count incomplete files created/updated since this plan's
        # lock was acquired. Old lock files from earlier downloads must not
        # disable progress reporting.
        lock_mtimes: list[int] = []
        for lock in target_locks:
            try:
                lock_mtimes.append(lock.stat().st_mtime_ns)
            except OSError:
                continue
        if lock_mtimes:
            oldest_lock_mtime = min(lock_mtimes)
            for candidate in cache_dir.glob("*.incomplete"):
                try:
                    if candidate.stat().st_mtime_ns >= oldest_lock_mtime:
                        cache_files.add(candidate)
                except OSError:
                    continue

    for path in cache_files:
        try:
            total += path.stat().st_size
        except OSError:
            # The final rename can happen between polling and stat.
            continue
    return total


def fmt_size(size: int) -> str:
    if size <= 0:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"
