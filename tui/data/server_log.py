"""Parse llama.cpp / llm-serve logs into a short, readable event feed."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tui.data.log_collapse import classify_family

CacheMode = Literal["lcp", "lru"]
EventKind = Literal[
    "launch",
    "cors",
    "unused_tensors",
    "loading",
    "ready",
    "hint",
    "request",
    "cancel",
    "shutdown",
    "http",
    "warn",
    "error",
    "info",
    "idle",
    "prefill",
    "generate",
    "checkpoint",
]
EventLabel = Literal[
    "started",
    "loading",
    "ready",
    "request",
    "warn",
    "error",
    "hint",
    "info",
    "cancel",
    "stopped",
    "http",
    "idle",
    "prefill",
    "gen",
    "ckpt",
]

FALLBACK_SESSION_LINES = 400


def is_unformatted_event(event: "LogEvent") -> bool:
    """Leftover llama.cpp lines that were not turned into a recap."""
    return event.kind == "info"

_LAUNCH = re.compile(
    r"^──\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+launch:\s+(\S+)\s+\(PID\s+(\d+)\)\s+──\s*$"
)
_LLAMA_FULL = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+([IWED])\s+([^\s:]+)\s+([^\s:]+):\s*(.*)$"
)
_LLAMA_FN = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+([IWED])\s+(\S+):\s*(.*)$"
)
_LLAMA_FREE = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+([IWED])\s+(.*)$"
)
_SLOT_BODY = re.compile(
    r"id\s+\d+\s*\|\s*task\s+(-?\d+)\s*\|\s*(.*)$"
)
_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?"
    r"([\d.]+)\s*tokens per second",
    re.DOTALL,
)
_EVAL = re.compile(
    r"(?<![a-z])eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?"
    r"([\d.]+)\s*tokens per second",
    re.DOTALL,
)
_RELEASE = re.compile(
    r"stop processing:\s*n_tokens\s*=\s*(\d+),\s*truncated\s*=\s*(\d+)"
)
_LCP = re.compile(
    r"selected slot by LCP similarity,\s*f_sim_best\s*=\s*([\d.]+).*?f_keep\s*=\s*([\d.]+)"
)
_LRU = re.compile(r"selected slot by LRU")
_SLOTS_INIT = re.compile(
    r"initializing,\s*n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*(\d+)"
)
_LISTEN = re.compile(r"listening on\s+(\S+)")
_LOAD_PATH = re.compile(r"loading model\s+'([^']+)'")
_THREADS = re.compile(r"n_threads\s*=\s*(\d+)")
_TASK_ID = re.compile(r"id_task\s*=\s*(\d+)")
_CORS_NOISE = re.compile(r"^-{3,}$")
_DONE_REQUEST = re.compile(
    r"done request:\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)"
)
_PROMPT_PROGRESS = re.compile(
    r"prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([\d.]+).*?"
    r"([\d.]+)\s*tokens per second",
    re.DOTALL,
)
_N_GEN = re.compile(
    r"n_gen\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+)\s*t/s"
)
_CHECKPOINT = re.compile(
    r"created context checkpoint\s+(\d+)\s+of\s+(\d+).*?"
    r"n_tokens\s*=\s*(\d+).*?size\s*=\s*([\d.]+)\s*MiB",
    re.DOTALL,
)
_RESTORED = re.compile(
    r"restored context checkpoint.*?n_tokens\s*=\s*(\d+).*?size\s*=\s*([\d.]+)\s*MiB",
    re.DOTALL,
)


@dataclass
class ParsedLine:
    raw: str
    elapsed_s: float | None = None
    level: str = ""
    category: str = ""
    function: str = ""
    message: str = ""
    is_launch: bool = False
    launch_model: str = ""
    launch_pid: int | None = None
    launch_when: str = ""


@dataclass
class LogEvent:
    kind: EventKind
    label: EventLabel
    message: str
    raw_lines: list[str]
    elapsed_s: float | None = None
    detail: str | None = None
    severity: str = "info"
    cache_mode: CacheMode | None = None
    f_sim: float | None = None
    f_keep: float | None = None
    prompt_tokens: int | None = None
    prompt_tps: float | None = None
    gen_tokens: int | None = None
    gen_tps: float | None = None
    context_tokens: int | None = None
    truncated: bool = False
    unused_count: int | None = None
    model: str | None = None
    pid: int | None = None
    listen_url: str | None = None
    n_slots: int | None = None
    n_ctx_slot: int | None = None
    count: int = 1
    family: str | None = None
    replace_last: bool = False
    progress: float | None = None
    ckpt_index: int | None = None
    ckpt_of: int | None = None
    ckpt_mib: float | None = None
    restored: bool = False
    client_addr: str | None = None
    http_status: int | None = None


@dataclass
class _RequestBuf:
    task_id: int
    raw_lines: list[str] = field(default_factory=list)
    elapsed_s: float | None = None
    cache_mode: CacheMode | None = None
    f_sim: float | None = None
    f_keep: float | None = None
    prompt_tokens: int | None = None
    prompt_tps: float | None = None
    gen_tokens: int | None = None
    gen_tps: float | None = None
    context_tokens: int | None = None
    truncated: bool | None = None
    client_addr: str | None = None
    http_status: int | None = None


def elapsed_from_fields(minutes: int, seconds: int, ms: int, us: int) -> float:
    """llama.cpp stamps MINUTES.SS.mmm.uuu from process start."""
    return minutes * 60 + seconds + ms / 1000.0 + us / 1_000_000.0


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{int(seconds // 3600)}h"


def fmt_tok(n: int) -> str:
    return f"{n:,}"


def fmt_tps(n: float) -> str:
    if n >= 100:
        return f"{n:.0f}"
    return f"{n:.1f}"


def fmt_mib(n: float | int) -> str:
    value = float(n)
    if value >= 100:
        return f"{value:,.0f} MiB"
    return f"{value:.1f} MiB"


def parse_line(raw: str) -> ParsedLine | None:
    line = raw.rstrip("\n\r")
    if not line.strip():
        return None

    launch = _LAUNCH.match(line)
    if launch:
        return ParsedLine(
            raw=line,
            is_launch=True,
            launch_when=launch.group(1),
            launch_model=launch.group(2),
            launch_pid=int(launch.group(3)),
        )

    full = _LLAMA_FULL.match(line)
    if full:
        return ParsedLine(
            raw=line,
            elapsed_s=elapsed_from_fields(*(int(full.group(i)) for i in range(1, 5))),
            level=full.group(5),
            category=full.group(6),
            function=full.group(7).strip().rstrip("_"),
            message=full.group(8),
        )

    fn = _LLAMA_FN.match(line)
    if fn:
        return ParsedLine(
            raw=line,
            elapsed_s=elapsed_from_fields(*(int(fn.group(i)) for i in range(1, 5))),
            level=fn.group(5),
            function=fn.group(6),
            message=fn.group(7),
        )

    free = _LLAMA_FREE.match(line)
    if free:
        return ParsedLine(
            raw=line,
            elapsed_s=elapsed_from_fields(*(int(free.group(i)) for i in range(1, 5))),
            level=free.group(5),
            message=free.group(6),
        )

    return ParsedLine(raw=line, message=line)


def is_launch_marker(line: str) -> bool:
    return bool(_LAUNCH.match(line.rstrip("\n\r")))


def slice_to_session(text: str, fallback_lines: int = FALLBACK_SESSION_LINES) -> str:
    """Keep from the last llm-serve launch marker, else the last N lines."""
    lines = text.splitlines(keepends=True)
    idx = None
    for i, line in enumerate(lines):
        if is_launch_marker(line):
            idx = i
    if idx is not None:
        return "".join(lines[idx:])
    return "".join(lines[-fallback_lines:])


def _esc(text: str) -> str:
    return text.replace("[", "\\[")


def _is_cors_line(parsed: ParsedLine) -> bool:
    if parsed.function != "llama_server":
        return False
    msg = parsed.message.strip()
    if _CORS_NOISE.match(msg):
        return True
    lowered = msg.lower()
    return any(
        needle in lowered
        for needle in ("cors", "api key", "security risk", "more info:")
    )


_HTTP_POLL_PATHS = {
    "/health",
    "/v1/health",
    "/models",
    "/v1/models",
    "/props",
    "/v1/props",
    "/metrics",
    "/slots",
    "/v1/slots",
}


def _is_loopback_addr(addr: str) -> bool:
    return addr in {"127.0.0.1", "::1", "localhost"}


def _is_http_noise(method: str, path: str, addr: str) -> bool:
    if path in _HTTP_POLL_PATHS:
        return True
    return method == "GET" and _is_loopback_addr(addr)


def _is_unused_tensor(parsed: ParsedLine) -> bool:
    return "unused tensor" in parsed.message.lower()


def _is_trace_noise(parsed: ParsedLine) -> bool:
    lowered = parsed.message.lower()
    if "cached n_tokens =" in lowered:
        return True
    if "checking sim" in lowered or "skipping, slot is empty" in lowered:
        return True
    if "init sampler" in lowered:
        return True
    if "graphs reused" in lowered or lowered.strip().startswith("total time"):
        return True
    if "sampler chain" in lowered or "sampler params" in lowered:
        return True
    if "looking for better prompt" in lowered or "cache state:" in lowered:
        return True
    if "prompt cache update" in lowered:
        return True
    if "erased invalidated" in lowered or "checking checkpoint" in lowered:
        return True
    if "conv_id=" in lowered:
        return True
    return False


def _slot_task_and_body(parsed: ParsedLine) -> tuple[int | None, str]:
    match = _SLOT_BODY.match(parsed.message)
    if not match:
        return None, parsed.message
    return int(match.group(1)), match.group(2).strip()


def _drop_timing_body(body: str) -> bool:
    lowered = body.lower()
    if "prompt processing" in lowered:
        return True
    if lowered.startswith("n_gen"):
        return True
    if "graphs reused" in lowered:
        return True
    if "total time" in lowered:
        return True
    return False


class LogAggregator:
    """Turn parsed log lines into collapsed, educational events."""

    def __init__(self) -> None:
        self._cors: list[ParsedLine] = []
        self._unused: list[ParsedLine] = []
        self._requests: dict[int, _RequestBuf] = {}
        self._pending_choice: tuple[CacheMode | None, float | None, float | None, str] | None = None
        self._n_slots: int | None = None
        self._n_ctx_slot: int | None = None
        self._pending_collapse: LogEvent | None = None
        self._emitted_collapse_this_batch = False
        self._recent_http: tuple[str, int] | None = None

    def reset(self) -> None:
        self._cors.clear()
        self._unused.clear()
        self._requests.clear()
        self._recent_http = None
        self._pending_choice = None
        self._n_slots = None
        self._n_ctx_slot = None
        self._pending_collapse = None
        self._emitted_collapse_this_batch = False

    def feed_text(self, text: str) -> list[LogEvent]:
        return self.feed_lines(text.splitlines())

    def feed_lines(self, lines: list[str]) -> list[LogEvent]:
        self._emitted_collapse_this_batch = False
        events: list[LogEvent] = []
        for raw in lines:
            events.extend(self._consume(raw.rstrip("\n\r")))
        return events

    def flush(self) -> list[LogEvent]:
        events: list[LogEvent] = []
        events.extend(self._flush_cors())
        events.extend(self._flush_unused())
        return events

    def _consume(self, line: str) -> list[LogEvent]:
        if not line.strip():
            return []
        parsed = parse_line(line)
        if parsed is None:
            return []

        events: list[LogEvent] = []
        if parsed.is_launch:
            self._pending_collapse = None
            events.extend(self.flush())
            self.reset()
            events.append(
                LogEvent(
                    kind="launch",
                    label="started",
                    severity="success",
                    elapsed_s=0.0,
                    message=f"{parsed.launch_model}  PID {parsed.launch_pid}",
                    raw_lines=[parsed.raw],
                    model=parsed.launch_model,
                    pid=parsed.launch_pid,
                )
            )
            return events

        if self._cors and not _is_cors_line(parsed):
            events.extend(self._flush_cors())
        if self._unused and not _is_unused_tensor(parsed):
            events.extend(self._flush_unused())

        if _is_cors_line(parsed):
            self._pending_collapse = None
            self._cors.append(parsed)
            return events
        if _is_unused_tensor(parsed):
            self._pending_collapse = None
            self._unused.append(parsed)
            return events

        handled = self._handle_known(parsed)
        if handled is not None:
            self._pending_collapse = None
            events.extend(handled)
            return events

        events.extend(self._ingest_passthrough(parsed))
        return events

    def _handle_known(self, parsed: ParsedLine) -> list[LogEvent] | None:
        if parsed.function == "DEPRECATED":
            return [
                LogEvent(
                    kind="hint",
                    label="warn",
                    severity="warn",
                    elapsed_s=parsed.elapsed_s,
                    message=f"deprecated flag: {parsed.message.strip()}",
                    raw_lines=[parsed.raw],
                )
            ]

        done = _DONE_REQUEST.search(parsed.message)
        if done:
            method, path, addr, status = done.groups()
            if _is_http_noise(method, path, addr):
                return []
            code = int(status)
            self._recent_http = (addr, code)
            if method == "POST" and "completions" in path and self._requests:
                if len(self._requests) == 1:
                    buf = next(iter(self._requests.values()))
                    buf.client_addr = addr
                    buf.http_status = code
                return []
            return [
                LogEvent(
                    kind="http",
                    label="http",
                    severity="info",
                    elapsed_s=parsed.elapsed_s,
                    message=f"{method} {path}  from {addr}  →  {status}",
                    raw_lines=[parsed.raw],
                    client_addr=addr,
                    http_status=code,
                )
            ]

        if parsed.message.startswith("request:") or parsed.message.startswith("response:"):
            return []

        if parsed.function == "load_model":
            path_match = _LOAD_PATH.search(parsed.message)
            if path_match:
                name = Path(path_match.group(1)).name
                return [
                    LogEvent(
                        kind="loading",
                        label="loading",
                        severity="info",
                        elapsed_s=parsed.elapsed_s,
                        message=name,
                        raw_lines=[parsed.raw],
                    )
                ]
            slots = _SLOTS_INIT.search(parsed.message)
            if slots:
                self._n_slots = int(slots.group(1))
                self._n_ctx_slot = int(slots.group(2))
                return []
            return None

        if parsed.function == "init" and "threadpool" in parsed.message:
            threads = _THREADS.search(parsed.message)
            n = threads.group(1) if threads else "?"
            return [
                LogEvent(
                    kind="hint",
                    label="hint",
                    severity="info",
                    elapsed_s=parsed.elapsed_s,
                    message=f"CPU threadpool  {n} threads",
                    raw_lines=[parsed.raw],
                )
            ]

        if parsed.function == "init" and "reasoning-preserve" in parsed.message:
            return [
                LogEvent(
                    kind="hint",
                    label="hint",
                    severity="info",
                    elapsed_s=parsed.elapsed_s,
                    message="chat template can preserve reasoning — consider --reasoning-preserve",
                    raw_lines=[parsed.raw],
                )
            ]

        if parsed.function == "llama_server":
            if "model loaded" in parsed.message:
                return []
            listen = _LISTEN.search(parsed.message)
            if listen:
                url = listen.group(1).rstrip("/")
                host = url.replace("http://", "").replace("https://", "")
                parts = [f"listening on {host}"]
                if self._n_slots is not None:
                    slot_word = "slot" if self._n_slots == 1 else "slots"
                    parts.append(f"{self._n_slots} {slot_word}")
                if self._n_ctx_slot is not None:
                    parts.append(f"ctx {fmt_tok(self._n_ctx_slot)}")
                return [
                    LogEvent(
                        kind="ready",
                        label="ready",
                        severity="success",
                        elapsed_s=parsed.elapsed_s,
                        message="  ·  ".join(parts),
                        raw_lines=[parsed.raw],
                        listen_url=url,
                        n_slots=self._n_slots,
                        n_ctx_slot=self._n_ctx_slot,
                    )
                ]
            return None

        if parsed.function == "common_param" and "verbosity" in parsed.message:
            return []

        if parsed.function == "stop" and "cancel task" in parsed.message:
            task_match = _TASK_ID.search(parsed.message)
            if task_match:
                self._requests.pop(int(task_match.group(1)), None)
            return [
                LogEvent(
                    kind="cancel",
                    label="cancel",
                    severity="warn",
                    elapsed_s=parsed.elapsed_s,
                    message="cancelled in-flight request",
                    raw_lines=[parsed.raw],
                )
            ]

        if "cleaning up before exit" in parsed.message:
            return [
                LogEvent(
                    kind="shutdown",
                    label="stopped",
                    severity="info",
                    elapsed_s=parsed.elapsed_s,
                    message="server stopped",
                    raw_lines=[parsed.raw],
                )
            ]

        if parsed.category == "slot":
            return self._handle_slot(parsed)

        return None

    def _ingest_passthrough(self, parsed: ParsedLine) -> list[LogEvent]:
        if parsed.elapsed_s is None and not parsed.is_launch:
            return []
        if _is_trace_noise(parsed):
            return []
        family = classify_family(parsed.raw)
        if family is None:
            family = "raw:" + parsed.raw
        event = self._format_trace(parsed, family) or self._passthrough(parsed)
        event.family = family
        pending = self._pending_collapse
        if pending is not None and pending.family == family:
            pending.count += 1
            pending.message = event.message
            pending.raw_lines.append(parsed.raw)
            pending.elapsed_s = event.elapsed_s
            pending.prompt_tokens = event.prompt_tokens
            pending.prompt_tps = event.prompt_tps
            pending.gen_tokens = event.gen_tokens
            pending.gen_tps = event.gen_tps
            pending.context_tokens = event.context_tokens
            pending.progress = event.progress
            pending.ckpt_index = event.ckpt_index
            pending.ckpt_of = event.ckpt_of
            pending.ckpt_mib = event.ckpt_mib
            pending.restored = event.restored
            pending.kind = event.kind
            pending.label = event.label
            pending.replace_last = True
            if self._emitted_collapse_this_batch:
                return []
            self._emitted_collapse_this_batch = True
            return [pending]
        event.count = 1
        event.replace_last = False
        self._pending_collapse = event
        self._emitted_collapse_this_batch = True
        return [event]

    def _format_trace(self, parsed: ParsedLine, family: str) -> LogEvent | None:
        body = parsed.message
        if family == "idle":
            return LogEvent(
                kind="idle",
                label="idle",
                severity="info",
                elapsed_s=parsed.elapsed_s,
                message="all slots idle",
                raw_lines=[parsed.raw],
            )
        progress = _PROMPT_PROGRESS.search(body)
        if progress:
            tokens = int(progress.group(1))
            pct = float(progress.group(2))
            tps = float(progress.group(3))
            return LogEvent(
                kind="prefill",
                label="prefill",
                severity="info",
                elapsed_s=parsed.elapsed_s,
                message=(
                    f"{fmt_tok(tokens)} tok  {pct * 100:.0f}%"
                    f"  @ {fmt_tps(tps)} t/s"
                ),
                raw_lines=[parsed.raw],
                prompt_tokens=tokens,
                prompt_tps=tps,
                progress=pct,
            )
        gen = _N_GEN.search(body)
        if gen:
            tokens = int(gen.group(1))
            tps = float(gen.group(2))
            return LogEvent(
                kind="generate",
                label="gen",
                severity="info",
                elapsed_s=parsed.elapsed_s,
                message=f"{fmt_tok(tokens)} tok  @ {fmt_tps(tps)} t/s",
                raw_lines=[parsed.raw],
                gen_tokens=tokens,
                gen_tps=tps,
            )
        ckpt = _CHECKPOINT.search(body)
        if ckpt:
            index = int(ckpt.group(1))
            total = int(ckpt.group(2))
            tokens = int(ckpt.group(3))
            mib = float(ckpt.group(4))
            return LogEvent(
                kind="checkpoint",
                label="ckpt",
                severity="info",
                elapsed_s=parsed.elapsed_s,
                message=(
                    f"{index}/{total}  {fmt_tok(tokens)} tok  {fmt_mib(mib)}"
                ),
                raw_lines=[parsed.raw],
                context_tokens=tokens,
                ckpt_index=index,
                ckpt_of=total,
                ckpt_mib=mib,
            )
        restored = _RESTORED.search(body)
        if restored:
            tokens = int(restored.group(1))
            mib = float(restored.group(2))
            return LogEvent(
                kind="checkpoint",
                label="ckpt",
                severity="info",
                elapsed_s=parsed.elapsed_s,
                message=f"restored  {fmt_tok(tokens)} tok  {fmt_mib(mib)}",
                raw_lines=[parsed.raw],
                context_tokens=tokens,
                ckpt_mib=mib,
                restored=True,
            )
        return None

    def _handle_slot(self, parsed: ParsedLine) -> list[LogEvent] | None:
        task_id, body = _slot_task_and_body(parsed)
        if parsed.function.startswith("get_availabl"):
            lcp = _LCP.search(parsed.message)
            if lcp:
                self._pending_choice = (
                    "lcp",
                    float(lcp.group(1)),
                    float(lcp.group(2)),
                    parsed.raw,
                )
                return []
            if _LRU.search(parsed.message):
                self._pending_choice = ("lru", None, None, parsed.raw)
                return []
            return []

        if parsed.function.startswith("launch_slot"):
            if task_id is not None and task_id >= 0:
                buf = self._request(task_id, parsed)
                buf.raw_lines.append(parsed.raw)
                if buf.elapsed_s is None:
                    buf.elapsed_s = parsed.elapsed_s
                self._apply_pending_choice(buf)
                return []
            return []

        if parsed.function == "print_timing":
            if task_id is None or task_id < 0:
                return None
            if _drop_timing_body(body):
                return None
            buf = self._request(task_id, parsed)
            buf.raw_lines.append(parsed.raw)
            if buf.elapsed_s is None:
                buf.elapsed_s = parsed.elapsed_s
            self._apply_pending_choice(buf)
            prompt = _PROMPT_EVAL.search(body)
            if prompt:
                buf.prompt_tokens = int(prompt.group(2))
                buf.prompt_tps = float(prompt.group(3))
                return []
            eval_match = _EVAL.search(body)
            if eval_match:
                buf.gen_tokens = int(eval_match.group(2))
                buf.gen_tps = float(eval_match.group(3))
                return []
            return None

        if parsed.function == "release":
            if task_id is None or task_id < 0:
                return None
            buf = self._request(task_id, parsed)
            buf.raw_lines.append(parsed.raw)
            rel = _RELEASE.search(body)
            if rel:
                buf.context_tokens = int(rel.group(1))
                buf.truncated = rel.group(2) != "0"
            event = self._emit_request(buf)
            self._requests.pop(task_id, None)
            return [event] if event else []

        return None

    def _request(self, task_id: int, parsed: ParsedLine) -> _RequestBuf:
        buf = self._requests.get(task_id)
        if buf is None:
            buf = _RequestBuf(task_id=task_id, elapsed_s=parsed.elapsed_s)
            self._requests[task_id] = buf
        return buf

    def _apply_pending_choice(self, buf: _RequestBuf) -> None:
        if self._pending_choice is None or buf.cache_mode is not None:
            return
        mode, f_sim, f_keep, raw = self._pending_choice
        buf.cache_mode = mode
        buf.f_sim = f_sim
        buf.f_keep = f_keep
        if raw not in buf.raw_lines:
            buf.raw_lines.insert(0, raw)
        self._pending_choice = None

    def _emit_request(self, buf: _RequestBuf) -> LogEvent | None:
        parts: list[str] = []
        if buf.cache_mode == "lcp" and buf.f_sim is not None:
            parts.append(f"reused cache {buf.f_sim * 100:.1f}%")
            if buf.f_keep is not None and buf.f_keep < 0.995:
                parts.append(f"kept {buf.f_keep * 100:.0f}% of KV")
        elif buf.cache_mode == "lru":
            parts.append("fresh prompt — no cache")

        if buf.prompt_tokens is not None:
            tps = f" @ {fmt_tps(buf.prompt_tps)} t/s" if buf.prompt_tps is not None else ""
            parts.append(f"read {fmt_tok(buf.prompt_tokens)} tok{tps}")
        if buf.gen_tokens is not None:
            tps = f" @ {fmt_tps(buf.gen_tps)} t/s" if buf.gen_tps is not None else ""
            parts.append(f"wrote {fmt_tok(buf.gen_tokens)} tok{tps}")
        if not parts:
            parts.append("finished")

        if buf.client_addr is None and self._recent_http is not None:
            buf.client_addr, buf.http_status = self._recent_http
        if buf.client_addr is not None:
            parts.append(f"from {buf.client_addr}")
        self._recent_http = None

        detail_parts: list[str] = []
        if buf.context_tokens is not None:
            detail_parts.append(f"context {fmt_tok(buf.context_tokens)}")
        if buf.truncated:
            detail_parts.append("context overflow — oldest tokens dropped")
        elif buf.truncated is False:
            detail_parts.append("no truncation")

        return LogEvent(
            kind="request",
            label="request",
            severity="error" if buf.truncated else "info",
            elapsed_s=buf.elapsed_s,
            message="  ·  ".join(parts),
            detail="  ·  ".join(detail_parts) if detail_parts else None,
            raw_lines=list(buf.raw_lines),
            cache_mode=buf.cache_mode,
            f_sim=buf.f_sim,
            f_keep=buf.f_keep,
            prompt_tokens=buf.prompt_tokens,
            prompt_tps=buf.prompt_tps,
            gen_tokens=buf.gen_tokens,
            gen_tps=buf.gen_tps,
            context_tokens=buf.context_tokens,
            truncated=bool(buf.truncated),
            client_addr=buf.client_addr,
            http_status=buf.http_status,
        )

    def _flush_cors(self) -> list[LogEvent]:
        if not self._cors:
            return []
        lines = self._cors
        self._cors = []
        return [
            LogEvent(
                kind="cors",
                label="warn",
                severity="warn",
                elapsed_s=lines[0].elapsed_s,
                message="no API key, CORS allows any origin — anyone who can reach this server can call it",
                raw_lines=[p.raw for p in lines],
            )
        ]

    def _flush_unused(self) -> list[LogEvent]:
        if not self._unused:
            return []
        lines = self._unused
        self._unused = []
        n = len(lines)
        return [
            LogEvent(
                kind="unused_tensors",
                label="warn",
                severity="warn",
                elapsed_s=lines[0].elapsed_s,
                message=(
                    f"ignored {n} unused tensor{'s' if n != 1 else ''} "
                    "(extra layers in this file, often MTP)"
                ),
                raw_lines=[p.raw for p in lines],
                unused_count=n,
            )
        ]

    def _passthrough(self, parsed: ParsedLine) -> LogEvent:
        if parsed.level == "E":
            kind: EventKind = "error"
            label: EventLabel = "error"
            severity = "error"
        elif parsed.level == "W":
            kind = "warn"
            label = "warn"
            severity = "warn"
        else:
            kind = "info"
            label = "info"
            severity = "info"
        text = parsed.message.strip() or parsed.raw
        if parsed.function and parsed.function not in text:
            text = f"{parsed.function}: {text}"
        return LogEvent(
            kind=kind,
            label=label,
            severity=severity,
            elapsed_s=parsed.elapsed_s,
            message=text,
            raw_lines=[parsed.raw],
        )


_LABEL_STYLE = {
    "started": "bold green",
    "loading": "cyan",
    "ready": "bold green",
    "request": "bold cyan",
    "warn": "bold yellow",
    "error": "bold red",
    "hint": "dim",
    "info": "dim",
    "cancel": "bold yellow",
    "stopped": "dim",
    "http": "cyan",
    "idle": "cyan",
    "prefill": "cyan",
    "gen": "cyan",
    "ckpt": "cyan",
}


def _y(value: object) -> str:
    return f"[bold yellow]{_esc(str(value))}[/]"


def _render_body(event: LogEvent) -> str:
    if event.kind == "launch":
        name = _esc(event.model or event.message)
        if event.pid is not None:
            return f"{name}  PID {_y(event.pid)}"
        return name
    if event.kind == "unused_tensors":
        n = event.unused_count if event.unused_count is not None else "?"
        noun = "tensor" if n == 1 else "tensors"
        return f"ignored {_y(n)} unused {noun} (extra layers in this file, often MTP)"
    if event.kind == "ready":
        host = (event.listen_url or "").replace("http://", "").replace("https://", "")
        parts = [f"listening on {_y(host or event.message)}"]
        if event.n_slots is not None:
            slot_word = "slot" if event.n_slots == 1 else "slots"
            parts.append(f"{_y(event.n_slots)} {slot_word}")
        if event.n_ctx_slot is not None:
            parts.append(f"ctx {_y(fmt_tok(event.n_ctx_slot))}")
        return "  ·  ".join(parts)
    if event.kind == "request":
        parts: list[str] = []
        if event.cache_mode == "lcp" and event.f_sim is not None:
            parts.append(f"reused cache {_y(f'{event.f_sim * 100:.1f}%')}")
            if event.f_keep is not None and event.f_keep < 0.995:
                parts.append(f"kept {_y(f'{event.f_keep * 100:.0f}%')} of KV")
        elif event.cache_mode == "lru":
            parts.append("fresh prompt — no cache")
        if event.prompt_tokens is not None:
            tps = f" @ {_y(fmt_tps(event.prompt_tps))} t/s" if event.prompt_tps is not None else ""
            parts.append(f"read {_y(fmt_tok(event.prompt_tokens))} tok{tps}")
        if event.gen_tokens is not None:
            tps = f" @ {_y(fmt_tps(event.gen_tps))} t/s" if event.gen_tps is not None else ""
            parts.append(f"wrote {_y(fmt_tok(event.gen_tokens))} tok{tps}")
        if event.client_addr:
            parts.append(f"from {_y(event.client_addr)}")
        return "  ·  ".join(parts) if parts else _esc(event.message)
    if event.kind == "http":
        if event.client_addr is not None and event.http_status is not None:
            method_path = event.message.split("  from ", 1)[0]
            return (
                f"{_esc(method_path)}  from {_y(event.client_addr)}"
                f"  →  {_y(event.http_status)}"
            )
        return _esc(event.message)
    if event.kind == "idle":
        body = "all slots idle"
        if event.count > 1:
            body += f"  ×{_y(event.count)}"
        return body
    if event.kind == "prefill":
        tokens = fmt_tok(event.prompt_tokens) if event.prompt_tokens is not None else "?"
        parts = [f"{_y(tokens)} tok"]
        if event.progress is not None:
            parts.append(f"{_y(f'{event.progress * 100:.0f}%')}")
        if event.prompt_tps is not None:
            parts.append(f"@ {_y(fmt_tps(event.prompt_tps))} t/s")
        body = "  ·  ".join(parts)
        if event.count > 1:
            body += f"  [dim]×{event.count}[/]"
        return body
    if event.kind == "generate":
        tokens = fmt_tok(event.gen_tokens) if event.gen_tokens is not None else "?"
        parts = [f"{_y(tokens)} tok"]
        if event.gen_tps is not None:
            parts.append(f"@ {_y(fmt_tps(event.gen_tps))} t/s")
        body = "  ·  ".join(parts)
        if event.count > 1:
            body += f"  [dim]×{event.count}[/]"
        return body
    if event.kind == "checkpoint":
        if event.restored:
            body = f"restored  {_y(fmt_tok(event.context_tokens or 0))} tok"
        elif event.ckpt_index is not None and event.ckpt_of is not None:
            body = f"{_y(f'{event.ckpt_index}/{event.ckpt_of}')}"
            if event.context_tokens is not None:
                body += f"  ·  {_y(fmt_tok(event.context_tokens))} tok"
        else:
            body = _esc(event.message)
        if event.ckpt_mib is not None:
            body += f"  ·  {_y(fmt_mib(event.ckpt_mib))}"
        if event.count > 1:
            body += f"  [dim]×{event.count}[/]"
        return body
    if event.severity == "warn":
        return f"[yellow]{_esc(event.message)}[/]"
    if event.severity == "error":
        return f"[bold red]{_esc(event.message)}[/]"
    if event.label in {"hint", "info", "stopped"}:
        body = f"[dim]{_esc(event.message)}[/]"
        if event.count > 1:
            body += f"  [dim]×{event.count}[/]"
        return body
    return _esc(event.message)


def render_event(event: LogEvent, *, show_raw: bool = False) -> str:
    ts = format_elapsed(event.elapsed_s)
    ts_bit = f"[dim]{ts:>4}[/]" if ts else "[dim]    [/]"
    style = _LABEL_STYLE.get(event.label, "cyan")
    label_bit = f"[{style}]{event.label:<8}[/]"
    lines = [f"{ts_bit}  {label_bit}  {_render_body(event)}"]
    if event.detail:
        if event.truncated:
            lines.append(f"            [bold red]{_esc(event.detail)}[/]")
        else:
            lines.append(f"            [dim]{_esc(event.detail)}[/]")
    if show_raw:
        for raw in event.raw_lines:
            lines.append(f"            [dim]{_esc(raw)}[/]")
    return "\n".join(lines)


class LogTailer:
    """Incrementally read a log file and emit new events."""

    def __init__(self) -> None:
        self.aggregator = LogAggregator()
        self._offset = 0
        self._partial = ""
        self._started = False

    def reset(self) -> None:
        self.aggregator.reset()
        self._offset = 0
        self._partial = ""
        self._started = False

    def poll(self, path: Path) -> tuple[list[LogEvent], bool]:
        """Return (new events, full_reload). full_reload replaces the session."""
        if not path.exists():
            self.reset()
            return [], True

        size = path.stat().st_size
        if not self._started or size < self._offset:
            self.aggregator.reset()
            data = path.read_bytes()
            text = data.decode("utf-8", errors="replace")
            session = slice_to_session(text)
            self._offset = size
            self._partial = ""
            self._started = True
            events = self.aggregator.feed_text(session)
            events.extend(self.aggregator.flush())
            return events, True

        if size == self._offset and not self._partial:
            return [], False

        with path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        self._offset += len(chunk)
        text = self._partial + chunk.decode("utf-8", errors="replace")
        if text.endswith("\n"):
            lines = text.splitlines()
            self._partial = ""
        else:
            parts = text.splitlines()
            if not parts:
                self._partial = text
                return [], False
            lines = parts[:-1]
            self._partial = parts[-1]
        return self.aggregator.feed_lines(lines), False
