"""Rolling throughput history, live rate computation, and sparkline rendering."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

from rich.text import Text

from tui.data.stats import Metrics

# Unicode block steps for a btop-style single-line bar chart.
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
_PREFILL_BLOCKS = "░█"

# Fixed display width — must fit the THROUGHPUT column without wrapping.
SPARKLINE_WIDTH = 40
PREFILL_BAR_WIDTH = 12

ThroughputSource = Literal["slots", "metrics_delta", "metrics_gauge", "idle"]
RequestStage = Literal["idle", "queued", "prefill", "generating"]

_STAGE_STRIP = (
    ("idle", "idle"),
    ("queue", "queued"),
    ("cache", "cache"),
    ("prefill", "prefill"),
    ("generate", "generating"),
)
_ACTIVE_STAGES = frozenset({"prefill", "generating"})


@dataclass
class SlotSnapshot:
    n_decoded: int = 0
    n_prompt_processed: int = 0
    n_prompt_total: int = 0
    n_prompt_cache: int = 0
    is_processing: bool = False

    @property
    def in_prefill(self) -> bool:
        """True while a slot is processing prompt tokens but has not generated yet."""
        if not self.is_processing or self.n_decoded > 0:
            return False
        if self.n_prompt_total > 0:
            return self.n_prompt_processed < self.n_prompt_total
        return True


@dataclass
class LastRequest:
    prompt_tokens: int = 0
    cache_tokens: int = 0
    gen_tokens: int = 0
    gen_tps: float = 0.0


@dataclass
class LiveThroughput:
    gen_tps: float = 0.0
    prompt_tps: float = 0.0
    source: ThroughputSource = "idle"
    stage: RequestStage = "idle"
    n_prompt_processed: int = 0
    n_prompt_total: int = 0
    n_prompt_cache: int = 0
    n_decoded: int = 0
    requests_deferred: int = 0
    last_request: LastRequest | None = None


def summarize_slots(slots: list | None) -> SlotSnapshot | None:
    """Aggregate token progress across active slots."""
    if not slots:
        return None

    snap = SlotSnapshot()
    for slot in slots:
        if not slot.get("is_processing"):
            continue
        snap.is_processing = True
        snap.n_prompt_processed += int(slot.get("n_prompt_tokens_processed", 0) or 0)
        snap.n_prompt_total += int(slot.get("n_prompt_tokens", 0) or 0)
        snap.n_prompt_cache += int(slot.get("n_prompt_tokens_cache", 0) or 0)

        next_token = slot.get("next_token")
        if isinstance(next_token, list) and next_token:
            snap.n_decoded += int(next_token[0].get("n_decoded", 0) or 0)
        elif isinstance(next_token, dict):
            snap.n_decoded += int(next_token.get("n_decoded", 0) or 0)

    return snap


def _direct_slot_rates(slots: list | None) -> tuple[float, float]:
    """Read tg_tps / pp_tps when present on newer llama-server builds."""
    if not slots:
        return 0.0, 0.0

    tg_vals: list[float] = []
    pp_vals: list[float] = []
    for slot in slots:
        if not slot.get("is_processing"):
            continue
        if slot.get("tg_tps") is not None:
            tg_vals.append(float(slot["tg_tps"]))
        if slot.get("pp_tps") is not None:
            pp_vals.append(float(slot["pp_tps"]))

    return (max(tg_vals) if tg_vals else 0.0, max(pp_vals) if pp_vals else 0.0)


def _metrics_rate(
    requests_processing: float,
    derived: float,
    gauge: float,
) -> tuple[float, ThroughputSource]:
    # While a request is in flight, only counter deltas are instantaneous.
    # The gauge is averaged since the last /metrics scrape and goes stale fast.
    if requests_processing > 0:
        if derived > 0:
            return derived, "metrics_delta"
        return 0.0, "idle"
    if gauge > 0:
        return gauge, "metrics_gauge"
    return 0.0, "idle"


def _pick_rate(
    slot_rate: float,
    metrics_rate: float,
    metrics_source: ThroughputSource,
) -> tuple[float, ThroughputSource]:
    if slot_rate > 0:
        return slot_rate, "slots"
    if metrics_rate > 0:
        return metrics_rate, metrics_source
    return 0.0, "idle"


def _infer_stage(
    gen_tps: float,
    prompt_tps: float,
    requests_processing: float,
    requests_deferred: float,
    slot_snap: SlotSnapshot | None,
) -> RequestStage:
    processing = requests_processing > 0 or (
        slot_snap is not None and slot_snap.is_processing
    )
    if not processing:
        if requests_deferred > 0:
            return "queued"
        return "idle"
    if slot_snap is not None and slot_snap.in_prefill:
        return "prefill"
    if slot_snap is not None and slot_snap.n_decoded > 0:
        return "generating"
    if gen_tps > 0:
        return "generating"
    if prompt_tps > 0:
        return "prefill"
    if slot_snap is not None and slot_snap.n_decoded == 0:
        return "prefill"
    return "generating"


def _copy_snap(live: LiveThroughput, snap: SlotSnapshot | None) -> None:
    if snap is None:
        return
    live.n_prompt_processed = snap.n_prompt_processed
    live.n_prompt_total = snap.n_prompt_total
    live.n_prompt_cache = snap.n_prompt_cache
    live.n_decoded = snap.n_decoded


def sample_tps_for_history(live: LiveThroughput) -> float:
    """Pick the generation tok/s sample to record for sparkline / rolling average.

    Prefill is a different unit of work and must not share this scale.
    """
    if live.stage != "generating":
        return 0.0
    if live.source not in ("slots", "metrics_delta"):
        return 0.0
    return live.gen_tps


def compute_live_tps(
    metrics: Metrics | None,
    slots: list | None,
    *,
    last_slot_snap: SlotSnapshot | None = None,
    last_slot_t: float | None = None,
    now: float | None = None,
) -> tuple[LiveThroughput, SlotSnapshot | None, float | None]:
    """Compute live tok/s from metrics and optional slot progress snapshots."""
    if not metrics:
        return LiveThroughput(), None, None

    tick = time.monotonic() if now is None else now
    slot_snap = summarize_slots(slots)
    direct_gen, direct_prompt = _direct_slot_rates(slots)

    slot_gen_tps = direct_gen
    slot_prompt_tps = direct_prompt

    if (
        slot_snap is not None
        and last_slot_snap is not None
        and last_slot_t is not None
        and slot_snap.is_processing
    ):
        dt = tick - last_slot_t
        if dt > 0:
            if slot_gen_tps <= 0:
                d_decoded = slot_snap.n_decoded - last_slot_snap.n_decoded
                if d_decoded > 0:
                    slot_gen_tps = d_decoded / dt
            if slot_prompt_tps <= 0:
                d_prompt = slot_snap.n_prompt_processed - last_slot_snap.n_prompt_processed
                if d_prompt > 0:
                    slot_prompt_tps = d_prompt / dt

    metrics_gen, metrics_gen_src = _metrics_rate(
        metrics.requests_processing,
        metrics.gen_tps_derived,
        metrics.predicted_tokens_seconds,
    )
    metrics_prompt, metrics_prompt_src = _metrics_rate(
        metrics.requests_processing,
        metrics.prompt_tps_derived,
        metrics.prompt_tokens_seconds,
    )

    gen_tps, gen_src = _pick_rate(slot_gen_tps, metrics_gen, metrics_gen_src)
    prompt_tps, prompt_src = _pick_rate(slot_prompt_tps, metrics_prompt, metrics_prompt_src)
    stage = _infer_stage(
        gen_tps,
        prompt_tps,
        metrics.requests_processing,
        metrics.requests_deferred,
        slot_snap,
    )

    source: ThroughputSource = gen_src
    if prompt_tps > gen_tps and prompt_src != "idle":
        source = prompt_src

    live = LiveThroughput(
        gen_tps=gen_tps,
        prompt_tps=prompt_tps,
        source=source,
        stage=stage,
        requests_deferred=int(metrics.requests_deferred),
    )
    _copy_snap(live, slot_snap)

    next_snap = slot_snap if slot_snap is not None else last_slot_snap
    next_t = tick if slot_snap is not None else last_slot_t
    return live, next_snap, next_t


def live_tps_from_metrics(metrics: Metrics) -> LiveThroughput:
    """Metrics-only fallback when slots are unavailable."""
    live, _, _ = compute_live_tps(metrics, None)
    return live


class ThroughputReader:
    """Stateful reader that tracks slot snapshots and stage stability."""

    # Hold last active stage briefly to avoid idle flicker between poll windows.
    PHASE_HOLD_S = 1.5

    def __init__(self) -> None:
        self._last_slot_snap: SlotSnapshot | None = None
        self._last_slot_t: float | None = None
        self._held_stage: RequestStage = "idle"
        self._last_active_t: float | None = None
        self._in_flight: LastRequest | None = None
        self._last_request: LastRequest | None = None

    def clear(self) -> None:
        self._last_slot_snap = None
        self._last_slot_t = None
        self._held_stage = "idle"
        self._last_active_t = None
        self._in_flight = None
        self._last_request = None

    def _stabilize_stage(self, live: LiveThroughput, slots: list | None) -> LiveThroughput:
        now = time.monotonic()
        slot_snap = summarize_slots(slots)
        busy = live.stage in _ACTIVE_STAGES or (
            slot_snap is not None and slot_snap.is_processing
        )

        if live.stage in _ACTIVE_STAGES:
            self._held_stage = live.stage
            self._last_active_t = now
            return live

        if (
            self._held_stage in _ACTIVE_STAGES
            and self._last_active_t is not None
            and (now - self._last_active_t) < self.PHASE_HOLD_S
        ):
            live.stage = self._held_stage
            _copy_snap(live, self._last_slot_snap)
            return live

        if not busy:
            self._held_stage = live.stage
            self._last_active_t = None
        return live

    def _track_request(self, live: LiveThroughput) -> None:
        if live.stage in _ACTIVE_STAGES:
            gen_tps = live.gen_tps if live.stage == "generating" and live.gen_tps > 0 else 0.0
            if self._in_flight is not None and gen_tps <= 0:
                gen_tps = self._in_flight.gen_tps
            self._in_flight = LastRequest(
                prompt_tokens=max(live.n_prompt_total, live.n_prompt_processed),
                cache_tokens=live.n_prompt_cache,
                gen_tokens=live.n_decoded,
                gen_tps=gen_tps,
            )
            return
        if self._in_flight is not None:
            self._last_request = self._in_flight
            self._in_flight = None

    def update(self, metrics: Metrics | None, slots: list | None) -> LiveThroughput:
        live, self._last_slot_snap, self._last_slot_t = compute_live_tps(
            metrics,
            slots,
            last_slot_snap=self._last_slot_snap,
            last_slot_t=self._last_slot_t,
        )
        live = self._stabilize_stage(live, slots)
        self._track_request(live)
        live.last_request = self._last_request
        return live


class ThroughputHistory:
    """Fixed-size rolling window of generation tok/s samples."""

    def __init__(self, max_samples: int = 120) -> None:
        self._samples: deque[float] = deque(maxlen=max_samples)

    def push(self, tps: float) -> None:
        self._samples.append(max(0.0, tps))

    def clear(self) -> None:
        self._samples.clear()

    @property
    def samples(self) -> list[float]:
        return list(self._samples)

    @property
    def average(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)


def format_avg_line(samples: list[float], poll_interval: float) -> Text | None:
    """Format the rolling average line for the status panel."""
    if not samples or not any(s > 0 for s in samples):
        return None
    avg = sum(samples) / len(samples)
    window_s = len(samples) * poll_interval
    line = Text()
    line.append(f"avg {avg:.1f} tok/s", style="bold cyan")
    line.append(f"  ({window_s:.0f}s rolling)", style="dim")
    return line


def fmt_compact_tps(tps: float) -> str:
    """Format a rate, using k for prefill-scale numbers."""
    if tps >= 1000:
        return f"{tps / 1000:.1f}k t/s"
    return f"{tps:.1f} t/s"


def format_last_request(last: LastRequest | None) -> Text | None:
    """One-line recap of the request that just finished."""
    if last is None or (last.prompt_tokens <= 0 and last.gen_tokens <= 0):
        return None
    line = Text()
    line.append("last: ", style="dim")
    line.append(f"{last.prompt_tokens:,} prompt", style="dim")
    if last.cache_tokens > 0:
        line.append(f" ({last.cache_tokens:,} cache)", style="dim")
    line.append(" · ", style="dim")
    line.append(f"{last.gen_tokens:,} gen", style="dim")
    if last.gen_tps > 0:
        line.append(f" @ {last.gen_tps:.0f} t/s", style="dim")
    return line


def _stage_lit(live: LiveThroughput, key: str) -> bool:
    if key == "idle":
        return live.stage == "idle" and live.requests_deferred <= 0
    if key == "queued":
        return live.requests_deferred > 0 or live.stage == "queued"
    if key == "cache":
        return live.n_prompt_cache > 0 and live.stage == "prefill"
    if key == "prefill":
        return live.stage == "prefill"
    if key == "generating":
        return live.stage == "generating"
    return False


def _stage_style(key: str) -> str:
    if key == "queued":
        return "bold yellow"
    if key == "cache":
        return "bold cyan"
    if key == "prefill":
        return "bold magenta"
    if key == "generating":
        return "bold green"
    return "bold"


def render_stage_strip(live: LiveThroughput) -> Text:
    """Always-visible request pipeline: idle queue cache prefill generate."""
    line = Text(no_wrap=True)
    for index, (label, key) in enumerate(_STAGE_STRIP):
        if index:
            line.append("  ", style="dim")
        if _stage_lit(live, key):
            line.append(label.upper(), style=_stage_style(key))
        else:
            line.append(label, style="dim")
    return line


def _prefill_done(live: LiveThroughput) -> int:
    done = live.n_prompt_cache + live.n_prompt_processed
    if live.n_prompt_total > 0:
        return min(done, live.n_prompt_total)
    return done


def render_prefill_progress(live: LiveThroughput, width: int = PREFILL_BAR_WIDTH) -> Text:
    """Progress bar of prompt tokens read, with cache called out."""
    if width <= 0:
        width = PREFILL_BAR_WIDTH

    total = live.n_prompt_total
    done = _prefill_done(live)
    filled = width if total <= 0 and done > 0 else 0
    if total > 0:
        filled = min(width, round(width * done / total))

    line = Text(no_wrap=True)
    line.append(_PREFILL_BLOCKS[1] * filled, style="bold magenta")
    line.append(_PREFILL_BLOCKS[0] * (width - filled), style="dim")
    if total > 0:
        line.append(f"  {done:,}/{total:,}", style="bold cyan")
    elif done > 0:
        line.append(f"  {done:,}", style="bold cyan")
    if live.prompt_tps > 0:
        line.append(f"  {fmt_compact_tps(live.prompt_tps)}", style="dim")
    if live.n_prompt_cache > 0:
        line.append(f"  cache {live.n_prompt_cache:,}", style="cyan")
    return line


def _bar_style(tps: float) -> str:
    if tps <= 0:
        return "dim"
    if tps >= 20:
        return "bold green"
    if tps >= 5:
        return "yellow"
    return "red"


def render_tps_sparkline(
    samples: list[float],
    width: int = SPARKLINE_WIDTH,
) -> Text:
    """Render recent tok/s as a fixed-width, right-aligned sparkline.

    New samples appear on the right; once full, older samples scroll off the left
    (btop-style). Width is always exactly ``width`` characters so the line never
    wraps or shifts the layout.
    """
    if width <= 0:
        width = SPARKLINE_WIDTH

    if not samples:
        return Text(" " * width, style="dim", no_wrap=True)

    view = samples[-width:]
    scale = max(max(view), 1.0)
    levels = len(_SPARK_BLOCKS) - 1

    line = Text(no_wrap=True)
    for _ in range(width - len(view)):
        line.append(" ", style="dim")
    for tps in view:
        if tps <= 0:
            line.append(" ", style="dim")
            continue
        level = min(levels, round((tps / scale) * levels))
        line.append(_SPARK_BLOCKS[level], style=_bar_style(tps))
    return line
