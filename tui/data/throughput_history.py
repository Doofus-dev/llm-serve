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

# Fixed display width — must fit the THROUGHPUT column without wrapping.
SPARKLINE_WIDTH = 40

ThroughputSource = Literal["slots", "metrics_delta", "metrics_gauge", "idle"]
ThroughputPhase = Literal["generating", "prompt", "idle"]


@dataclass
class SlotSnapshot:
    n_decoded: int = 0
    n_prompt_processed: int = 0
    n_prompt_total: int = 0
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
class LiveThroughput:
    gen_tps: float = 0.0
    prompt_tps: float = 0.0
    source: ThroughputSource = "idle"
    phase: ThroughputPhase = "idle"


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


def _infer_phase(
    gen_tps: float,
    prompt_tps: float,
    requests_processing: float,
    slot_snap: SlotSnapshot | None,
) -> ThroughputPhase:
    active = requests_processing > 0 or (slot_snap is not None and slot_snap.is_processing)
    if not active:
        return "idle"
    if slot_snap is not None and slot_snap.in_prefill:
        return "prompt"
    if gen_tps > 0:
        return "generating"
    if prompt_tps > 0:
        return "prompt"
    if active:
        # Slot is busy but rates are between poll windows — infer from progress.
        if slot_snap is not None and slot_snap.n_decoded == 0:
            return "prompt"
        return "generating"
    return "idle"


def sample_tps_for_history(live: LiveThroughput) -> float:
    """Pick the throughput sample to record for sparkline / rolling average."""
    if live.phase == "idle":
        return 0.0
    if live.source not in ("slots", "metrics_delta"):
        return 0.0
    if live.phase == "prompt":
        return live.prompt_tps
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
    phase = _infer_phase(gen_tps, prompt_tps, metrics.requests_processing, slot_snap)

    source: ThroughputSource = gen_src
    if prompt_tps > gen_tps and prompt_src != "idle":
        source = prompt_src

    live = LiveThroughput(
        gen_tps=gen_tps,
        prompt_tps=prompt_tps,
        source=source,
        phase=phase,
    )

    next_snap = slot_snap if slot_snap is not None else last_slot_snap
    next_t = tick if slot_snap is not None else last_slot_t
    return live, next_snap, next_t


def live_tps_from_metrics(metrics: Metrics) -> LiveThroughput:
    """Metrics-only fallback when slots are unavailable."""
    live, _, _ = compute_live_tps(metrics, None)
    return live


class ThroughputReader:
    """Stateful reader that tracks slot snapshots and phase stability."""

    # Hold last phase briefly to avoid idle flicker between poll windows.
    PHASE_HOLD_S = 1.5

    def __init__(self) -> None:
        self._last_slot_snap: SlotSnapshot | None = None
        self._last_slot_t: float | None = None
        self._held_phase: ThroughputPhase = "idle"
        self._last_active_t: float | None = None

    def clear(self) -> None:
        self._last_slot_snap = None
        self._last_slot_t = None
        self._held_phase = "idle"
        self._last_active_t = None

    def _stabilize_phase(self, live: LiveThroughput, slots: list | None) -> LiveThroughput:
        now = time.monotonic()
        slot_snap = summarize_slots(slots)
        busy = (
            live.phase != "idle"
            or (slot_snap is not None and slot_snap.is_processing)
        )

        if busy and live.phase != "idle":
            self._held_phase = live.phase
            self._last_active_t = now
            return live

        if (
            self._held_phase != "idle"
            and self._last_active_t is not None
            and (now - self._last_active_t) < self.PHASE_HOLD_S
        ):
            live.phase = self._held_phase
            return live

        if not busy:
            self._held_phase = "idle"
            self._last_active_t = None
        return live

    def update(self, metrics: Metrics | None, slots: list | None) -> LiveThroughput:
        live, self._last_slot_snap, self._last_slot_t = compute_live_tps(
            metrics,
            slots,
            last_slot_snap=self._last_slot_snap,
            last_slot_t=self._last_slot_t,
        )
        return self._stabilize_phase(live, slots)


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
    def rolling_average(self) -> float | None:
        """Mean of all samples in the rolling window (zeros lower the average over time)."""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    @property
    def has_activity(self) -> bool:
        return any(s > 0 for s in self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)


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
