"""Poll llama-server endpoints: /health, /metrics (Prometheus), /slots, /props."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx


@dataclass
class Metrics:
    prompt_tokens_total: float = 0.0
    tokens_predicted_total: float = 0.0
    prompt_tokens_seconds: float = 0.0      # current prompt tok/s
    predicted_tokens_seconds: float = 0.0   # current gen tok/s
    prompt_seconds_total: float = 0.0
    tokens_predicted_seconds_total: float = 0.0
    n_decode_total: float = 0.0
    requests_processing: float = 0.0
    requests_deferred: float = 0.0
    # Derived rates from counter deltas (fallback when current gauges lag)
    gen_tps_derived: float = 0.0
    prompt_tps_derived: float = 0.0

    @property
    def avg_gen_tps(self) -> float:
        if self.tokens_predicted_seconds_total > 0:
            return self.tokens_predicted_total / self.tokens_predicted_seconds_total
        return 0.0

    @property
    def avg_prompt_tps(self) -> float:
        if self.prompt_seconds_total > 0:
            return self.prompt_tokens_total / self.prompt_seconds_total
        return 0.0

    @property
    def ms_per_token(self) -> float:
        if self.predicted_tokens_seconds > 0:
            return 1000.0 / self.predicted_tokens_seconds
        return 0.0


def parse_prometheus(text: str) -> Metrics:
    m = Metrics()
    mapping = {
        "llamacpp:prompt_tokens_total": "prompt_tokens_total",
        "llamacpp:tokens_predicted_total": "tokens_predicted_total",
        "llamacpp:prompt_tokens_seconds": "prompt_tokens_seconds",
        "llamacpp:predicted_tokens_seconds": "predicted_tokens_seconds",
        "llamacpp:prompt_seconds_total": "prompt_seconds_total",
        "llamacpp:tokens_predicted_seconds_total": "tokens_predicted_seconds_total",
        "llamacpp:n_decode_total": "n_decode_total",
        "llamacpp:requests_processing": "requests_processing",
        "llamacpp:requests_deferred": "requests_deferred",
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        if key in mapping:
            try:
                setattr(m, mapping[key], float(parts[1]))
            except ValueError:
                pass
    return m


class ServerClient:
    def __init__(self, host: str, port: int, timeout: float = 3.0):
        self.base = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(timeout=timeout)
        self._last: tuple[Metrics, float] | None = None

    async def close(self):
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self.base}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def metrics(self) -> Metrics | None:
        try:
            r = await self._client.get(f"{self.base}/metrics")
            if r.status_code == 200:
                m = parse_prometheus(r.text)
                # Rate fallback: current gauges are 0 when idle and update only
                # at generation boundaries. Derive from counter deltas instead.
                now = time.monotonic()
                if self._last is not None:
                    last, last_t = self._last
                    dt = now - last_t
                    if dt > 0:
                        m.gen_tps_derived = max(
                            0.0, (m.tokens_predicted_total - last.tokens_predicted_total) / dt)
                        m.prompt_tps_derived = max(
                            0.0, (m.prompt_tokens_total - last.prompt_tokens_total) / dt)
                self._last = (m, now)
                return m
        except httpx.HTTPError:
            pass
        return None

    async def props(self) -> dict | None:
        try:
            r = await self._client.get(f"{self.base}/props")
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        return None

    async def slots(self) -> list | None:
        try:
            r = await self._client.get(f"{self.base}/slots")
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        return None
