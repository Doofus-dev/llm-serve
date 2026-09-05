"""On-machine measurements from actually running models.

Estimates in the Hub are heuristics. This file stores what the TUI observed
while llama-server was up — VRAM in use and tok/s after real generation —
so Hub columns can show estimated vs actual for this GPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MAX_RUNS = 200
VRAM_DELTA_MB = 16.0
TPS_DELTA = 0.5


@dataclass
class RunBaseline:
    model: str
    file: str
    file_size: int
    gpu_name: str
    ctx: int
    gpu_layers: int
    total_layers: int
    cache_k: str
    cache_v: str
    vram_used_mb: float
    gen_tps: float | None = None
    prompt_tps: float | None = None
    tokens_predicted: float = 0.0
    updated_at: str = ""

    @property
    def offload_ratio(self) -> float:
        if self.gpu_layers <= 0:
            return 0.0
        if self.gpu_layers >= 99 or (
            self.total_layers > 0 and self.gpu_layers >= self.total_layers
        ):
            return 1.0
        if self.total_layers <= 0:
            return 1.0
        return max(0.0, min(1.0, self.gpu_layers / self.total_layers))

    @classmethod
    def from_dict(cls, data: dict) -> RunBaseline:
        return cls(
            model=str(data.get("model", "")),
            file=str(data.get("file", "")),
            file_size=int(data.get("file_size") or 0),
            gpu_name=str(data.get("gpu_name", "")),
            ctx=int(data.get("ctx") or 0),
            gpu_layers=int(data.get("gpu_layers") or 0),
            total_layers=int(data.get("total_layers") or 0),
            cache_k=str(data.get("cache_k") or ""),
            cache_v=str(data.get("cache_v") or ""),
            vram_used_mb=float(data.get("vram_used_mb") or 0.0),
            gen_tps=_optional_float(data.get("gen_tps")),
            prompt_tps=_optional_float(data.get("prompt_tps")),
            tokens_predicted=float(data.get("tokens_predicted") or 0.0),
            updated_at=str(data.get("updated_at") or ""),
        )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _norm_gpu(name: str) -> str:
    return " ".join(name.lower().split())


def _filename(path: str) -> str:
    return Path(path).name.lower()


def _identity(run: RunBaseline) -> tuple:
    return (
        run.model,
        _filename(run.file),
        _norm_gpu(run.gpu_name),
        int(run.ctx),
        int(run.gpu_layers),
        run.cache_k,
        run.cache_v,
    )


def load_baselines(path: Path) -> list[RunBaseline]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("runs", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return [RunBaseline.from_dict(row) for row in rows if isinstance(row, dict)]


def save_baselines(path: Path, runs: list[RunBaseline]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = sorted(runs, key=lambda run: run.updated_at, reverse=True)[:MAX_RUNS]
    payload = {"runs": [asdict(run) for run in trimmed]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def record_baseline(path: Path, observation: RunBaseline) -> bool:
    """Upsert a live observation. Returns True if the file changed."""
    runs = load_baselines(path)
    key = _identity(observation)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for existing in runs:
        if _identity(existing) != key:
            continue
        changed = False
        if abs(existing.vram_used_mb - observation.vram_used_mb) >= VRAM_DELTA_MB:
            existing.vram_used_mb = observation.vram_used_mb
            changed = True
        if observation.gen_tps and (
            existing.gen_tps is None
            or observation.tokens_predicted >= existing.tokens_predicted
        ):
            if (
                existing.gen_tps is None
                or abs(existing.gen_tps - observation.gen_tps) >= TPS_DELTA
                or observation.tokens_predicted > existing.tokens_predicted
            ):
                existing.gen_tps = observation.gen_tps
                existing.tokens_predicted = observation.tokens_predicted
                changed = True
        if observation.prompt_tps and (
            existing.prompt_tps is None
            or abs(existing.prompt_tps - observation.prompt_tps) >= TPS_DELTA
        ):
            existing.prompt_tps = observation.prompt_tps
            changed = True
        if changed:
            existing.updated_at = now
            save_baselines(path, runs)
        return changed

    observation.updated_at = now
    runs.append(observation)
    save_baselines(path, runs)
    return True


def lookup_baseline(
    runs: list[RunBaseline],
    *,
    filename: str,
    file_size: int,
    ctx: int,
    offload_ratio: float,
    gpu_name: str,
) -> RunBaseline | None:
    """Best measurement for a Hub row on this GPU / context / offload."""
    want_name = _filename(filename)
    want_gpu = _norm_gpu(gpu_name)
    best: RunBaseline | None = None
    best_score = -1.0
    for run in runs:
        if want_gpu and _norm_gpu(run.gpu_name) != want_gpu:
            continue
        if run.ctx != ctx:
            continue
        if abs(run.offload_ratio - offload_ratio) > 0.2:
            continue
        name_match = _filename(run.file) == want_name and bool(want_name)
        size_match = (
            file_size > 0
            and run.file_size > 0
            and abs(run.file_size - file_size) / max(file_size, run.file_size) <= 0.08
        )
        if not name_match and not size_match:
            continue
        score = 2.0 if name_match else 1.0
        score -= abs(run.offload_ratio - offload_ratio)
        if score > best_score:
            best = run
            best_score = score
    return best
