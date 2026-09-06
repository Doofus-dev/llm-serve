"""On-machine measurements from actually running models.

Estimates in the Hub are heuristics. This file stores what the TUI observed
while llama-server was up — VRAM in use and the best rolling generation
average — so Hub columns can show estimated vs actual for this GPU even
after the server has been down.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from tui.data.gpu import gpu_match_key, same_gpu

MAX_RUNS = 200
VRAM_DELTA_MB = 16.0
TPS_DELTA = 0.5
MIN_VRAM_MB = 256.0
MIN_VRAM_FILE_RATIO = 0.10


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


def _filename(path: str) -> str:
    return Path(path).name.lower()


def plausible_vram(vram_used_mb: float, file_size: int = 0) -> bool:
    """Reject idle desktop VRAM and pre-load readings."""
    if vram_used_mb < MIN_VRAM_MB:
        return False
    if file_size > 0:
        file_mb = file_size / 1_000_000
        if vram_used_mb < file_mb * MIN_VRAM_FILE_RATIO:
            return False
    return True


def _identity(run: RunBaseline) -> tuple:
    return (
        run.model,
        _filename(run.file),
        gpu_match_key(run.gpu_name) or " ".join(run.gpu_name.lower().split()),
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
    """Upsert a live observation. Returns True if the file changed.

    Generation / prompt tok/s are high-water marks of recorded averages:
    a slower later sample never replaces a faster one, so Hub still shows
    the best seen speed after the server has been idle or down.
    """
    vram_ok = plausible_vram(observation.vram_used_mb, observation.file_size)
    has_speed = bool(observation.gen_tps or observation.prompt_tps)
    if not vram_ok and not has_speed:
        return False

    runs = load_baselines(path)
    key = _identity(observation)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for existing in runs:
        if _identity(existing) != key:
            continue
        changed = False
        if vram_ok and abs(existing.vram_used_mb - observation.vram_used_mb) >= VRAM_DELTA_MB:
            existing.vram_used_mb = observation.vram_used_mb
            changed = True
        if observation.gen_tps and (
            existing.gen_tps is None or observation.gen_tps > existing.gen_tps + TPS_DELTA
        ):
            existing.gen_tps = observation.gen_tps
            existing.tokens_predicted = observation.tokens_predicted
            changed = True
        if observation.prompt_tps and (
            existing.prompt_tps is None
            or observation.prompt_tps > existing.prompt_tps + TPS_DELTA
        ):
            existing.prompt_tps = observation.prompt_tps
            changed = True
        if changed:
            existing.updated_at = now
            save_baselines(path, runs)
        return changed

    if not vram_ok:
        return False
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
    """Best measurement for a file on this GPU. Closer context scores higher."""
    want_name = _filename(filename)
    best: RunBaseline | None = None
    best_score = -1.0
    for run in runs:
        if gpu_name and run.gpu_name and not same_gpu(run.gpu_name, gpu_name):
            continue
        if abs(run.offload_ratio - offload_ratio) > 0.2:
            continue
        run_name = _filename(run.file)
        name_match = bool(want_name) and run_name == want_name
        size_match = (
            file_size > 0
            and run.file_size > 0
            and abs(run.file_size - file_size) / max(file_size, run.file_size) <= 0.08
        )
        if want_name and run_name:
            if not name_match:
                continue
        elif not size_match:
            continue
        ctx_span = max(ctx, run.ctx, 1)
        score = 2.0 if name_match else 1.0
        score -= abs(run.offload_ratio - offload_ratio)
        score -= abs(run.ctx - ctx) / ctx_span
        if run.gen_tps:
            score += 2.0
        if plausible_vram(run.vram_used_mb, run.file_size):
            score += 1.0
        if score > best_score:
            best = run
            best_score = score
    return best


def latest_baseline_ctx(
    runs: list[RunBaseline],
    *,
    filenames: set[str] | list[str],
    gpu_name: str,
) -> int | None:
    """Most recent measured ctx for any of these files on this GPU."""
    want = {_filename(name) for name in filenames if name}
    if not want:
        return None
    best: RunBaseline | None = None
    for run in runs:
        if _filename(run.file) not in want:
            continue
        if gpu_name and run.gpu_name and not same_gpu(run.gpu_name, gpu_name):
            continue
        if not plausible_vram(run.vram_used_mb, run.file_size) and not run.gen_tps:
            continue
        if best is None or run.updated_at > best.updated_at:
            best = run
    return best.ctx if best and best.ctx > 0 else None
