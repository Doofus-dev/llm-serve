"""llama-bench runs stored separately from live-server throughput baselines.

``tui-benchmarks.json`` records offline ``llama-bench`` speed for a local
GGUF on a machine fingerprint (hostname, GPU, VRAM, CPU). Re-running
replaces this host's row for the same model+quant+config; other hosts
keep their rows. Live STATUS t/s still lives in ``tui-baselines.json``.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from tui.data.fsutil import atomic_write
from tui.data.gpu import GPUStats, query_gpu
from tui.data.models_json import Registry, quant_rel_paths
from tui.data.quant import is_sidecar_gguf, quant_from_filename

DEFAULT_N_PROMPT = 512
DEFAULT_N_GEN = 128
DEFAULT_REPETITIONS = 5
DEFAULT_N_DEPTH = 0

_TS_CELL = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)\s*([0-9]+(?:\.[0-9]+)?)"
)
_TEST_CELL = re.compile(r"(pp|tg)\s*(\d+)", re.IGNORECASE)


class LlamaBenchError(Exception):
    """llama-bench could not be run or its output could not be parsed."""


class LlamaBenchMissing(LlamaBenchError):
    """The vendored llama-bench binary is missing or not executable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        super().__init__(f"llama-bench not found at {self.path}")


@dataclass(frozen=True)
class BenchConfig:
    n_prompt: int = DEFAULT_N_PROMPT
    n_gen: int = DEFAULT_N_GEN
    repetitions: int = DEFAULT_REPETITIONS
    n_depth: int = DEFAULT_N_DEPTH

    def identity(self) -> tuple[int, int, int, int]:
        return (self.n_prompt, self.n_gen, self.repetitions, self.n_depth)


@dataclass(frozen=True)
class HardwareFingerprint:
    hostname: str
    gpu_name: str = ""
    vram_total_mb: float = 0.0
    cpu_model: str = ""

    def label(self) -> str:
        bits = [self.hostname or "unknown-host"]
        if self.gpu_name:
            bits.append(self.gpu_name)
        if self.vram_total_mb > 0:
            bits.append(_fmt_vram(self.vram_total_mb))
        if self.cpu_model:
            bits.append(self.cpu_model)
        return " · ".join(bits)


@dataclass(frozen=True)
class BenchTarget:
    model: str
    display: str
    quant: str
    file_rel: str
    paths: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.model}/{self.quant}"


@dataclass
class LlamaBenchMeasurement:
    """Speeds and backend metadata parsed from one llama-bench invocation."""

    pp_tps: float | None = None
    pp_stddev: float | None = None
    tg_tps: float | None = None
    tg_stddev: float | None = None
    n_gpu_layers: int | None = None
    n_threads: int | None = None
    backend: str = ""
    gpu_info: str = ""
    cpu_info: str = ""
    n_prompt: int = DEFAULT_N_PROMPT
    n_gen: int = DEFAULT_N_GEN
    n_depth: int = DEFAULT_N_DEPTH
    timestamp: str = ""
    model_filename: str = ""
    model_type: str = ""


@dataclass
class BenchRow:
    hostname: str
    model: str
    quant: str
    file: str = ""
    gpu_name: str = ""
    vram_total_mb: float = 0.0
    cpu_model: str = ""
    n_prompt: int = DEFAULT_N_PROMPT
    n_gen: int = DEFAULT_N_GEN
    repetitions: int = DEFAULT_REPETITIONS
    n_depth: int = DEFAULT_N_DEPTH
    pp_tps: float | None = None
    pp_stddev: float | None = None
    tg_tps: float | None = None
    tg_stddev: float | None = None
    n_gpu_layers: int | None = None
    n_threads: int | None = None
    backend: str = ""
    updated_at: str = ""
    display: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> BenchRow:
        return cls(
            hostname=str(data.get("hostname") or ""),
            model=str(data.get("model") or ""),
            quant=str(data.get("quant") or ""),
            file=str(data.get("file") or ""),
            gpu_name=str(data.get("gpu_name") or ""),
            vram_total_mb=_optional_float(data.get("vram_total_mb")) or 0.0,
            cpu_model=str(data.get("cpu_model") or ""),
            n_prompt=int(data.get("n_prompt") or DEFAULT_N_PROMPT),
            n_gen=int(data.get("n_gen") or DEFAULT_N_GEN),
            repetitions=int(data.get("repetitions") or DEFAULT_REPETITIONS),
            n_depth=int(data.get("n_depth") or 0),
            pp_tps=_optional_float(data.get("pp_tps")),
            pp_stddev=_optional_float(data.get("pp_stddev")),
            tg_tps=_optional_float(data.get("tg_tps")),
            tg_stddev=_optional_float(data.get("tg_stddev")),
            n_gpu_layers=_optional_int(data.get("n_gpu_layers")),
            n_threads=_optional_int(data.get("n_threads")),
            backend=str(data.get("backend") or ""),
            updated_at=str(data.get("updated_at") or ""),
            display=str(data.get("display") or ""),
        )

    def config(self) -> BenchConfig:
        return BenchConfig(
            n_prompt=self.n_prompt,
            n_gen=self.n_gen,
            repetitions=self.repetitions,
            n_depth=self.n_depth,
        )

    def fingerprint(self) -> HardwareFingerprint:
        return HardwareFingerprint(
            hostname=self.hostname,
            gpu_name=self.gpu_name,
            vram_total_mb=self.vram_total_mb,
            cpu_model=self.cpu_model,
        )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _fmt_vram(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cpu_model_name(cpuinfo: str | None = None) -> str:
    """Best-effort CPU marketing name from /proc/cpuinfo."""
    text = cpuinfo
    if text is None:
        try:
            text = Path("/proc/cpuinfo").read_text()
        except OSError:
            return ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def collect_fingerprint(
    gpu: GPUStats | None = None,
    *,
    hostname: str | None = None,
    cpu_model: str | None = None,
) -> HardwareFingerprint:
    stats = gpu if gpu is not None else query_gpu()
    return HardwareFingerprint(
        hostname=hostname if hostname is not None else socket.gethostname(),
        gpu_name=stats.name,
        vram_total_mb=stats.vram_total_mb,
        cpu_model=cpu_model if cpu_model is not None else cpu_model_name(),
    )


def catalogued_ggufs(registry: Registry, models_dir: Path) -> list[BenchTarget]:
    """Local GGUFs already in models.json that exist on disk."""
    targets: list[BenchTarget] = []
    root = Path(models_dir)
    for name, cfg in registry.models.items():
        quants = cfg.params.get("quants")
        quant_ids: list[str] = []
        if isinstance(quants, dict) and quants:
            quant_ids = [str(qid) for qid in quants]
        elif cfg.active_quant:
            quant_ids = [cfg.active_quant]
        for quant_id in quant_ids:
            rels = quant_rel_paths(cfg, quant_id)
            if not rels:
                continue
            paths = [root / rel for rel in rels]
            if any(is_sidecar_gguf(path.name) for path in paths):
                continue
            if not all(path.is_file() for path in paths):
                continue
            targets.append(
                BenchTarget(
                    model=name,
                    display=cfg.display,
                    quant=quant_id,
                    file_rel=rels[0],
                    paths=tuple(paths),
                )
            )
    return targets


def _row_identity(row: BenchRow) -> tuple:
    return (
        row.hostname,
        row.model,
        row.quant,
        int(row.n_prompt),
        int(row.n_gen),
        int(row.repetitions),
        int(row.n_depth),
    )


def load_benchmarks(path: Path) -> list[BenchRow]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = data.get("runs", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    return [BenchRow.from_dict(row) for row in rows if isinstance(row, dict)]


def save_benchmarks(path: Path, rows: list[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": [asdict(row) for row in rows]}
    atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def record_benchmark(path: Path, row: BenchRow) -> bool:
    """Upsert by host+model+quant+Prompt/Gen/Repeats. True if a row was replaced."""
    if not row.updated_at:
        row.updated_at = utc_now()
    runs = load_benchmarks(path)
    key = _row_identity(row)
    for index, existing in enumerate(runs):
        if _row_identity(existing) == key:
            runs[index] = row
            save_benchmarks(path, runs)
            return True
    runs.append(row)
    save_benchmarks(path, runs)
    return False


def _speed_sort_key(value: float | None) -> float:
    # Missing speeds sink; higher tok/s first via negative sort.
    if value is None:
        return float("inf")
    return -float(value)


def rows_for_table(
    rows: Sequence[BenchRow],
    *,
    this_hostname: str,
    host_filter: str = "",
    current: BenchConfig | None = None,
) -> list[BenchRow]:
    """Matching knobs first, then fastest Gen t/s. Same identity stays comparable."""
    filtered = [
        row
        for row in rows
        if not host_filter or row.hostname == host_filter
    ]
    return sorted(
        filtered,
        key=lambda row: (
            0 if current is not None and row.config() == current else 1,
            -int(row.n_prompt),
            -int(row.n_gen),
            -int(row.repetitions),
            _speed_sort_key(row.tg_tps),
            _speed_sort_key(row.pp_tps),
            (row.display or row.model).lower(),
            row.model.lower(),
            row.quant.lower(),
            0 if row.hostname == this_hostname else 1,
            row.hostname.lower(),
            row.updated_at,
        ),
    )


def hostnames_in(rows: Sequence[BenchRow]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row.hostname and row.hostname not in seen:
            seen.append(row.hostname)
    return seen


def format_speed(avg: float | None, stddev: float | None = None) -> str:
    if avg is None:
        return "—"
    if stddev is None:
        return f"{avg:.2f}"
    return f"{avg:.2f} ± {stddev:.2f}"


def _extract_json_array(text: str) -> list | None:
    start = text.find("[")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _merge_json_records(records: list[dict]) -> LlamaBenchMeasurement:
    measurement = LlamaBenchMeasurement()
    for raw in records:
        if not isinstance(raw, dict):
            continue
        n_prompt = int(raw.get("n_prompt") or 0)
        n_gen = int(raw.get("n_gen") or 0)
        n_depth = int(raw.get("n_depth") or 0)
        avg = _optional_float(raw.get("avg_ts"))
        std = _optional_float(raw.get("stddev_ts"))
        if n_prompt > 0 and n_gen == 0:
            measurement.pp_tps = avg
            measurement.pp_stddev = std
            measurement.n_prompt = n_prompt
        elif n_gen > 0:
            measurement.tg_tps = avg
            measurement.tg_stddev = std
            measurement.n_gen = n_gen
        if n_depth:
            measurement.n_depth = n_depth
        measurement.n_gpu_layers = _optional_int(raw.get("n_gpu_layers"))
        measurement.n_threads = _optional_int(raw.get("n_threads"))
        measurement.backend = str(raw.get("backends") or measurement.backend)
        measurement.gpu_info = str(raw.get("gpu_info") or measurement.gpu_info)
        measurement.cpu_info = str(raw.get("cpu_info") or measurement.cpu_info)
        measurement.model_filename = str(
            raw.get("model_filename") or measurement.model_filename
        )
        measurement.model_type = str(raw.get("model_type") or measurement.model_type)
        stamp = str(raw.get("test_time") or "")
        if stamp:
            measurement.timestamp = stamp
    if not measurement.timestamp:
        measurement.timestamp = utc_now()
    return measurement


def parse_llama_bench_json(text: str) -> LlamaBenchMeasurement:
    records = _extract_json_array(text)
    if not records:
        raise LlamaBenchError("llama-bench JSON output was empty or invalid")
    measurement = _merge_json_records(records)
    if measurement.pp_tps is None and measurement.tg_tps is None:
        raise LlamaBenchError("llama-bench JSON had no pp/tg speed rows")
    return measurement


def _header_index(header: list[str], *names: str) -> int | None:
    want = {name.lower() for name in names}
    for index, cell in enumerate(header):
        if cell.lower().strip() in want:
            return index
    return None


def parse_llama_bench_markdown(text: str) -> LlamaBenchMeasurement:
    rows: list[list[str]] = []
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if header is None:
            if any(cell.lower() == "test" for cell in cells) or any(
                "t/s" in cell.lower() for cell in cells
            ):
                header = [cell.lower() for cell in cells]
            continue
        if all(set(cell) <= set("-: ") and cell for cell in cells):
            continue
        rows.append(cells)
    if not header or not rows:
        raise LlamaBenchError("llama-bench markdown table was empty or invalid")

    test_i = _header_index(header, "test")
    ts_i = _header_index(header, "t/s")
    backend_i = _header_index(header, "backend")
    ngl_i = _header_index(header, "ngl")
    threads_i = _header_index(header, "threads")
    model_i = _header_index(header, "model")
    if test_i is None or ts_i is None:
        raise LlamaBenchError("llama-bench markdown table missing test or t/s columns")

    measurement = LlamaBenchMeasurement()
    for cells in rows:
        if max(test_i, ts_i) >= len(cells):
            continue
        test_match = _TEST_CELL.search(cells[test_i].replace(" ", ""))
        if not test_match:
            test_match = _TEST_CELL.search(cells[test_i])
        ts_match = _TS_CELL.search(cells[ts_i])
        if not test_match or not ts_match:
            continue
        kind = test_match.group(1).lower()
        count = int(test_match.group(2))
        avg = float(ts_match.group(1))
        std = float(ts_match.group(2))
        if kind == "pp":
            measurement.pp_tps = avg
            measurement.pp_stddev = std
            measurement.n_prompt = count
        else:
            measurement.tg_tps = avg
            measurement.tg_stddev = std
            measurement.n_gen = count
        if backend_i is not None and backend_i < len(cells):
            measurement.backend = cells[backend_i]
        if ngl_i is not None and ngl_i < len(cells):
            measurement.n_gpu_layers = _optional_int(cells[ngl_i])
        if threads_i is not None and threads_i < len(cells):
            measurement.n_threads = _optional_int(cells[threads_i])
        if model_i is not None and model_i < len(cells):
            measurement.model_type = cells[model_i]
    if measurement.pp_tps is None and measurement.tg_tps is None:
        raise LlamaBenchError("llama-bench markdown table had no pp/tg speed rows")
    if not measurement.timestamp:
        measurement.timestamp = utc_now()
    return measurement


def parse_llama_bench_output(stdout: str, stderr: str = "") -> LlamaBenchMeasurement:
    """Parse llama-bench stdout; JSON preferred, markdown as fallback."""
    if _extract_json_array(stdout):
        return parse_llama_bench_json(stdout)
    try:
        return parse_llama_bench_markdown(stdout)
    except LlamaBenchError as exc:
        detail = (stderr or stdout or str(exc)).strip()
        raise LlamaBenchError(detail[:500] or "llama-bench produced no parseable output") from exc


def require_llama_bench(binary: Path) -> Path:
    path = Path(binary)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise LlamaBenchMissing(path)
    return path


def build_bench_command(
    binary: Path,
    model_paths: Sequence[Path],
    config: BenchConfig,
) -> list[str]:
    if not model_paths:
        raise LlamaBenchError("no GGUF path given to llama-bench")
    cmd = [str(binary)]
    for path in model_paths:
        cmd.extend(["-m", str(path)])
    cmd.extend(
        [
            "-p",
            str(config.n_prompt),
            "-n",
            str(config.n_gen),
            "-r",
            str(config.repetitions),
            "-o",
            "json",
        ]
    )
    if config.n_depth:
        cmd.extend(["-d", str(config.n_depth)])
    return cmd


Runner = Callable[..., subprocess.CompletedProcess]


def run_llama_bench(
    *,
    binary: Path,
    model_paths: Sequence[Path],
    config: BenchConfig | None = None,
    runner: Runner | None = None,
) -> LlamaBenchMeasurement:
    """Run vendored llama-bench against local GGUFs. Fails loud if missing."""
    cfg = config or BenchConfig()
    path = require_llama_bench(binary)
    cmd = build_bench_command(path, model_paths, cfg)
    run = runner or subprocess.run
    try:
        result = run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise LlamaBenchError(f"failed to start llama-bench: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        raise LlamaBenchError(detail[:500] or f"llama-bench exited {result.returncode}")
    measurement = parse_llama_bench_output(result.stdout, stderr=result.stderr)
    if not measurement.n_prompt:
        measurement.n_prompt = cfg.n_prompt
    if not measurement.n_gen:
        measurement.n_gen = cfg.n_gen
    if measurement.n_depth == 0:
        measurement.n_depth = cfg.n_depth
    return measurement


def row_from_run(
    *,
    target: BenchTarget,
    measurement: LlamaBenchMeasurement,
    fingerprint: HardwareFingerprint,
    config: BenchConfig,
) -> BenchRow:
    quant = target.quant or quant_from_filename(measurement.model_filename)
    return BenchRow(
        hostname=fingerprint.hostname,
        model=target.model,
        quant=quant,
        file=target.file_rel,
        gpu_name=fingerprint.gpu_name,
        vram_total_mb=fingerprint.vram_total_mb,
        cpu_model=fingerprint.cpu_model,
        n_prompt=config.n_prompt,
        n_gen=config.n_gen,
        repetitions=config.repetitions,
        n_depth=config.n_depth,
        pp_tps=measurement.pp_tps,
        pp_stddev=measurement.pp_stddev,
        tg_tps=measurement.tg_tps,
        tg_stddev=measurement.tg_stddev,
        n_gpu_layers=measurement.n_gpu_layers,
        n_threads=measurement.n_threads,
        backend=measurement.backend,
        updated_at=measurement.timestamp or utc_now(),
        display=target.display,
    )
