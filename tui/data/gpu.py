"""GPU stats — AMD (rocm-smi/sysfs) first, NVIDIA fallback, else system RAM."""

from __future__ import annotations

import subprocess
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPUStats:
    name: str = ""
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    utilization_pct: float = 0.0
    temp_c: float = 0.0
    available: bool = False

    @property
    def vram_pct(self) -> float:
        return (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb else 0.0


def _try_nvidia() -> GPUStats | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        parts = [p.strip() for p in r.stdout.splitlines()[0].split(",")]
        return GPUStats(
            name=parts[0],
            utilization_pct=float(parts[1]),
            vram_used_mb=float(parts[2]),
            vram_total_mb=float(parts[3]),
            temp_c=float(parts[4]),
            available=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


def _try_rocm() -> GPUStats | None:
    try:
        r = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        stats = GPUStats(name="AMD GPU", available=True)
        rows = list(csv.reader(line for line in r.stdout.splitlines() if line.strip()))
        header_index = next(
            (
                index for index, row in enumerate(rows)
                if any("vram total memory (b)" in cell.lower() for cell in row)
            ),
            None,
        )
        if header_index is not None and header_index + 1 < len(rows):
            header = [cell.strip().lower() for cell in rows[header_index]]
            values = [cell.strip() for cell in rows[header_index + 1]]
            try:
                total_index = header.index("vram total memory (b)")
                used_index = header.index("vram total used memory (b)")
                stats.vram_total_mb = float(values[total_index]) / 1e6
                stats.vram_used_mb = float(values[used_index]) / 1e6
            except (ValueError, IndexError):
                pass
            try:
                utilization_index = header.index("gpu use (%)")
                stats.utilization_pct = float(values[utilization_index])
            except (ValueError, IndexError):
                pass
            for index, cell in enumerate(header):
                if ("temperature" in cell or "edge" in cell or "junction" in cell):
                    try:
                        stats.temp_c = float(values[index])
                    except (ValueError, IndexError):
                        pass
                    break
        return stats
    except (FileNotFoundError, subprocess.TimeoutExpired, csv.Error):
        pass

    # sysfs fallback for AMD
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        busy_f = card / "device" / "gpu_busy_percent"
        if not busy_f.exists():
            continue
        dev = card / "device"
        try:
            busy = float(busy_f.read_text().strip())
            used = int((dev / "mem_info_vram_used").read_text().strip()) / 1e6
            total = int((dev / "mem_info_vram_total").read_text().strip()) / 1e6
            name = "AMD GPU"
            try:
                name = f"AMD ({(dev / 'product_name').read_text().strip()})"
            except OSError:
                pass
            temp = 0.0
            for hw in dev.glob("hwmon/hwmon*/temp1_input"):
                temp = int(hw.read_text().strip()) / 1000
                break
            return GPUStats(name=name, utilization_pct=busy,
                            vram_used_mb=used, vram_total_mb=total,
                            temp_c=temp, available=True)
        except (OSError, ValueError):
            continue
    return None


def _ram_fallback() -> GPUStats:
    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = line.partition(":")
        info[k] = int(v.strip().split()[0])  # kB
    total = info.get("MemTotal", 0) / 1024
    avail = info.get("MemAvailable", 0) / 1024
    return GPUStats(
        name="System RAM (CPU-only)",
        vram_used_mb=total - avail,
        vram_total_mb=total,
        available=True,
    )


def query_gpu() -> GPUStats:
    return _try_nvidia() or _try_rocm() or _ram_fallback()
