"""GPU stats — AMD (rocm-smi/sysfs) first, NVIDIA fallback, else system RAM."""

from __future__ import annotations

import csv
import subprocess
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
    unified: bool = False
    dedicated_used_mb: float = 0.0
    dedicated_total_mb: float = 0.0

    @property
    def vram_pct(self) -> float:
        return (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb else 0.0

    @property
    def memory_label(self) -> str:
        return "GPU RAM (GTT)" if self.unified else "VRAM"


def _pci_product_name() -> str | None:
    """Best-effort GPU marketing name from lspci, used when rocm-smi is generic."""
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        lower = line.lower()
        if not any(token in lower for token in ("vga", "3d controller", "display controller")):
            continue
        if not any(token in lower for token in ("amd", "ati", "nvidia", "radeon")):
            continue
        if ": " in line:
            return line.split(": ", 1)[1].strip()
    return None


def _bytes_to_mb(raw: float) -> float:
    return raw / 1e6


def effective_gpu_memory(
    vram_used_mb: float,
    vram_total_mb: float,
    gtt_used_mb: float = 0.0,
    gtt_total_mb: float = 0.0,
) -> tuple[float, float, bool]:
    """Choose the pool llama.cpp actually fills.

    Discrete GPUs: dedicated VRAM. APUs report a tiny VRAM BAR (often 512 MB)
    while weights live in GTT (GPU-accessible system RAM).
    """
    unified = (
        gtt_total_mb > 0
        and vram_total_mb > 0
        and vram_total_mb < 2048
        and gtt_total_mb > vram_total_mb * 2
    )
    if unified:
        return gtt_used_mb, gtt_total_mb, True
    return vram_used_mb, vram_total_mb, False


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


def parse_rocm_csv(text: str) -> GPUStats | None:
    """Parse `rocm-smi --csv` VRAM / util / temp for the first GPU."""
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    header_index = next(
        (
            index for index, row in enumerate(rows)
            if any("vram total memory (b)" in cell.lower() for cell in row)
        ),
        None,
    )
    if header_index is None or header_index + 1 >= len(rows):
        return None
    header = [cell.strip().lower() for cell in rows[header_index]]
    values = [cell.strip() for cell in rows[header_index + 1]]
    stats = GPUStats(name="AMD GPU", available=True)
    try:
        total_index = header.index("vram total memory (b)")
        used_index = header.index("vram total used memory (b)")
        stats.vram_total_mb = _bytes_to_mb(float(values[total_index]))
        stats.vram_used_mb = _bytes_to_mb(float(values[used_index]))
    except (ValueError, IndexError):
        pass
    try:
        utilization_index = header.index("gpu use (%)")
        stats.utilization_pct = float(values[utilization_index])
    except (ValueError, IndexError):
        pass
    for index, cell in enumerate(header):
        if "temperature" in cell or "edge" in cell or "junction" in cell:
            try:
                stats.temp_c = float(values[index])
            except (ValueError, IndexError):
                pass
            break
    return stats


def _amd_sysfs_cards() -> list[GPUStats]:
    cards: list[GPUStats] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        busy_f = card / "device" / "gpu_busy_percent"
        vram_total_f = card / "device" / "mem_info_vram_total"
        if not busy_f.exists() or not vram_total_f.exists():
            continue
        dev = card / "device"
        try:
            busy = float(busy_f.read_text().strip())
            vram_used = _bytes_to_mb(int((dev / "mem_info_vram_used").read_text().strip()))
            vram_total = _bytes_to_mb(int(vram_total_f.read_text().strip()))
            gtt_used = 0.0
            gtt_total = 0.0
            try:
                gtt_used = _bytes_to_mb(int((dev / "mem_info_gtt_used").read_text().strip()))
                gtt_total = _bytes_to_mb(int((dev / "mem_info_gtt_total").read_text().strip()))
            except (OSError, ValueError):
                pass
            used, total, unified = effective_gpu_memory(
                vram_used, vram_total, gtt_used, gtt_total
            )
            name = "AMD GPU"
            try:
                product = (dev / "product_name").read_text().strip()
                if product:
                    name = f"AMD ({product})"
            except OSError:
                pass
            temp = 0.0
            for hw in dev.glob("hwmon/hwmon*/temp1_input"):
                temp = int(hw.read_text().strip()) / 1000
                break
            cards.append(
                GPUStats(
                    name=name,
                    utilization_pct=busy,
                    vram_used_mb=used,
                    vram_total_mb=total,
                    temp_c=temp,
                    available=True,
                    unified=unified,
                    dedicated_used_mb=vram_used,
                    dedicated_total_mb=vram_total,
                )
            )
        except (OSError, ValueError):
            continue
    return cards


def _best_amd_sysfs() -> GPUStats | None:
    cards = _amd_sysfs_cards()
    if not cards:
        return None
    return max(cards, key=lambda stats: stats.vram_total_mb)


def _try_rocm() -> GPUStats | None:
    stats: GPUStats | None = None
    try:
        r = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            stats = parse_rocm_csv(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        stats = None

    sysfs = _best_amd_sysfs()
    if sysfs is not None:
        if stats is None:
            stats = sysfs
        else:
            stats.vram_used_mb = sysfs.vram_used_mb
            stats.vram_total_mb = sysfs.vram_total_mb
            stats.unified = sysfs.unified
            stats.dedicated_used_mb = sysfs.dedicated_used_mb
            stats.dedicated_total_mb = sysfs.dedicated_total_mb
            if sysfs.utilization_pct and not stats.utilization_pct:
                stats.utilization_pct = sysfs.utilization_pct
            if sysfs.temp_c and not stats.temp_c:
                stats.temp_c = sysfs.temp_c
            if sysfs.name != "AMD GPU":
                stats.name = sysfs.name

    if stats is None:
        return None

    if stats.name == "AMD GPU":
        pci_name = _pci_product_name()
        if pci_name:
            stats.name = pci_name
    if stats.unified and "unified" not in stats.name.lower() and "gtt" not in stats.name.lower():
        stats.name = f"{stats.name} (unified)"
    return stats


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
        unified=True,
    )


def query_gpu() -> GPUStats:
    return _try_nvidia() or _try_rocm() or _ram_fallback()
