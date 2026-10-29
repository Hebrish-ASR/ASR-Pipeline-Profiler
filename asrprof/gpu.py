"""GPU probes, read from the sampler process.

Two backends:

- amdgpu sysfs (`/sys/class/drm/card*/device/gpu_busy_percent`): device-wide busy %, VRAM and
  GTT used. Not attributable to a process: anything else drawing on screen shows up too.
- NVML (`pynvml`, optional): device utilisation %, device memory, and this process's own
  memory from the running-process lists.

`probe()` returns the first backend that answers, or None. A probe that returns None means
"no GPU telemetry", not "the GPU was idle".
"""

from __future__ import annotations

import glob
import os
from pathlib import Path


class AmdSysfs:
    kind = "amdgpu-sysfs"

    def __init__(self, dev: Path):
        self.dev = dev
        self.name = f"amdgpu ({dev.parent.name})"
        try:
            pci = (dev / "uevent").read_text()
            pid = next((line.split("=", 1)[1] for line in pci.splitlines() if line.startswith("PCI_ID=")), "")
            if pid:
                self.name += f" PCI {pid}"
        except OSError:
            pass

    def _int(self, name: str) -> int | None:
        try:
            return int((self.dev / name).read_text().strip())
        except (OSError, ValueError):
            return None

    def info(self) -> dict:
        return {"backend": self.kind, "name": self.name, "attributable": False,
                "vram_total_mb": _mb(self._int("mem_info_vram_total")),
                "gtt_total_mb": _mb(self._int("mem_info_gtt_total"))}

    def read(self, pid: int) -> dict:
        return {"busy": self._int("gpu_busy_percent"),
                "vram_mb": _mb(self._int("mem_info_vram_used")),
                "gtt_mb": _mb(self._int("mem_info_gtt_used"))}


class Nvml:
    kind = "nvml"

    def __init__(self, nv, handle):
        self.nv, self.h = nv, handle
        n = nv.nvmlDeviceGetName(handle)
        self.name = n.decode() if isinstance(n, bytes) else n

    def info(self) -> dict:
        mem = self.nv.nvmlDeviceGetMemoryInfo(self.h)
        return {"backend": self.kind, "name": self.name, "attributable": True,
                "vram_total_mb": _mb(mem.total)}

    def read(self, pid: int) -> dict:
        nv = self.nv
        out = {"busy": nv.nvmlDeviceGetUtilizationRates(self.h).gpu,
               "vram_mb": _mb(nv.nvmlDeviceGetMemoryInfo(self.h).used), "proc_vram_mb": 0.0}
        try:
            procs = (nv.nvmlDeviceGetComputeRunningProcesses(self.h)
                     + nv.nvmlDeviceGetGraphicsRunningProcesses(self.h))
            out["proc_vram_mb"] = _mb(sum(p.usedGpuMemory or 0 for p in procs if p.pid == pid))
        except Exception:
            pass
        return out


def _mb(b: int | None) -> float | None:
    return None if b is None else round(b / 2**20, 1)


def probe():
    if os.environ.get("ASRPROF_GPU", "") == "off":
        return None
    try:
        import pynvml as nv
        nv.nvmlInit()
        if nv.nvmlDeviceGetCount():
            return Nvml(nv, nv.nvmlDeviceGetHandleByIndex(int(os.environ.get("ASRPROF_GPU_INDEX", "0"))))
    except Exception:
        pass
    for f in sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")):
        g = AmdSysfs(Path(f).parent)
        if g.read(0)["busy"] is not None:
            return g
    return None
