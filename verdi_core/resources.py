"""Model-independent GPU discovery and conservative placement helpers."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int

    @property
    def free_memory_mib(self) -> int:
        return max(0, self.memory_total_mib - self.memory_used_mib)


class GPUInventory:
    def __init__(self, gpus: list[GPU]):
        self.gpus = list(gpus)

    @classmethod
    def discover(cls) -> "GPUInventory":
        try:
            raw = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
            ], text=True, stderr=subprocess.STDOUT, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return cls([])
        gpus: list[GPU] = []
        for line in raw.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 5:
                continue
            try:
                gpus.append(GPU(int(fields[0]), fields[1], int(fields[2]), int(fields[3]), int(float(fields[4]))))
            except ValueError:
                continue
        return cls(gpus)

    def select(self, count: int, *, min_free_memory_mib: int = 0) -> list[int] | None:
        candidates = [gpu for gpu in self.gpus if gpu.free_memory_mib >= min_free_memory_mib]
        candidates.sort(key=lambda gpu: (gpu.utilization_pct, -gpu.free_memory_mib, gpu.index))
        if len(candidates) < count:
            return None
        return [gpu.index for gpu in candidates[:count]]
