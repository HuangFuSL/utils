'''
device.py - Utilities for managing and monitoring GPU devices in PyTorch.
'''
import dataclasses
import time
from typing import List, Tuple

import torch


@dataclasses.dataclass
class GpuStat:
    idx: int
    name: str
    avg_util: float
    avg_free_gb: float
    total_gb: float

try:
    import pynvml

    def _sample_once(handles) -> List[Tuple[float, float, float]]:
        stats = []
        for h in handles:
            u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            stats.append((u, mem.free, mem.total))
        return stats

    def _collect_stats(samples: int, interval: float) -> List[GpuStat]:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

        util_hist = [[0.0] * samples for _ in range(n)]
        free_hist = [[0.0] * samples for _ in range(n)]
        total = [0.0] * n

        for s in range(samples):
            time.sleep(interval) if s else None
            snap = _sample_once(handles)
            for i, (u, f, t) in enumerate(snap):
                util_hist[i][s] = u
                free_hist[i][s] = f
                if s == 0:
                    total[i] = t

        gpu_stats = []
        for i, h in enumerate(handles):
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            total_gb = total[i] / 1024 ** 3
            avg_util = sum(util_hist[i]) / samples
            avg_free_gb = (sum(free_hist[i]) / samples) // 1024 ** 3
            gpu_stats.append(GpuStat(i, name, avg_util, avg_free_gb, total_gb))

        pynvml.nvmlShutdown()
        return gpu_stats
except ImportError:
    def _sample_once(handles) -> List[Tuple[float, float, float]]:
        raise ImportError(
            'pynvml is required to collect GPU statistics. '
            'Please install it with `pip install pynvml`.'
        )
    def _collect_stats(samples: int, interval: float) -> List[GpuStat]:
        raise ImportError(
            'pynvml is required to collect GPU statistics. '
            'Please install it with `pip install pynvml`.'
        )


def get_best_device(
    window_sec: float = 3.0,
    interval_sec: float = 0.5
) -> str:
    if torch.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        if pynvml is None:
            raise ImportError(
                'pynvml is required to detect CUDA devices. Please install it with `pip install pynvml`.'
            )
        pynvml.nvmlInit()
        stats = _collect_stats(
            samples=int(window_sec / interval_sec),
            interval=interval_sec
        )
        stats.sort(key=lambda s: (s.avg_util, -s.avg_free_gb, -s.total_gb))
        pynvml.nvmlShutdown()
        return f'cuda:{stats[0].idx}'
    # CPU fallback
    return 'cpu'