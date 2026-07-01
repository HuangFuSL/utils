'''
`utils.ctorch.device` - Utilities for managing and monitoring GPU devices in PyTorch.
'''
import dataclasses
import time
from typing import List, Tuple

import torch


@dataclasses.dataclass
class GpuStat:
    '''
    Dataclass that represents the status of a GPU device.

    Args:
        idx (int): Index of the GPU.
        name (str): Name of the GPU.
        avg_util (float): Average utilization percentage over the sampling period.
        avg_free_gb (float): Average free memory in GB over the sampling period.
        total_gb (float): Total memory in GB of the GPU.
    '''
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

    def _collect_stats(samples: int, interval: float,
                       handles: list = None) -> List[GpuStat]:
        own_init = handles is None
        if own_init:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

        util_hist = [[0.0] * samples for _ in range(len(handles))]
        free_hist = [[0.0] * samples for _ in range(len(handles))]
        total = [0.0] * len(handles)

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

        if own_init:
            pynvml.nvmlShutdown()
        return gpu_stats
except ImportError:
    def _sample_once(handles) -> List[Tuple[float, float, float]]:
        raise ImportError(
            'pynvml is required to collect GPU statistics. '
            'Please install it with `pip install pynvml`.'
        )
    def _collect_stats(samples: int, interval: float,
                       handles: list = None) -> List[GpuStat]:
        raise ImportError(
            'pynvml is required to collect GPU statistics. '
            'Please install it with `pip install pynvml`.'
        )


def get_best_device(
    window_sec: float = 3.0,
    interval_sec: float = 0.5,
    min_free_gb: float = 0,
    retry_interval_sec: float = 10.0,
    timeout_sec: float = 0,
) -> str:
    '''
    Automatically selects the best available device based on GPU utilization
    and memory.  When ``min_free_gb`` is positive the function will wait (and
    retry) until the chosen device has at least that much free VRAM.

    Args:
        window_sec (float): Total time in seconds to collect GPU statistics
            per sampling window.
        interval_sec (float): Time in seconds between each sample within a
            window.
        min_free_gb (float): Minimum free VRAM (GiB) required on the selected
            device.  Defaults to 0 (disabled).  Set to a positive value (e.g.
            4) to wait for sufficient free memory.
        retry_interval_sec (float): Seconds to wait between retry attempts
            when no device meets the VRAM threshold.
        timeout_sec (float): Maximum seconds to wait before raising an error.
            0 (default) means block indefinitely.

    Returns:
        str: The best device identifier, either ``'mps'``, ``'cuda:<idx>'``,
        or ``'cpu'``.

    Raises:
        RuntimeError: If ``timeout_sec > 0`` and no suitable CUDA device is
            found within the time limit.
    '''
    if torch.mps.is_available():
        return 'mps'

    if torch.cuda.is_available():
        if pynvml is None:
            raise ImportError(
                'pynvml is required to detect CUDA devices. '
                'Please install it with `pip install pynvml`.'
            )

        samples = int(window_sec / interval_sec)

        pynvml.nvmlInit()
        try:
            n = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        except Exception:
            pynvml.nvmlShutdown()
            raise

        start_time = time.time()
        while True:
            stats = _collect_stats(
                samples=samples,
                interval=interval_sec,
                handles=handles,
            )
            stats.sort(key=lambda s: (s.avg_util, -s.avg_free_gb, -s.total_gb))

            for stat in stats:
                if stat.avg_free_gb >= min_free_gb:
                    pynvml.nvmlShutdown()
                    return f'cuda:{stat.idx}'

            elapsed = time.time() - start_time
            if timeout_sec > 0 and elapsed >= timeout_sec:
                pynvml.nvmlShutdown()
                raise RuntimeError(
                    f'No CUDA device with ≥ {min_free_gb} GiB free memory '
                    f'found within {timeout_sec} s timeout.'
                )

            time.sleep(retry_interval_sec)

    # CPU fallback
    return 'cpu'