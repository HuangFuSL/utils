from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .data import Trajectory

try:
    from ...sampler.negsampler import weighted_batched_sample_with_negative
except ImportError:
    def weighted_batched_sample_with_negative(
        range_: np.ndarray, weights: np.ndarray,
        exclude: np.ndarray, k: int,
    ) -> np.ndarray:
        p = weights.astype(np.float64, copy=False)
        p /= p.sum()
        return np.random.choice(range_, size=k, replace=False, p=p).reshape(1, -1)

_NP_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.long: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
}

def _restore_dtype(s: str) -> torch.dtype:
    return getattr(torch, s.split('.')[-1])

def _resolve_target(x: torch.device | str | Path) -> torch.device | Path:
    '''
    Resolve a target to either a ``torch.device`` (for in-memory storage)
    or a ``Path`` (for mmap-backed disk storage).
    '''
    if isinstance(x, Path):
        return x
    try:
        return torch.device(str(x))
    except (RuntimeError, TypeError):
        return Path(str(x))

class CircularTensor:
    '''
    Implements a circular buffer for storing tensors. Supports ``state_dict()`` and ``load_state_dict()`` for checkpointing.

    Args:
        size (int): The maximum number of elements in the buffer.
        dtype (torch.dtype, optional): The data type of the elements in the buffer. Defaults to torch.float32.
        device (torch.device | str | Path): Storage target. Accepts device strings (``'cpu'``, ``'cuda:0'``) or a directory/file path for mmap-backed storage.
    '''
    def __init__(
        self, size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | Path = 'cpu'
    ):
        self._size = size
        self.dtype = dtype
        self.data: torch.Tensor | None = None
        self.length: int = 0
        self.counter: int = 0

        resolved = _resolve_target(device)
        if isinstance(resolved, Path):
            self.device = torch.device('cpu')
            self._mmap_file: Path | None = resolved
        else:
            self.device = resolved
            self._mmap_file = None

    @property
    def size(self) -> int:
        '''
        Get the maximum size of the buffer.

        Returns:
            int: The maximum size of the buffer.
        '''
        return self._size

    @property
    def is_mmap(self) -> bool:
        return self._mmap_file is not None

    def to(self, target: torch.device | str | Path) -> 'CircularTensor':
        '''
        Relocate buffer storage.

        * ``to('cpu')`` / ``to('cuda:0')`` — move tensor to that device.
        * ``to('/path/to/file.npy')`` — pin storage to a memory-mapped file.
        '''
        resolved = _resolve_target(target)
        if isinstance(resolved, Path):
            return self._to_mmap(resolved)
        else:
            return self._to_device(resolved)

    def _to_device(self, device: torch.device) -> 'CircularTensor':
        self._mmap_file = None
        self.device = device
        if self.data is not None:
            self.data = self.data.to(device)
        return self

    def _to_mmap(self, filepath: Path) -> 'CircularTensor':
        filepath = filepath.resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cpu')

        if self.data is not None:
            old_data = self.data.cpu()
            np_dtype = _NP_DTYPE[self.dtype]
            mmap = np.lib.format.open_memmap(
                str(filepath), mode='w+',
                dtype=np_dtype, shape=tuple(old_data.shape),
            )
            np.copyto(mmap, old_data.numpy())
            mmap.flush()
            self.data = torch.from_numpy(mmap)
        else:
            self._mmap_file = filepath

        self._mmap_file = filepath
        return self

    def flush(self) -> None:
        '''
        Flush mmap-backed data to disk. No-op for in-memory storage.
        '''
        if self._mmap_file is not None and self.data is not None:
            arr = self.data.numpy()
            if isinstance(arr, np.memmap):
                arr.flush()

    def state_dict(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            'size': self.size,
            'dtype': str(self.dtype),
            'device': str(self.device),
            'length': self.length,
            'counter': self.counter,
        }
        if self._mmap_file is not None:
            base['mmap_file'] = str(self._mmap_file)
            base['data'] = None
        else:
            base['data'] = self.data
        return base

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._size = state['size']
        self.dtype = _restore_dtype(state['dtype'])
        self.device = torch.device(state['device'])
        self.length = state['length']
        self.counter = state['counter']

        mmap_file = state.get('mmap_file')
        if mmap_file is not None:
            self._mmap_file = Path(mmap_file)
            self.data = torch.from_numpy(
                np.lib.format.open_memmap(str(self._mmap_file), mode='r+')
            )
        else:
            self._mmap_file = None
            self.data = state['data']

    def append(self, value: Any):
        '''
        Append a batch of new elements to the buffer.

        Args:
            value (Any): The new elements to append. Supports any input that can be converted to a tensor.

        Shapes:

            * Input shape: (B, \\*)
        '''
        value = torch.as_tensor(value, device=self.device)
        B, *N = value.shape
        if value.dtype != self.dtype:
            value = value.to(self.dtype)
        if self.data is None:
            shape = (self.size, *N)
            if self._mmap_file is not None:
                self._mmap_file.parent.mkdir(parents=True, exist_ok=True)
                self.data = torch.from_numpy(
                    np.lib.format.open_memmap(
                        str(self._mmap_file), mode='w+',
                        dtype=_NP_DTYPE[self.dtype], shape=shape,
                    )
                )
            else:
                self.data = torch.zeros(shape, dtype=self.dtype, device=self.device)
        elif tuple(N) != tuple(self.data.shape[1:]):
            raise ValueError(f'Incompatible shape, expected {self.data.shape[1:]}, got {N}')
        if not B:
            return
        if B > self.size:
            value = value[-self.size:]
            B = self.size
        logical_idx = (self.counter + torch.arange(B, device=self.device)) % self.size
        self.data[logical_idx] = value
        self.counter = (self.counter + B) % self.size
        self.length = min(self.length + B, self.size)

    def as_tensor(self):
        ''' Get the underlying tensor. '''
        if self.data is None:
            raise ValueError('Buffer is empty')
        return self.data[:self.length]

    def as_numpy(self, force: bool = False):
        ''' Get the underlying numpy array. '''
        if self.data is None:
            raise ValueError('Buffer is empty')
        return self.as_tensor().numpy(force=force)

    def __len__(self) -> int:
        ''' Get the current length of the buffer. '''
        return self.length

    def __setitem__(self, idx: torch.Tensor, value: torch.Tensor):
        ''' Overwrite a batch of elements according to the index. '''
        if self.data is None:
            raise IndexError('Buffer is empty')
        idx = torch.as_tensor(idx, device=self.device)
        idx %= self.size
        if torch.any(idx >= self.length):
            raise IndexError('Index out of range')
        self.data[idx] = value.to(self.dtype)

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        ''' Get a batch of elements according to the index. '''
        if self.data is None:
            raise IndexError('Buffer is empty')
        idx = torch.as_tensor(idx, device=self.device)
        idx %= self.size
        if torch.any(idx >= self.length):
            raise IndexError('Index out of range')
        return self.data[idx]

    def forward(self, x):
        ''' Get a batch of elements according to the index. '''
        return self[x]

    __call__ = forward

class ReplayBuffer:
    '''
    Implements a replay buffer for storing and sampling experiences.

    Args:
        size (int): The maximum size of the buffer.
        continuous_action (bool, optional): Whether the action space is continuous. Defaults to False.
        device (torch.device | str | Path): Storage device. Accepts device strings or a directory path for mmap-backed storage (one ``.npy`` file per field).
    '''
    def __init__(self, size: int, continuous_action: bool = False, device: torch.device | str | Path = 'cpu'):
        resolved = _resolve_target(device)
        if isinstance(resolved, Path):
            self.device = torch.device('cpu')
            self._mmap_dir: Path | None = resolved
        else:
            self.device = resolved
            self._mmap_dir = None

        self.size = size
        action_dtype = torch.float32 if continuous_action else torch.long
        self.keys = ['state', 'action', 'reward', 'next_state', 'term', 'trunc', 'log_pi']
        self.dtypes = [torch.float32 if k != 'action' else action_dtype for k in self.keys]

        self.data: Dict[str, CircularTensor] = {
            k: CircularTensor(size, dtype=dtype, device=device)
            for k, dtype in zip(self.keys, self.dtypes)
        }

    @property
    def length(self):
        ''' Get the current length of the buffer. '''
        return min(len(v) for v in self.data.values())

    @property
    def is_mmap(self) -> bool:
        return self._mmap_dir is not None

    def to(self, device: torch.device | str | Path) -> 'ReplayBuffer':
        '''
        Relocate buffer storage.

        * ``to('cuda:0')`` — move to GPU.
        * ``to('/path/to/dir')`` — pin each field to ``{dir}/{key}.npy``.
        '''
        resolved = _resolve_target(device)
        if isinstance(resolved, Path):
            self._mmap_dir = resolved
            self.device = torch.device('cpu')
            resolved.mkdir(parents=True, exist_ok=True)
            for k, v in self.data.items():
                v.to(resolved / f'{k}.npy')
        else:
            self._mmap_dir = None
            self.device = resolved
            for v in self.data.values():
                v.to(resolved)
        return self

    def flush(self) -> None:
        for v in self.data.values():
            v.flush()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'device': str(self.device),
            'mmap_dir': str(self._mmap_dir) if self._mmap_dir else None,
            'keys': self.keys,
            'dtypes': [str(d) for d in self.dtypes],
            'data': {k: v.state_dict() for k, v in self.data.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.device = torch.device(state['device'])
        self._mmap_dir = Path(state['mmap_dir']) if state.get('mmap_dir') else None
        for k, v in self.data.items():
            v.load_state_dict(state['data'][k])

    def __getitem__(self, idx: torch.Tensor) -> Trajectory:
        ''' Sample a batch of experiences from the buffer. '''
        return Trajectory.from_tensors(*(self.data[k][idx] for k in self.keys))

    def get_batch(self, idx: torch.Tensor) -> Trajectory:
        ''' Sample a batch of experiences from the buffer. '''
        return self[idx]

    def forward(self, idx: torch.Tensor) -> Trajectory:
        ''' Sample a batch of experiences from the buffer. '''
        return self[idx]

    __call__ = forward

    def sample_index(self, batch_size: int) -> torch.Tensor:
        ''' Randomly sample a batch of indices from the buffer. '''
        if self.length == 0:
            raise ValueError('Buffer is empty')
        if self.length < batch_size:
            raise ValueError('Not enough samples in buffer')
        return torch.randint(0, self.length, (batch_size,), dtype=torch.long)

    def store(self, trajectory: Trajectory):
        ''' Store a batch of new experience in the buffer. '''
        B = None
        for k in self.keys:
            v = trajectory.get(k)
            if B is not None and v.shape[0] != B:
                raise ValueError('Inconsistent batch size')
            v = torch.as_tensor(v, dtype=self.data[k].dtype, device=self.data[k].device)
            self.data[k].append(v)
            B = v.shape[0]

class PrioritizedReplayBuffer(ReplayBuffer):
    '''
    Implements a prioritized replay buffer for storing and sampling experiences.

    Args:
        size (int): The maximum size of the buffer.
        continuous_action (bool, optional): Whether the action space is continuous. Defaults to False.
        p_max (float | int, optional): Default priority value for newly incoming experiences. Defaults to 1e3.
        device (torch.device | str | Path): Storage device.
    '''
    def __init__(
        self, size: int, continuous_action: bool = False,
        p_max: float | int = 1e3,
        device: torch.device | str | Path = 'cpu'
    ):
        super().__init__(size, continuous_action, device)
        self.weight = CircularTensor(size, dtype=torch.float, device=self.device)
        self.p_max: torch.Tensor = torch.tensor(
            p_max, dtype=torch.float32, device=self.device,
        )
        self._sample_range: np.ndarray | None = None

    def set_weight(self, idx: torch.Tensor, weight: torch.Tensor, alpha: float = 1.0):
        ''' Set the sampling weight for a batch of experiences. '''
        self.weight[idx] = weight.detach().reshape(-1).abs().clamp_min(1e-6).pow(alpha)

    def get_weight(self, idx: torch.Tensor) -> torch.Tensor:
        ''' Get the sampling weight for a batch of experiences. '''
        return self.weight[idx]

    def get_ipw(self, idx: torch.Tensor, beta: float = 1.0, eps: float = 5e-4):
        '''
        Get the inverse probability weighting (IPW) for a batch of experiences, to balance the loss function. The IPW weight is given by:

        .. math::
            \\tilde w_i = \\left(\\frac{N\\cdot w_i}{\\sum_j w_j}\\right)^{\\beta}

        Args:
            idx (torch.Tensor): The indices of the experiences to retrieve.
            beta (float): The beta parameter for the importance sampling.

        Returns:
            torch.Tensor: The importance sampling weights for the specified experiences.
        '''
        sample_weights = self.get_weight(idx).clamp_min(eps)
        ipw = (sample_weights / sample_weights.mean(dim=0, keepdim=True)).pow(-beta)
        ipw /= ipw.max()
        return ipw

    def sample_index(self, batch_size: int) -> torch.Tensor:
        ''' Sample a batch of indices from the replay buffer. '''
        if self.length == 0:
            raise ValueError('Buffer is empty')
        if self.length < batch_size:
            raise ValueError('Not enough samples in buffer')

        if self._sample_range is None or len(self._sample_range) != self.length:
            self._sample_range = np.arange(self.length, dtype=np.int64)
        w = self.weight.as_tensor()[:self.length].cpu().numpy().astype(np.float64)
        idx = weighted_batched_sample_with_negative(
            range_=self._sample_range,
            weights=w,
            exclude=np.full((1, 1), -1, dtype=np.int64),
            k=batch_size,
        )
        return torch.as_tensor(idx[0], dtype=torch.long)

    def store(self, trajectory: Trajectory):
        ''' Store a batch of new experience in the buffer. '''
        super().store(trajectory)
        B = trajectory.get('state').shape[0]
        weight = self.p_max.unsqueeze(0).expand(B)
        self.weight.append(weight)

    def to(self, target: torch.device | str | Path) -> 'PrioritizedReplayBuffer':
        resolved = _resolve_target(target)
        if isinstance(resolved, Path):
            super().to(resolved)
            self.weight.to('cpu')
        else:
            super().to(resolved)
            self.weight.to(resolved)
            self.p_max = self.p_max.to(resolved)
        return self

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        w_sd = self.weight.state_dict()
        w_sd.pop('mmap_file', None)
        state['weight'] = w_sd
        state['p_max'] = self.p_max
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if 'mmap_file' in state.get('weight', {}):
            state['weight'].pop('mmap_file')
        self.weight.load_state_dict(state['weight'])
        self.p_max = state['p_max']
        self._sample_range = None  # invalidate cache — length may differ
