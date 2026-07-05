from typing import Any, MutableMapping

import torch

from .. import nn
from .data import Trajectory

class CircularTensor(nn.Module):
    '''
    Implements a circular buffer for storing tensors. Supports ``state_dict()`` and ``load_state_dict()`` for checkpointing.

    Args:
        size (int): The maximum number of elements in the buffer.
        dtype (torch.dtype, optional): The data type of the elements in the buffer. Defaults to torch.float32.
    '''
    def __init__(self, size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self._size: torch.Tensor
        self.register_buffer('_size', torch.tensor(size, dtype=torch.long), persistent=True)
        self.data: torch.Tensor | None
        self.register_buffer('data', None, persistent=True)
        self.length: torch.Tensor
        self.register_buffer('length', torch.tensor(0, dtype=torch.long), persistent=True)
        self.counter: torch.Tensor
        self.register_buffer('counter', torch.tensor(0, dtype=torch.long), persistent=True)
        self.dtype = dtype

    @property
    def size(self) -> int:
        '''
        Get the maximum size of the buffer.

        Returns:
            int: The maximum size of the buffer.
        '''
        return int(self._size.item())

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
            raise ValueError(f'Incompatible dtype, expected {self.dtype}, got {value.dtype}')
        if self.data is None:
            self.data = torch.zeros((self.size, *N), dtype=self.dtype, device=self.device)
        elif tuple(N) != tuple(self.data.shape[1:]):
            raise ValueError(f'Incompatible shape, expected {self.data.shape[1:]}, got {N}')
        if not B:
            return
        if B > self.size:
            value = value[-self.size:]
            B = self.size
        logical_idx = (self.counter + torch.arange(B, device=self.device)) % self.size
        self.data[logical_idx] = value
        self.counter.copy_((self.counter + B) % self.size)
        self.length.copy_((self.length + B).clamp_max(self._size))

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
        return int(self.length.item())

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
        return self.data[idx].contiguous()

    def forward(self, x):
        ''' Get a batch of elements according to the index. '''
        return self[x]

class ReplayBuffer(nn.Module):
    '''
    Implements a replay buffer for storing and sampling experiences.

    Args:
        size (int): The maximum size of the buffer.
        continuous_action (bool, optional): Whether the action space is continuous. Defaults to False.
    '''
    def __init__(self, size: int, continuous_action: bool = False):
        super().__init__()
        self.size = size
        action_dtype = torch.float32 if continuous_action else torch.long
        self.keys = ['state', 'action', 'reward', 'next_state', 'done', 'log_pi']
        self.dtypes = [torch.float32 if k != 'action' else action_dtype for k in self.keys]
        self.data: MutableMapping[str, CircularTensor] = torch.nn.ModuleDict({
            k: CircularTensor(size, dtype=dtype)
            for k, dtype in zip(self.keys, self.dtypes)
        }) # type: ignore

    @property
    def length(self):
        ''' Get the current length of the buffer. '''
        return min(len(v) for v in self.data.values())

    def __getitem__(self, idx: torch.Tensor) -> Trajectory:
        ''' Sample a batch of experiences from the buffer. '''
        return Trajectory(*(self.data[k][idx] for k in self.keys), total_reward=float('nan'))

    def get_batch(self, idx: torch.Tensor) -> Trajectory:
        ''' Sample a batch of experiences from the buffer. '''
        return self[idx]

    def forward(self, idx: torch.Tensor) -> Trajectory:
        ''' Sample a batch of experiences from the buffer. '''
        return self[idx]

    def sample_index(self, batch_size: int) -> torch.Tensor:
        ''' Randomly sample a batch of indices from the buffer. '''
        if self.length == 0:
            raise ValueError('Buffer is empty')
        if self.length < batch_size:
            raise ValueError('Not enough samples in buffer')
        return torch.randperm(self.length, dtype=torch.long)[:batch_size]

    def store(self, trajectory: Trajectory):
        ''' Store a batch of new experience in the buffer. '''
        B = None
        for k in self.keys:
            if B is not None and v.shape[0] != B:
                raise ValueError('Inconsistent batch size')
            v = trajectory.get(k)
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
    '''
    def __init__(self, size: int, continuous_action: bool = False, p_max: float | int = 1e3):
        super().__init__(size, continuous_action)
        self.weight = CircularTensor(size, dtype=torch.float)
        self.p_max: torch.Tensor
        self.register_buffer('p_max', torch.tensor(p_max, dtype=torch.float32), persistent=True)

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
        idx = torch.multinomial(
            self.weight.as_tensor(), batch_size, replacement=False
        ).to(dtype=torch.long)
        return idx

    def store(self, trajectory: Trajectory):
        ''' Store a batch of new experience in the buffer. '''
        super().store(trajectory)
        B = trajectory.get('state').shape[0]
        weight = self.p_max.unsqueeze(0).expand(B)
        self.weight.append(weight)
