import dataclasses
from typing import Callable, Dict, List, Tuple
import copy
import functools

import torch


def _default_shape(
    s: torch.Tensor, a: torch.Tensor,
    r: torch.Tensor, s_prime: torch.Tensor,
    term: torch.Tensor, trunc: torch.Tensor
) -> torch.Tensor:
    return r


RewardMapping = Callable[
    [
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor
    ], torch.Tensor
]

def calculate_mask(done: torch.Tensor, trunc: torch.Tensor | None = None) -> torch.Tensor:
    if trunc is not None:
        done = ((done + trunc) > 0).to(torch.int64)
    *B, L = done.shape
    traj_len = torch.argmax(done, dim=-1) # (*B,)
    mask = torch.arange(L, device=done.device).expand(*B, L) \
        < traj_len.unsqueeze(-1) # (*B, L)
    return mask

@dataclasses.dataclass()
class Trajectory():
    _data: torch.Tensor
    state_shape: torch.Size | Tuple[int, ...]
    action_shape: torch.Size | Tuple[int, ...]
    action_dtype: torch.dtype = torch.float32
    total_reward: float = float('nan')

    @property
    def tensor_fields(self) -> List[str]:
        return [
            'state', 'action', 'reward', 'next_state', 'term', 'trunc', 'log_pi'
        ]

    @functools.cached_property
    def state_size(self) -> int:
        return functools.reduce(lambda x, y: x * y, self.state_shape, 1)

    @functools.cached_property
    def action_size(self) -> int:
        return functools.reduce(lambda x, y: x * y, self.action_shape, 1)

    @functools.cached_property
    def all_size(self) -> int:
        return self.state_size + self.action_size + 1 + self.state_size + 1 + 1

    @functools.cached_property
    def slice_dict(self) -> Dict[str, slice]:
        ret = {}
        left, right = 0, 0
        sizes = [self.state_size, self.action_size, 1, self.state_size, 1, 1, 1]
        for name, size in zip(self.tensor_fields, sizes):
            left, right = right, right + size
            ret[name] = slice(left, right)
        return ret

    @property
    def state(self) -> torch.Tensor:
        return self._data[:, self.slice_dict['state']] \
            .view(-1, *self.state_shape)

    @state.setter
    def state(self, value: torch.Tensor):
        self._data[:, self.slice_dict['state']] = \
            value.view(-1, self.state_size)

    @property
    def action(self) -> torch.Tensor:
        return self._data[:, self.slice_dict['action']] \
            .view(-1, *self.action_shape).to(self.action_dtype)

    @action.setter
    def action(self, value: torch.Tensor):
        self._data[:, self.slice_dict['action']] = \
            value.view(-1, self.action_size).to(torch.float32)

    @property
    def reward(self) -> torch.Tensor:
        return self._data[:, self.slice_dict['reward']].view(-1)

    @reward.setter
    def reward(self, value: torch.Tensor):
        self._data[:, self.slice_dict['reward']] = value.view(-1, 1)

    @property
    def next_state(self) -> torch.Tensor:
        return self._data[:, self.slice_dict['next_state']] \
            .view(-1, *self.state_shape)

    @next_state.setter
    def next_state(self, value: torch.Tensor):
        self._data[:, self.slice_dict['next_state']] = \
            value.view(-1, self.state_size)

    @property
    def term(self) -> torch.Tensor:
        return self._data[:, self.slice_dict['term']].view(-1)

    @term.setter
    def term(self, value: torch.Tensor):
        self._data[:, self.slice_dict['term']] = value.view(-1, 1)

    @property
    def trunc(self) -> torch.Tensor:
        return self._data[:, self.slice_dict['trunc']].view(-1)

    @trunc.setter
    def trunc(self, value: torch.Tensor):
        self._data[:, self.slice_dict['trunc']] = value.view(-1, 1)

    @property
    def done(self) -> torch.Tensor:
        return ((
            self._data[:, self.slice_dict['term']] +
            self._data[:, self.slice_dict['trunc']]
        ) > 0).to(torch.int64)

    @property
    def log_pi(self) -> torch.Tensor:
        return self._data[:, self.slice_dict['log_pi']].view(-1)

    @log_pi.setter
    def log_pi(self, value: torch.Tensor):
        self._data[:, self.slice_dict['log_pi']] = value.view(-1, 1)

    def get(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    def __iter__(self):
        for key in [
            'state', 'action', 'reward', 'next_state', 'done', 'log_pi'
        ]:
            yield self.get(key)

    @classmethod
    def fixed_length(
        cls, length: int,
        state_shape: torch.Size | Tuple[int, ...],
        action_shape: torch.Size | Tuple[int, ...],
        action_dtype: torch.dtype = torch.float32,
        total_reward: float = float('nan'),
        pin_memory: bool = False
    ) -> 'Trajectory':
        if length <= 0:
            raise ValueError('length must be positive.')
        state_size = functools.reduce(lambda x, y: x * y, state_shape, 1)
        action_size = functools.reduce(lambda x, y: x * y, action_shape, 1)
        all_size = state_size + action_size + 1 + state_size + 1 + 1
        data = torch.zeros(
            (length, all_size), dtype=torch.float32, pin_memory=pin_memory
        )
        return cls(
            _data=data,
            state_shape=state_shape,
            action_shape=action_shape,
            action_dtype=action_dtype,
            total_reward=total_reward
        )

    @classmethod
    def from_tensors(
        cls, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
        next_state: torch.Tensor, term: torch.Tensor, trunc: torch.Tensor,
        log_pi: torch.Tensor,
        total_reward: float = float('nan'),
    ) -> 'Trajectory':
        if not (
            state.shape[0] == action.shape[0] == reward.shape[0] ==
            next_state.shape[0] == term.shape[0] == trunc.shape[0] ==
            log_pi.shape[0]
        ):
            raise ValueError('All tensors must have the same first dimension.')
        B = state.shape[0]
        ret = cls.fixed_length(
            B, state_shape=state.shape[1:], action_shape=action.shape[1:],
            action_dtype=action.dtype, total_reward=total_reward
        )

        ret.state = state
        ret.action = action
        ret.reward = reward
        ret.next_state = next_state
        ret.term = term
        ret.trunc = trunc
        ret.log_pi = log_pi
        return ret

    @classmethod
    def concat(cls, trajectories: List['Trajectory']) -> 'Trajectory':
        if len(trajectories) == 0:
            raise ValueError('trajectories must not be empty.')
        state_shape = trajectories[0].state_shape
        action_shape = trajectories[0].action_shape
        action_dtype = trajectories[0].action_dtype
        total_reward = sum(t.total_reward for t in trajectories)
        data = torch.cat([t._data for t in trajectories], dim=0)
        return cls(
            _data=data, state_shape=state_shape, action_shape=action_shape,
            action_dtype=action_dtype, total_reward=total_reward
        )

    @classmethod
    def cat(cls, trajectories: List['Trajectory']) -> 'Trajectory':
        return cls.concat(trajectories)

    @classmethod
    def concatenate(cls, trajectories: List['Trajectory']) -> 'Trajectory':
        return cls.concat(trajectories)

    # PyTorch interface

    def clone(self) -> 'Trajectory':
        ret = copy.copy(self)
        ret._data = self._data.clone()
        return ret

    def to(self, device: torch.device | str) -> 'Trajectory':
        ret = copy.copy(self)
        ret._data = self._data.to(device).contiguous()
        return ret

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int | slice | torch.Tensor) -> 'Trajectory':
        ret = copy.copy(self)
        tgt = self._data[idx]
        if tgt.ndim == 1:
            tgt = tgt.unsqueeze(0)
        ret._data = tgt
        ret.total_reward = float('nan')
        return ret

    def __setitem__(self, idx: int | slice, values: tuple):
        state, action, reward, next_state, term, trunc, log_pi = values
        d = self._data
        sl = self.slice_dict
        d[idx, sl['state']] = state.reshape(d[idx, sl['state']].shape)
        d[idx, sl['action']] = action.reshape(d[idx, sl['action']].shape).to(torch.float32)
        d[idx, sl['reward']] = reward.reshape(d[idx, sl['reward']].shape)
        d[idx, sl['next_state']] = next_state.reshape(d[idx, sl['next_state']].shape)
        d[idx, sl['term']] = term.reshape(d[idx, sl['term']].shape)
        d[idx, sl['trunc']] = trunc.reshape(d[idx, sl['trunc']].shape)
        d[idx, sl['log_pi']] = log_pi.reshape(d[idx, sl['log_pi']].shape)

    @property
    def device(self) -> torch.device:
        return self._data.device

    def detach(self) -> 'Trajectory':
        ret = copy.copy(self)
        ret._data = self._data.detach()
        return ret

    def cpu(self) -> 'Trajectory':
        ret = copy.copy(self)
        ret._data = self._data.cpu().contiguous()
        return ret

    def cuda(self, device: torch.device | int | None = None) -> 'Trajectory':
        ret = copy.copy(self)
        ret._data = self._data.cuda(device).contiguous()
        return ret

    @staticmethod
    def _tau_step(
        _data: torch.Tensor, tau: int, gamma: float, slice_dict: Dict[str, slice]
    ):
        if tau <= 0:
            raise ValueError('tau must be positive.')
        if tau > _data.shape[-2]:
            raise ValueError('tau must be less than or equal to the length of the trajectory.')
        if gamma < 0 or gamma > 1:
            raise ValueError('gamma must be in [0, 1].')

        if tau == 1:
            return _data
        *B, _, _ = _data.shape
        kernel = torch.tensor(gamma, device=_data.device).pow(
            torch.arange(tau, dtype=torch.float, device=_data.device)
        )
        r = _data[..., slice_dict['reward']].squeeze(-1).unsqueeze(-2)
        if r.ndim == 2:
            r = r.unsqueeze(0)
        kernel = kernel.reshape(1, 1, tau)
        r_conv = torch.nn.functional.conv1d(r, kernel).reshape(*B, -1, 1)
        ret_length = r_conv.shape[-2]
        return torch.cat([
            _data[..., :ret_length, slice_dict['state']],
            _data[..., :ret_length, slice_dict['action']],
            r_conv.reshape(*B, ret_length, 1),
            _data[..., tau - 1:tau - 1 + ret_length, slice_dict['next_state']],
            _data[..., tau - 1:tau - 1 + ret_length, slice_dict['term']],
            _data[..., tau - 1:tau - 1 + ret_length, slice_dict['trunc']],
            _data[..., :ret_length, slice_dict['log_pi']]
        ], dim=-1)

    def tau_step(self, tau: int, gamma: float) -> 'Trajectory':
        if tau == 1:
            return self
        return Trajectory(
            _data=self._tau_step(self._data, tau, gamma, self.slice_dict),
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            action_dtype=self.action_dtype,
            total_reward=self.total_reward
        )

    @staticmethod
    def _shape_reward(
        _data: torch.Tensor,
        reward_shape: RewardMapping,
        state_shape: torch.Size | Tuple[int, ...],
        action_shape: torch.Size | Tuple[int, ...],
        slice_dict: Dict[str, slice]
    ):
        *B, L, _ = _data.shape
        term = _data[..., slice_dict['term']].squeeze(-1).to(torch.int64)
        trunc = _data[..., slice_dict['trunc']].squeeze(-1).to(torch.int64)
        mask = calculate_mask(term, trunc)

        _data[..., slice_dict['reward']][mask] = reward_shape(
            _data[..., slice_dict['state']].reshape(*B, L, *state_shape),
            _data[..., slice_dict['action']].reshape(*B, L, *action_shape),
            _data[..., slice_dict['reward']].squeeze(-1),
            _data[..., slice_dict['next_state']].reshape(*B, L, *state_shape),
            term,
            trunc
        )[mask].unsqueeze(-1)

    def shape_reward(self, reward_shape: RewardMapping):
        self._shape_reward(
            self._data, reward_shape, self.state_shape, self.action_shape,
            self.slice_dict
        )
