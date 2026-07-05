import dataclasses
from typing import ClassVar, List

import torch


@dataclasses.dataclass(slots=True)
class Trajectory():
    '''
    A batch of states, actions, and rewards collected from the environment.

    Args:
        state (torch.Tensor): The states in the trajectory of shape (L, *state_shape).
        action (torch.Tensor): The actions taken in the trajectory of shape (L, *action_shape).
        reward (torch.Tensor): The rewards received in the trajectory of shape (L,).
        next_state (torch.Tensor): The next states in the trajectory of shape (L, *state_shape).
        done (torch.Tensor): The done flags in the trajectory of shape (L,).
        log_pi (torch.Tensor): The log probabilities of the actions taken in the trajectory of shape (L,).
        total_reward (float): The total reward of the trajectory. `nan` if the batch is not a trajectory.
    '''
    tensor_fields: ClassVar[List[str]] = dataclasses.field(init=False, repr=False, default=[
        'state', 'action', 'reward', 'next_state', 'done', 'log_pi'
    ])
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    log_pi: torch.Tensor
    total_reward: float

    def clone(self):
        for k in self.tensor_fields:
            v = getattr(self, k)
            if not isinstance(v, torch.Tensor):
                raise TypeError(f'Field {k} must be a torch.Tensor, got {type(v)}')
            setattr(self, k, v.clone())

    def get(self, key: str) -> torch.Tensor:
        if key not in self.tensor_fields:
            raise ValueError(f'Invalid key: {key}')
        return getattr(self, key)

    def __iter__(self):
        for k in self.tensor_fields:
            yield getattr(self, k)

    def __len__(self):
        return self.state.shape[0]

    def to(self, device: torch.device | str):
        for k in self.tensor_fields:
            v = getattr(self, k)
            if not isinstance(v, torch.Tensor):
                raise TypeError(f'Field {k} must be a torch.Tensor, got {type(v)}')
            setattr(self, k, v.to(device))
        return self