import dataclasses
import functools
import math
from typing import Any, Callable, Dict, Tuple

import gymnasium
import torch
import numpy as np

from .model import BaseRLModel
from .data import Trajectory, RewardMapping, _default_shape

def torch_step(env: gymnasium.Env, device: torch.device | str = 'cpu'):
    '''
    A wrapper for the environment step function to convert actions from torch tensors and results to torch tensors.

    Args:
        env (gymnasium.Env): The environment to wrap.

    Returns:
        Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]: The wrapped ``env.step`` function.
    '''
    @functools.wraps(env.step)
    def _wrapper(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_np = action.numpy().astype(env.action_space.dtype)
        *ret, info = env.step(action_np)
        return [torch.as_tensor(r, device=device).float() for r in ret] + [info] # type: ignore
    return _wrapper

@dataclasses.dataclass(init=False)
class EnvironmentInfo:
    env_fn: Callable[[], gymnasium.Env]

    state_shape: Tuple[int, ...]
    state_numel: int
    state_dtype: torch.dtype
    state_dtype_np: np.dtype

    action_shape: Tuple[int, ...]
    action_dtype: torch.dtype
    action_dtype_np: np.dtype
    action_numel: int

    buffer_dim: int
    slice_dict: Dict[str, slice]

    max_len: int

    def __init__(
        self,
        env: gymnasium.Env | None = None,
        env_fn: Callable[[], gymnasium.Env] | None = None,
        max_len: int | None = None
    ):
        if env is not None:
            if env_fn is not None:
                raise ValueError("Provide either env or env_fn, not both.")
            self.env_fn = lambda: env # type: ignore
        elif env_fn is not None:
            self.env_fn = env_fn
        else:
            raise ValueError("Provide either env or env_fn.")

        test_env = self.env_fn()

        if test_env.observation_space is None:
            raise ValueError('State shape is None.')
        if test_env.action_space is None:
            raise ValueError('Action shape is None.')
        if test_env.action_space.dtype is None:
            raise ValueError('Action dtype is None.')


        self.state_shape = test_env.observation_space.shape # type: ignore
        self.state_dtype_np = test_env.observation_space.dtype # type: ignore
        if np.issubdtype(self.state_dtype_np, np.integer):
            self.state_dtype = torch.long
        else:
            self.state_dtype = torch.float32
        self.state_numel = math.prod(self.state_shape)

        self.action_shape = test_env.action_space.shape # type: ignore
        self.action_dtype_np = test_env.action_space.dtype # type: ignore
        if np.issubdtype(self.action_dtype_np, np.integer):
            self.action_dtype = torch.long
        else:
            self.action_dtype = torch.float32
        self.action_numel = math.prod(self.action_shape)

        if max_len is None:
            self.max_len = test_env._max_episode_steps
        else:
            self.max_len = max_len

        sample_traj = Trajectory.fixed_length(
            self.max_len, self.state_shape, self.action_shape, self.action_dtype
        )
        self.buffer_dim = sample_traj._data.shape[-1]
        self.slice_dict = sample_traj.slice_dict

        if env is None:
            test_env.close()

@torch.no_grad()
def run_episode(
    env: gymnasium.Env, model: BaseRLModel,
    max_episode_steps: int | None = None,
    reward_shape: RewardMapping = _default_shape,
    **act_kwargs: Any
) -> Trajectory:
    '''
    Simulates a single episode in the environment using the given model.

    Args:
        env (gymnasium.Env): The environment to simulate.
        model (BaseRLModel): The RL model to use for action selection.
        eps (float): The exploration rate.
        max_episode_steps (int | None): The maximum number of steps in the episode.
        reward_shape (Callable[[s, a, r, s_prime, terminated, truncated], torch.Tensor]): A function to adjust the reward based on the current reward, arrived state, terminal state, and time limit truncation. Returns the adjusted reward to be passed to the model.

    Returns:
        Trajectory: The collected trajectory.
    '''
    # Parse env
    s_shape: Tuple[int, ...] = env.observation_space.shape # type: ignore
    a_shape: Tuple[int, ...] = env.action_space.shape # type: ignore
    a_dtype: np.dtype = env.action_space.dtype # type: ignore
    if max_episode_steps is None:
        max_len = env._max_episode_steps # type: ignore
    else:
        max_len = max_episode_steps
    if s_shape is None:
        raise ValueError('State shape is None.')
    if a_shape is None:
        raise ValueError('Action shape is None.')
    if a_dtype is None:
        raise ValueError('Action dtype is None.')

    device = model.device
    pin_memory = torch.cuda.is_available() and model.device.type == 'cuda'
    action_dtype = torch.long if np.issubdtype(a_dtype, np.integer) else torch.float32
    result = Trajectory.fixed_length(
        max_len, state_shape=s_shape, action_shape=a_shape,
        action_dtype=action_dtype, total_reward=float('nan'),
        pin_memory=pin_memory
    )

    rewards = torch.tensor(0.0, device='cpu')
    steps = 0

    step_fn = torch_step(env)
    state, _ = env.reset()
    state = torch.as_tensor(state, device='cpu').float()
    if pin_memory:
        state = state.pin_memory()
    while True:
        action, log_pi = model.act(state.to(device), **act_kwargs)
        action = action.detach().to('cpu', non_blocking=True)
        log_pi = log_pi.detach().to('cpu', non_blocking=True)

        next_state, reward, term, trunc, _ = step_fn(action)
        rewards += reward
        result[steps] = (state, action, reward, next_state, term, trunc, log_pi)

        steps += 1
        state = next_state

        if torch.any(term + trunc * 2) or steps >= max_len:
            break

    result = result[:steps].to(device)
    result.total_reward = rewards.item()

    result.shape_reward(reward_shape)
    return result.tau_step(model.tau, model.gamma)
