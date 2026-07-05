import functools
from typing import Any, Callable, Dict, Tuple

import gymnasium
import torch
import numpy as np

from .model import BaseRLModel
from .data import Trajectory

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

        next_state, reward, done, time_exceed, _ = step_fn(action)
        rewards += reward
        reward = reward_shape(state, action, reward, next_state, done, time_exceed)

        result[steps] = (state, action, reward, next_state, done, log_pi)

        steps += 1
        state = next_state

        if torch.any(done + time_exceed) or steps >= max_len:
            break

    result = result[:steps].to(device)
    result.total_reward = rewards.item()

    if model.tau > 1:
        gamma = model.gamma
        kernel = torch.tensor(gamma, device=device).pow(
            torch.arange(model.tau, dtype=torch.float, device=device)
        )

        r = result.reward.reshape(1, 1, -1)
        kernel = kernel.reshape(1, 1, -1)
        r_conv = torch.nn.functional.conv1d(r, kernel).reshape(-1)
        ret_length = r_conv.shape[0]
        result = Trajectory.from_tensors(
            result.state[:ret_length], result.action[:ret_length],
            r_conv, result.next_state[model.tau - 1:],
            result.done[model.tau - 1:], result.log_pi[:ret_length],
            rewards.item()
        )
    return result
