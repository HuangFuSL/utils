import dataclasses
import functools
from typing import Any, Callable, ClassVar, Dict, List, Tuple

import gymnasium
import torch

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


@torch.inference_mode()
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
    s_shape = env.observation_space.shape
    a_shape = env.action_space.shape
    a_dtype = env.action_space.dtype
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
    result_s = torch.zeros(max_len, 2, *s_shape, dtype=torch.float, pin_memory=pin_memory)
    result_a = torch.zeros(max_len, *a_shape, dtype=torch.float, pin_memory=pin_memory)
    result_r = torch.zeros(max_len, dtype=torch.float, pin_memory=pin_memory)
    result_d = torch.zeros(max_len, dtype=torch.float, pin_memory=pin_memory)
    result_pi = torch.zeros(max_len, dtype=torch.float, pin_memory=pin_memory)

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

        result_s[steps, 0] = state
        result_a[steps] = action.unsqueeze(0)
        result_r[steps] = reward.unsqueeze(0)
        result_s[steps, 1] = next_state
        result_d[steps] = done.unsqueeze(0)
        result_pi[steps] = log_pi.unsqueeze(0)

        steps += 1
        state = next_state

        if torch.any(done + time_exceed) or steps >= max_len:
            break

    s, a, r, s_prime, d, log_pi = map(lambda x: x.to(device), [
        result_s[:steps, 0],  # state
        result_a[:steps], # action
        result_r[:steps], # reward
        result_s[:steps, 1], # next_state
        result_d[:steps], # done
        result_pi[:steps]
    ])

    if model.tau > 1:
        gamma = model.gamma
        kernel = torch.tensor(gamma, device=device).pow(
            torch.arange(model.tau, dtype=torch.float, device=device)
        )

        r = r.reshape(1, 1, -1)
        kernel = kernel.reshape(1, 1, -1)
        r_conv = torch.nn.functional.conv1d(r, kernel).reshape(-1)
        ret_length = r_conv.shape[0]
        return Trajectory(
            s[:ret_length], a[:ret_length],
            r_conv, s_prime[model.tau - 1:], d[model.tau - 1:], \
            log_pi[:ret_length], rewards.item()
        )
    return Trajectory(s, a, r, s_prime, d, log_pi, rewards.item())
