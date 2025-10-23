'''
rl.py - Utilities Components for Reinforcement Learning
'''
from __future__ import annotations

import abc
import copy
import dataclasses
import functools
from typing import Any, Callable, ClassVar, Dict, Iterator, List, MutableMapping, Self, Tuple

import gymnasium
import torch

from . import nn


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

    def set_weight(self, idx: torch.Tensor, weight: torch.Tensor):
        ''' Set the sampling weight for a batch of experiences. '''
        self.weight[idx] = weight.reshape(-1).abs().clamp_min(1e-6)

    def get_weight(self, idx: torch.Tensor) -> torch.Tensor:
        ''' Get the sampling weight for a batch of experiences. '''
        return self.weight[idx]

    def get_ipw(self, idx: torch.Tensor, beta: float, eps: float = 5e-4):
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


class BaseRLModel(nn.Module, abc.ABC):
    '''
    Abstract base class for reinforcement learning models.

    Args:
        state_shape (Tuple[int, ...]): The shape dimensions of the input state.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''
    def __init__(self, state_shape: Tuple[int, ...] | int, *, gamma: float = 0.99, tau: int = 1):
        super().__init__()
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        if (
            (isinstance(state_shape, int) and state_shape <= 0) or
            (isinstance(state_shape, tuple) and any(d <= 0 for d in state_shape))
        ):
            raise ValueError('state_shape must be positive.')
        self.state_shape = state_shape
        self.gamma = gamma
        self.tau = tau

    @abc.abstractmethod
    @torch.no_grad()
    def act(self, state: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Policy for selecting actions based on the current state and exploration rate. By default, the policy takes exploration actions with a probability of :math:`\\varepsilon` and exploitation actions with a probability of :math:`1 - \\varepsilon`.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the selected action and the log probability of the action.
        '''
        ...

    @abc.abstractmethod
    def loss(self) -> torch.Tensor:
        ...


class TargetNetworkMixin(nn.Module):
    '''
    Mixin class for models with a target network.
    '''

    def __init__(self):
        super().__init__()
        self._target: Self | None = None

    def setup_target(self):
        '''
        Create a target network by copying the current network. This method can only be called once.
        '''
        if self._target is not None:
            return
        self._target = copy.deepcopy(self)
        self._target.requires_grad_(False)
        self._target.eval()
        self._target._target = None  # Avoid nested targets
        self._target.setup_target = lambda: None  # Disable further calls

    @property
    def target(self) -> Self:
        '''
        The target network for the Q-learning algorithm. Used in double DQN. If not set, the current network is used.

        Returns:
            Module: The target network or self if not set.
        '''
        if self._target is not None:
            return self._target
        return self

    @torch.no_grad()
    def copy_params(self, src: torch.Tensor | None, tgt: torch.Tensor | None, weight: float = 1.0):
        if src is None or tgt is None or \
            tgt.shape != src.shape or tgt.dtype != src.dtype:
            return
        if torch.is_floating_point(tgt):
            tgt.data.mul_(1.0 - weight).add_(src.data, alpha=weight)
        else:
            tgt.data.copy_(src.data)

    @torch.no_grad()
    def update_target(self, weight: float = 1.0):
        '''
        Update the target network by copying the weights from the current network. No-op if the target network is not used.

        Args:
            weight (float, optional): The interpolation weight for the update. By default, it is 1.0.
        '''
        if not (0.0 < weight <= 1.0):
            raise ValueError('weight must be in (0.0, 1.0]')
        if self._target is None:
            return
        if weight == 1.0:
            self._target.load_state_dict(self.state_dict(), strict=False)
            return
        # Polyak averaging
        src_params = dict(self.named_parameters(remove_duplicate=False))
        tgt_params = dict(self._target.named_parameters(remove_duplicate=False))
        for src_name, src_param in src_params.items():
            tgt_param = tgt_params.get(src_name, None)
            self.copy_params(src_param, tgt_param, weight=weight)

        src_buffers = dict(self.named_buffers(remove_duplicate=False))
        tgt_buffers = dict(self._target.named_buffers(remove_duplicate=False))
        for src_name, src_buffer in src_buffers.items():
            tgt_buffer = tgt_buffers.get(src_name, None)
            self.copy_params(src_buffer, tgt_buffer, weight=weight)
        self.target.requires_grad_(False)


class BasePolicyNetwork(BaseRLModel, TargetNetworkMixin):
    '''
    Abstract base class for policy-based reinforcement learning models.

    Args:
        state_shape (Tuple[int, ...]): The shape dimensions of the input state.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''
    def __init__(
        self,
        state_shape: Tuple[int, ...] | int,
        *, gamma: float = 0.99, tau: int = 1
    ):
        super().__init__(state_shape, gamma=gamma, tau=tau)

    def setup_target(self):
        try:
            self.value_model
        except (NotImplementedError, AttributeError):
            super().setup_target()
            return
        raise RuntimeError(
            'To avoid duplicate targets, the target network should be set up '
            'before setting up value_model.'
        )

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Return the action sampled from the policy given the state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled action and its log probability.
        '''
        dist = self(state)
        if dist.has_rsample:
            ret = dist.rsample()
        else:
            ret = dist.sample()
        return ret, dist.log_prob(ret)

    def policy_parameters(self) -> Iterator[torch.nn.Parameter]:
        '''
        Get the parameters of the policy network.

        Returns:
            Iterator[torch.nn.Parameter]: An iterator over the parameters of the policy network.
        '''
        inner_ids = { id(p) for p in self.value_parameters() }
        if not inner_ids:
            return self.parameters()
        return (p for p in self.parameters() if id(p) not in inner_ids)

    @property
    def value_model(self) -> 'BaseValueNetwork':
        '''
        Get the state value sub-network. If not implemented, raises NotImplementedError.
        '''
        raise NotImplementedError('This model does not have a value function.')

    def value_parameters(self) -> Iterator[torch.nn.Parameter]:
        '''
        Get the parameters of the value network.

        Returns:
            Iterator[torch.nn.Parameter]: An iterator over the parameters of the value network.
        '''
        try:
            return self.value_model.parameters()
        except NotImplementedError:
            return iter(())

    @abc.abstractmethod
    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        '''
        Forward pass to compute the action distribution given the state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.distributions.Distribution: The action distribution given the state.
        '''
        raise NotImplementedError

    def log_pi(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Return the log probability of the action given the state.
        '''
        return self(state).log_prob(action)

    def pi(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Return the action distribution probability given the state.
        '''
        return torch.exp(self.log_pi(state, action))

    def cumulative_reward(self, r: torch.Tensor, lambda_: float = 1) -> torch.Tensor:
        '''
        Calculate the cumulative reward for a trajectory, or the GAE(lambda) estimate.

        Args:
            r (torch.Tensor): The reward tensor of shape (L,), for GAE, r is the state-value TD error :math:`r_t + \\gamma V(s_{t+1}) - V(s_t)`.
            lambda\\_ (float, optional): The lambda parameter for GAE. Defaults to 1 for cumulative reward.

        Returns:
            torch.Tensor: The cumulative reward or GAE estimate of shape (L,).

        Shapes:

            - r: (L,)
            - output: (L,)
        '''
        gamma = self.gamma * lambda_
        y = torch.zeros_like(r)
        acc = torch.zeros_like(r[..., 0])
        T = r.size(-1)
        for t in range(T - 1, -1, -1):
            acc = r[..., t] + gamma * acc
            y[..., t] = acc
        return y

    def normalize_trajectory(self, r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (r - r.mean().detach()) / (r.std(unbiased=False).detach() + eps)


class BaseValueNetwork(BaseRLModel, TargetNetworkMixin):
    '''
    Abstract base class for Value Networks.

    One should either implement the ``forward`` method, or `V` method.

    Args:
        state_shape (Tuple[int, ...]): The shape dimensions of the input state.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''
    def __init__(self, state_shape: Tuple[int, ...] | int, *, gamma: float = 0.99, tau: int = 1):
        TargetNetworkMixin.__init__(self)
        BaseRLModel.__init__(self, state_shape=state_shape, gamma=gamma, tau=tau)

    @abc.abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Forward pass of the value network. The implementation should either implement the state-value function V(s) or the action-value function Q(s, a).

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor | None): The input action tensor. If provided, the network should compute the action-value function Q(s, a). If not provided, the network should compute the state-value function V(s).

        Returns:
            torch.Tensor: The value function V(s) for the given state.

        Shapes:
            - state: (\\*, (state_dims,))
            - action: (\\*, (action_dims,)), (\\*), or None
            - output: (\\*)
        '''
        raise NotImplementedError

    def V(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Compute the state-value function V(s) for a given state s.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The state-value function V(s) for a given state s.

        Shapes:
            - state: (\\*, (state_dims,))
            - output: (\\*)
        '''
        return self(state)

    def Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Compute the action-value function Q(s, a) for a given state s and action a.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.
        Returns:
            torch.Tensor: The action-value function Q(s, a) for a given state s and action a.

        Shapes:
            - state: (\\*, (state_dims,))
            - action: (\\*, (action_dims,)) or (\\*)
            - output: (\\*)
        '''
        return self(state, action)

    def A(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Compute the advantage function A(s, a) for a given state s and action a.

        .. math::
            A(s, a) = Q(s, a) - V(s)

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.

        Returns:
            torch.Tensor: The advantage function A(s, a) for a given state s and action a.

        Shapes:
            - state: (\\*, (state_dims,))
            - action: (\\*)
            - output: (\\*)
        '''
        return self.Q(state, action) - self.V(state)

    def act(self, state: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('Value networks do not have a policy for selecting actions.')

    def _td_step(
        self, trajectory: Trajectory, a_prime: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state, action, reward, state_prime, is_terminal, log_pi = trajectory
        current = self.V(state) if a_prime is None else self.Q(state, action)
        with torch.no_grad():
            mode = self.target.training
            self.target.eval()
            next_step = self.target.V(state_prime) if a_prime is None \
                else self.target.Q(state_prime, a_prime)
            target = reward + (self.gamma ** self.tau) * next_step * (1 - is_terminal)
            self.target.train(mode)
        return current, target.detach()

    def value_td_step(self, trajectory: Trajectory):
        '''
        The loss function for training the value network using the TD loss. If a_prime is provided, the action-value function Q(s, a) is used. Otherwise, the state-value function V(s) is used.

        .. math::
            \\begin{aligned}
                \\text{LHS} &= V(s) \\\\
                \\text{RHS} &= \\left\\{\\begin{aligned}
                    & r + V(s') & \\text{if not terminal} \\\\
                    & r & \\text{if terminal}
                \\end{aligned}\\right.
            \\end{aligned}

        The LHS and RHS can be optimized using ``MSELoss`` or ``SmoothL1Loss``.

        Args:
            trajectory (Trajectory): A batch of trajectories.
            a_prime (torch.Tensor | None): The next action tensor. If provided, the action-value function Q(s, a) is used. If not provided, the state-value function V(s) is used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The LHS and RHS of the TD loss.
        '''
        return self._td_step(trajectory, a_prime=None)

    def action_td_step(
        self, trajectory: Trajectory, a_prime: torch.Tensor
    ):
        '''
        The loss function for training the value network using the TD loss. If a_prime is provided, the action-value function Q(s, a) is used. Otherwise, the state-value function V(s) is used.

        .. math::
            \\begin{aligned}
                \\text{LHS} &= V(s) \\\\
                \\text{RHS} &= \\left\\{\\begin{aligned}
                    & r + V(s') & \\text{if not terminal} \\\\
                    & r & \\text{if terminal}
                \\end{aligned}\\right.
            \\end{aligned}

        The LHS and RHS can be optimized using ``MSELoss`` or ``SmoothL1Loss``.

        Args:
            trajectory (Trajectory): A batch of trajectories.
            a_prime (torch.Tensor | None): The next action tensor. If provided, the action-value function Q(s, a) is used. If not provided, the state-value function V(s) is used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The LHS and RHS of the TD loss.
        '''
        return self._td_step(trajectory, a_prime=a_prime)

class BaseQNetwork(BaseValueNetwork, abc.ABC):
    '''
    Abstract base class for DQN with discrete action space.

    One should either implement the ``forward`` method to return Q-values for all actions or specified actions.

    Args:
        state_shape (Tuple[int, ...]): The shape dimensions of the input state.
        num_actions (int): The number of actions the agent can take.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''

    def __init__(self, state_shape: Tuple[int, ...] | int, num_actions: int, *, gamma: float = 0.99, tau: int = 1):
        super(BaseQNetwork, self).__init__(state_shape=state_shape, gamma=gamma, tau=tau)
        if num_actions <= 0:
            raise ValueError('num_actions must be positive.')

        self.num_actions = num_actions

    # Value functions

    def Q_all(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Compute the action-value function Q(s, a) for all actions a given state s.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The action-value function Q(s, a) for all actions a.

        Shapes:
            - state: (*, (state_dims,))
            - output: (*, num_actions)
        '''
        return self(state)

    def V(self, state: torch.Tensor) -> torch.Tensor:
        return self.Q_all(state).amax(dim=-1)

    def A_all(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Compute the advantage function A(s, a) for a given state s and action a.

        .. math::
            A(s, a) = Q(s, a) - V(s)

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The advantage function A(s, a) for a given state s and all actions.

        Shapes:
            - state: (\\*, (state_dims,))
            - action: (\\*)
            - output: (\\*)
        '''
        return self.Q_all(state) - self.V(state)

    @torch.no_grad()
    def act(
        self, state: torch.Tensor, eps: float = 0.0, sample_wise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Policy for selecting actions based on the current state and exploration rate. By default, the policy takes exploration actions with a probability of :math:`\\varepsilon` and exploitation actions with a probability of :math:`1 - \\varepsilon`.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The selected actions.
        '''
        state = state.to(self.device)
        A, B = self.explore(state), self.exploit(state)
        # For greedy actions, prob is 1 - eps + eps / num_actions
        # For random actions, prob is eps / num_actions
        if not sample_wise:
            if torch.rand(()) < eps:
                ret = A
            else:
                ret = B
        unif = torch.rand(state.shape[:-len(self.state_shape)], device=self.device)
        ret = torch.where(unif < eps, A, B)
        prob = (eps / self.num_actions) + (1 - eps) * (ret == B).to(torch.float32)
        return ret, prob.log()

    # Policies
    @torch.no_grad()
    def explore(self, state: torch.Tensor) -> torch.Tensor:
        '''
        The exploration policy, defaulting to random actions.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The selected actions.
        '''
        target_shape = state.shape[:-len(self.state_shape)]
        return torch.randint(self.num_actions, target_shape, device=self.device)

    @torch.no_grad()
    def exploit(self, state: torch.Tensor) -> torch.Tensor:
        '''
        The exploitation policy, selecting actions based on the current Q-values.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The selected actions.
        '''
        state = state.to(self.device)
        return torch.argmax(self.Q_all(state), dim=-1)

    # Training

    def td_step(
        self, trajectory: Trajectory, a_prime: torch.Tensor | None = None
    ):
        if a_prime is None:
            a_prime = self.exploit(trajectory.next_state)
        return super().action_td_step(trajectory, a_prime)

    # Forward
    @abc.abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None):
        '''
        Returns the Q-values for the given state and action.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor | None): The input action tensor.

        Returns:
            torch.Tensor: The Q-values for the given state and action. If action is None, returns the Q-value over all actions.
        '''
        raise NotImplementedError

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
        action_np = action.numpy(force=True).astype(env.action_space.dtype)
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
