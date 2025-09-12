'''
rl.py - Utilities Components for Reinforcement Learning
'''
from __future__ import annotations

import abc
import copy
import functools
from typing import Any, Callable, Dict, Iterator, List, MutableMapping, Tuple

import gymnasium
import torch

from . import nn


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
        self.keys = ['state', 'action', 'reward', 'next_state', 'done']
        self.dtypes = [torch.float32 if k != 'action' else action_dtype for k in self.keys]
        self.data: MutableMapping[str, CircularTensor] = torch.nn.ModuleDict({
            k: CircularTensor(size, dtype=dtype)
            for k, dtype in zip(self.keys, self.dtypes)
        }) # type: ignore

    @property
    def length(self):
        ''' Get the current length of the buffer. '''
        return min(len(v) for v in self.data.values())

    def __getitem__(self, idx: torch.Tensor) -> List[torch.Tensor]:
        ''' Sample a batch of experiences from the buffer. '''
        return [self.data[k][idx] for k in self.keys]

    def get_batch(self, idx: torch.Tensor) -> List[torch.Tensor]:
        ''' Sample a batch of experiences from the buffer. '''
        return self[idx]

    def forward(self, idx: torch.Tensor) -> List[torch.Tensor]:
        ''' Sample a batch of experiences from the buffer. '''
        return self[idx]

    def sample_index(self, batch_size: int) -> torch.Tensor:
        ''' Randomly sample a batch of indices from the buffer. '''
        if self.length == 0:
            raise ValueError('Buffer is empty')
        if self.length < batch_size:
            raise ValueError('Not enough samples in buffer')
        return torch.randperm(self.length, dtype=torch.long)[:batch_size]

    def store(self, state, action, reward, next_state, done):
        ''' Store a batch of new experience in the buffer. '''
        B = None
        for k, v in zip(self.keys, [state, action, reward, next_state, done]):
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

    def store(self, state, action, reward, next_state, done):
        ''' Store a batch of new experience in the buffer. '''
        super().store(state, action, reward, next_state, done)
        B = state.shape[0]
        weight = self.p_max.unsqueeze(0).expand(B)
        self.weight.append(weight)


class BaseRLModel(nn.Module, abc.ABC):
    '''
    Abstract base class for reinforcement learning models.

    Args:
        shape_dim (Tuple[int, ...]): The shape dimensions of the input state.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''
    def __init__(self, shape_dim: Tuple[int, ...] | int, *, gamma: float = 0.99, tau: int = 1):
        super().__init__()
        if isinstance(shape_dim, int):
            shape_dim = (shape_dim,)
        if (
            (isinstance(shape_dim, int) and shape_dim <= 0) or
            (isinstance(shape_dim, tuple) and any(d <= 0 for d in shape_dim))
        ):
            raise ValueError('shape_dim must be positive.')
        self.shape_dim = shape_dim
        self.gamma = gamma
        self.tau = tau

    @abc.abstractmethod
    @torch.no_grad()
    def act(self, state: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        '''
        Policy for selecting actions based on the current state and exploration rate. By default, the policy takes exploration actions with a probability of :math:`\\varepsilon` and exploitation actions with a probability of :math:`1 - \\varepsilon`.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The selected actions.
        '''
        ...

class BasePolicyNetwork(BaseRLModel):
    '''
    Abstract base class for policy-based reinforcement learning models.

    Args:
        shape_dim (Tuple[int, ...]): The shape dimensions of the input state.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''
    def __init__(
        self,
        state_dim: Tuple[int, ...] | int,
        *, gamma: float = 0.99, tau: int = 1
    ):
        super().__init__(state_dim, gamma=gamma, tau=tau)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Return the action sampled from the policy given the state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The sampled action from the policy.
        '''
        dist = self(state)
        if dist.has_rsample:
            return dist.rsample()
        return dist.sample()

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


class TargetNetworkMixin(nn.Module):
    '''
    Mixin class for models with a target network.
    '''
    def __init__(self):
        super().__init__()
        self._target: nn.Module | None = None

    def setup_target(self):
        '''
        Create a target network by copying the current network. This method can only be called once.
        '''
        if self._target is not None:
            return
        self._target = copy.deepcopy(self)
        self._target.requires_grad_(False)
        self._target.setup_target = lambda: None # Disable further calls

    @property
    def target(self):
        '''
        The target network for the Q-learning algorithm. Used in double DQN. If not set, the current network is used.

        Returns:
            BaseQNetwork: The target network or self if not set.
        '''
        if self._target is not None:
            return self._target
        return self

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
        for p_t, p in zip(self._target.parameters(), self.parameters(), strict=True):
            p_t.mul_(1.0 - weight).add_(p, alpha=weight)

        for b_t, b in zip(self._target.buffers(), self.buffers(), strict=True):
            if torch.is_floating_point(b_t) and torch.is_floating_point(b):
                b_t.mul_(1.0 - weight).add_(b, alpha=weight)
            else:
                b_t.copy_(b)
        self.target.requires_grad_(False)


class BaseValueNetwork(BaseRLModel, TargetNetworkMixin):
    '''
    Abstract base class for Value Networks.

    One should either implement the ``forward`` method, or `V` method.

    Args:
        shape_dim (Tuple[int, ...]): The shape dimensions of the input state.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''
    def __init__(self, shape_dim: Tuple[int, ...] | int, *, gamma: float = 0.99, tau: int = 1):
        TargetNetworkMixin.__init__(self)
        BaseRLModel.__init__(self, shape_dim=shape_dim, gamma=gamma, tau=tau)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fwd_overridden = cls.forward is not BaseValueNetwork.forward
        v_overridden = getattr(cls, "V", BaseValueNetwork.V) is not BaseValueNetwork.V
        if not (fwd_overridden or v_overridden):
            raise TypeError(
                f'{cls.__name__} must override `forward` OR `V`.'
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Returns the value function V(s) for the given state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The value function V(s) for the given state.

        Shapes:
            - state: (\\*, (state_dims,))
            - output: (\\*)
        '''
        return self.V(state)

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

    def act(self, state: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError('Value networks do not have a policy for selecting actions.')

    def td_step(
        self,
        state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
        state_prime: torch.Tensor, is_terminal: torch.Tensor,
        a_prime: torch.Tensor | None = None
    ):
        '''
        The loss function for training the value network using the TD loss.

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
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): Not used.
            reward (torch.Tensor): The reward tensor over :math:`\\tau` steps.
            state_prime (torch.Tensor): The next state tensor.
            is_terminal (torch.Tensor): The terminal state indicator tensor.
            a_prime (torch.Tensor | None): Not used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The LHS and RHS of the TD loss.
        '''
        current = self(state)
        with torch.no_grad():
            mode = self.target.training
            self.target.eval()
            target = reward + (self.gamma ** self.tau) * self.target(state_prime) * (1 - is_terminal)
            self.target.train(mode)
        return current, target.detach()

class BaseQNetwork(BaseValueNetwork, abc.ABC):
    '''
    Abstract base class for Q-Networks.

    One should either implement the ``forward`` method, or both `Q` and `action_Q` methods.

    Args:
        shape_dim (Tuple[int, ...]): The shape dimensions of the input state.
        num_actions (int): The number of actions the agent can take.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''

    def __init__(self, shape_dim: Tuple[int, ...] | int, num_actions: int, *, gamma: float = 0.99, tau: int = 1):
        super(BaseQNetwork, self).__init__(shape_dim=shape_dim, gamma=gamma, tau=tau)
        if num_actions <= 0:
            raise ValueError('num_actions must be positive.')

        self.num_actions = num_actions

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        fwd_overridden = cls.forward is not BaseQNetwork.forward
        q_overridden = getattr(cls, "Q", BaseQNetwork.Q) is not BaseQNetwork.Q
        a_overridden = getattr(cls, "action_Q", BaseQNetwork.action_Q) is not BaseQNetwork.action_Q
        if not (fwd_overridden or (q_overridden and a_overridden)):
            raise TypeError(
                f'{cls.__name__} must override `forward` OR both `Q` and `action_Q`.'
            )

    # Value functions

    def Q(self, state: torch.Tensor) -> torch.Tensor:
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

    def action_Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Compute the action-value function Q(s, a) for a given state s and action a.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.

        Returns:
            torch.Tensor: The action-value function Q(s, a) for a given state s and action a.

        Shapes:
            - state: (\\*, (state_dims,))
            - action: (\\*)
            - output: (\\*)
        '''
        action = action.to(torch.long)
        return self(state, action)

    def V(self, state: torch.Tensor) -> torch.Tensor:
        return self.Q(state).amax(dim=-1)

    def A(self, state: torch.Tensor) -> torch.Tensor:
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
        return self.Q(state) - self.V(state)

    def action_A(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
        action = action.to(torch.long)
        return self.action_Q(state, action) - self.V(state)

    @torch.no_grad()
    def act(self, state: torch.Tensor, eps: float = 0.0, sample_wise: bool = True) -> torch.Tensor:
        '''
        Policy for selecting actions based on the current state and exploration rate. By default, the policy takes exploration actions with a probability of :math:`\\varepsilon` and exploitation actions with a probability of :math:`1 - \\varepsilon`.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The selected actions.
        '''
        state = state.to(self.device)
        if not sample_wise:
            if torch.rand(()) < eps:
                return self.explore(state)
            return self.exploit(state)
        A, B = self.explore(state), self.exploit(state)
        unif = torch.rand(state.shape[:-len(self.shape_dim)], device=self.device)
        return torch.where(unif < eps, A, B)

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
        target_shape = state.shape[:-len(self.shape_dim)]
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
        return torch.argmax(self.Q(state), dim=-1)

    # Training

    def td_step(
        self,
        state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
        state_prime: torch.Tensor, is_terminal: torch.Tensor,
        a_prime: torch.Tensor | None = None
    ):
        '''
        The loss function for training the Q-network using the TD loss.

        .. math::
            \\begin{aligned}
                \\text{LHS} &= Q(s, a) \\\\
                \\text{RHS} &= \\left\\{\\begin{aligned}
                    & r + Q_{\\text{target}}(s', \\arg\\max_{a'} Q(s', a')) & \\text{if not terminal} \\\\
                    & r & \\text{if terminal}
                \\end{aligned}\\right.
            \\end{aligned}

        The LHS and RHS can be optimized using ``MSELoss`` or ``SmoothL1Loss``.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.
            reward (torch.Tensor): The reward tensor over :math:`\\tau` steps.
            state_prime (torch.Tensor): The next state tensor.
            is_terminal (torch.Tensor): The terminal state indicator tensor.
            a_prime (torch.Tensor | None): The next action tensor for SARSA. If None, the action is selected using the exploitation policy (DQN).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The LHS and RHS of the TD loss.
        '''
        to = lambda x: x.to(self.device)
        state, action, reward, state_prime, is_terminal = map(
            to, (state, action, reward, state_prime, is_terminal)
        )
        action = action.to(torch.long)
        is_terminal = is_terminal.to(torch.float)

        if a_prime is None:
            a_prime = self.exploit(state_prime)
        with torch.no_grad():
            mode = self.target.training
            self.target.eval()
            target = reward + (self.gamma ** self.tau) \
                * self.target(state_prime, a_prime) * (1 - is_terminal)
            self.target.train(mode)

        current = self(state, action)
        return current, target

    # Forward
    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None):
        '''
        Returns the Q-values for the given state and action.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor | None): The input action tensor.

        Returns:
            torch.Tensor: The Q-values for the given state and action. If action is None, returns the Q-value over all actions.
        '''
        if action is None:
            return self.Q(state)
        action = action.to(torch.long)
        return self.action_Q(state, action)

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
):
    '''
    Simulates a single episode in the environment using the given model.

    Args:
        env (gymnasium.Env): The environment to simulate.
        model (BaseRLModel): The RL model to use for action selection.
        eps (float): The exploration rate.
        max_episode_steps (int | None): The maximum number of steps in the episode.
        reward_shape (Callable[[s, a, r, s_prime, terminated, truncated], torch.Tensor]): A function to adjust the reward based on the current reward, arrived state, terminal state, and time limit truncation. Returns the adjusted reward to be passed to the model.

    Returns:
        Tuple[List[torch.Tensor], float]: The collected experience :math:`(s, a, r, s', d)` tuple, and the overall *raw* reward.
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

    rewards = torch.tensor(0.0, device='cpu')
    steps = 0

    step_fn = torch_step(env)
    state, _ = env.reset()
    state = torch.as_tensor(state, device='cpu').float()
    if pin_memory:
        state = state.pin_memory()
    while True:
        action = model.act(state.to(device), **act_kwargs).detach().to('cpu', non_blocking=True)
        next_state, reward, done, time_exceed, _ = step_fn(action)
        rewards += reward
        reward = reward_shape(state, action, reward, next_state, done, time_exceed)

        result_s[steps, 0] = state
        result_a[steps] = action.unsqueeze(0)
        result_r[steps] = reward.unsqueeze(0)
        result_s[steps, 1] = next_state
        result_d[steps] = done.unsqueeze(0)

        steps += 1
        state = next_state

        if torch.any(done + time_exceed) or steps >= max_len:
            break

    s, a, r, s_prime, d = map(lambda x: x.to(device), [
        result_s[:steps, 0],  # state
        result_a[:steps], # action
        result_r[:steps], # reward
        result_s[:steps, 1], # next_state
        result_d[:steps] # done
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
        return [
            s[:ret_length], a[:ret_length],
            r_conv, s_prime[model.tau - 1:], d[model.tau - 1:]
        ], rewards.item()
    return [s, a, r, s_prime, d], rewards.item()
