from __future__ import annotations

import abc
import copy
from typing import Any, Iterator, List, Self, Tuple

import torch

from .. import nn
from .data import Trajectory

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

    def value_parameters(self) -> Iterator[torch.nn.Parameter]:
        '''
        Get the parameters of the value network.

        Returns:
            Iterator[torch.nn.Parameter]: An iterator over the parameters of the value network.
        '''
        if self._target is not None:
            target_ids = { id(p) for p in self.target.parameters() }
        else:
            target_ids = set()
        return (p for p in self.parameters() if id(p) not in target_ids)

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
    def update_target(self, weight: float | None = None, tau: float | None = None):
        '''
        Update the target network by copying the weights from the current network. No-op if the target network is not used.

        Args:
            weight (float, optional): The interpolation weight for the update. By default, it is 1.0.
        '''
        if weight is None:
            if tau is None:
                weight = 1.0
            else:
                weight = 1 - tau

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
        try:
            inner_ids = { id(p) for p in self.value_model.parameters() }
        except (NotImplementedError, AttributeError):
            inner_ids = set()
        if self._target is not None:
            inner_ids.update({ id(p) for p in self._target.parameters() })
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
            if hasattr(self.value_model, 'value_parameters'):
                return self.value_model.value_parameters()
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
        T = r.size(-1)
        device, dtype = r.device, r.dtype
        r_rev = torch.flip(r, dims=[-1])
        log_gamma = torch.tensor(
            self.gamma * lambda_, device=device, dtype=torch.float64
        ).log()
        log_w = log_gamma * torch.arange(T, device=device, dtype=torch.float64)

        inv_w = torch.exp(-log_w)
        w = torch.exp(log_w)
        y_rev = w * torch.cumsum(inv_w * r_rev.to(torch.float64), dim=-1)
        return torch.flip(y_rev, dims=[-1]).to(dtype)

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

class BaseDistributionalQNetwork(BaseQNetwork, abc.ABC):
    '''
    Abstract base class for Distributional DQN with discrete action space.

    One should implement the ``forward`` method to return un-softmaxed logits or log-softmaxed logits for all actions or specified actions, in shape (..., num_actions, num_atoms) or (..., num_atoms).

    Args:
        state_shape (Tuple[int, ...]): The shape dimensions of the input state.
        num_actions (int): The number of actions the agent can take.
        atoms (List[int | float] | torch.Tensor): The support of the value distribution.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.
    '''

    def __init__(
        self,
        state_shape: Tuple[int, ...] | int,
        num_actions: int,
        atoms: List[int | float] | torch.Tensor,
        *, gamma: float = 0.99, tau: int = 1
    ):
        super(BaseDistributionalQNetwork, self).__init__(
            state_shape=state_shape, num_actions=num_actions, gamma=gamma, tau=tau
        )
        self.atoms: torch.Tensor
        self.register_buffer(
            'atoms', torch.as_tensor(atoms, dtype=torch.float32).sort()[0],
        )
        self.num_atoms = self.atoms.numel()
        if self.num_atoms < 2:
            raise ValueError('atoms must contain at least two elements.')

    def Q_all_dist(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Compute the action-value distribution Q(s, a) for all actions a given state s.

        Args:
            state (torch.Tensor): The input state tensor.
        Returns:
            torch.Tensor: The action-value distribution Q(s, a) for all actions a.

        Shapes:
            - state: (*, (state_dims,))
            - output: (*, num_actions, num_atoms)
        '''
        return torch.softmax(self(state), dim=-1)

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
        return torch.einsum('...an,n->...a', self.Q_all_dist(state), self.atoms)

    def Q_dist(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        '''
        Compute the action-value distribution Q(s, a) for a given state s and action a.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.
        Returns:
            torch.Tensor: The action-value distribution Q(s, a) for a given state s and action a.

        Shapes:
            - state: (\\*, (state_dims,))
            - action: (\\*)
            - output: (\\*, num_atoms)
        '''
        return torch.softmax(self(state, action), dim=-1)

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
            - action: (\\*)
            - output: (\\*)
        '''
        q_dist = self.Q_dist(state, action)
        return torch.einsum('...n,n->...', q_dist, self.atoms)

    def V_dist(self, state: torch.Tensor) -> torch.Tensor:
        '''
        Compute the state-value distribution V(s) for a given state s.

        Args:
            state (torch.Tensor): The input state tensor.
        Returns:
            torch.Tensor: The state-value distribution V(s) for a given state s.

        Shapes:
            - state: (\\*, (state_dims,))
            - output: (\\*, num_atoms)
        '''
        q_all_dist = self.Q_all_dist(state)
        B = q_all_dist.shape[:-2]
        q_values = torch.einsum('...an,n->...a', q_all_dist, self.atoms)
        idxs = torch.argmax(q_values, dim=-1, keepdim=True).unsqueeze(-1).expand(*B, 1, self.num_atoms)
        v_dist = q_all_dist.gather(-2, idxs).squeeze(-2)
        return v_dist

    def _td_target(
        self, r: torch.Tensor, q: torch.Tensor, is_terminal: torch.Tensor,
        softmax: bool = True, eps: float = 1e-8
    ) -> torch.Tensor:
        '''
        Compute the transformed distribution :math:`r + \\gamma q` over the original support using linear projection.

        Args:
            q (torch.Tensor): The second distribution tensor of shape (..., num_atoms).
            r (torch.Tensor): The reward tensor of shape (...,).
            is_terminal (torch.Tensor): The terminal state indicator tensor of shape (...,).
            softmax (bool, optional): Whether to apply softmax to the input distributions. Defaults to True.
            eps (float, optional): A small value to avoid log(0). Defaults to 1e-8.

        Returns:
            torch.Tensor: The projected distribution tensor of shape (..., num_atoms).
        '''
        # Map r + gamma * q to the support of p using projection
        B, N = r.shape, self.atoms.numel()
        B_ones = [1] * len(B)

        inf = torch.tensor(float('inf'), device=q.device).view(1)
        atom = torch.cat([-inf, self.atoms, inf], dim=0)  # (num_atoms + 2,)
        gamma = (self.gamma ** self.tau * (1.0 - is_terminal)).unsqueeze(-1)
        q_field = r.unsqueeze(-1) + gamma * self.atoms.view(*B_ones, -1)

        if not softmax:
            one = torch.tensor(1.0, device=q.device)
            if torch.any(q < 0) or torch.any(q > 1) or \
                not torch.allclose(q.sum(dim=-1), one):
                raise ValueError('Invalid probability distributions.')
        q_prob_old = torch.softmax(q, dim=-1) if softmax else q

        q_prob = torch.zeros_like(q_prob_old)
        bucket_width = torch.diff(atom)  # N + 1 bins
        bucket_idx = torch.bucketize(q_field, atom, right=True) - 1 # 0..N
        # Here left_bucket_idx does not count the boundary inf

        right_bar_weight = (
            q_field - atom[bucket_idx]
        ) / bucket_width[bucket_idx]
        right_bar_weight = right_bar_weight.nan_to_num(1.0)  # inf / inf = nan case
        left_bar_weight = 1.0 - right_bar_weight

        q_prob.scatter_add_(
            -1, (bucket_idx - 1).clamp_min(0), q_prob_old * left_bar_weight
        )
        q_prob.scatter_add_(
            -1, bucket_idx.clamp_max(N - 1), q_prob_old * right_bar_weight
        )
        q_prob = q_prob.clamp_min(eps)
        q_prob /= q_prob.sum(dim=-1, keepdim=True)
        q_prob = q_prob.detach()

        return q_prob

    def _td_step(
        self, trajectory: Trajectory, a_prime: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns the current distribution log probability and the TD target distribution log probability for the given trajectory.

        Args:
            trajectory (Trajectory): A batch of trajectories.
            a_prime (torch.Tensor | None): The next action tensor. If provided, the model behaves like SARSA. If not provided, it behaves like Q-learning.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The current distribution log probability and the TD target distribution log probability.
        '''
        state, action, reward, state_prime, is_terminal, log_pi = trajectory
        current_logits = self(state, action).log_softmax(dim=-1)
        with torch.no_grad():
            mode = self.target.training
            self.target.eval()
            if a_prime is None:
                a_prime = self.exploit(state_prime)
            next_value = self.target(state_prime, a_prime)
            td_target_dist = self._td_target(reward, next_value, is_terminal)
            self.target.train(mode)
        return current_logits, td_target_dist.log()