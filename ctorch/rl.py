'''
rl.py - Utilities Components for Reinforcement Learning
'''
from __future__ import annotations

import abc
from typing import Tuple

import torch

from . import nn


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
        self.shape_dim = shape_dim
        self.gamma = gamma
        self.tau = tau

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

    @abc.abstractmethod
    @torch.no_grad()
    def explore(self, state: torch.Tensor) -> torch.Tensor:
        '''
        The exploration policy, defaulting to random actions.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The selected actions.
        '''
        ...

    @abc.abstractmethod
    @torch.no_grad()
    def exploit(self, state: torch.Tensor) -> torch.Tensor:
        '''
        The exploitation policy, selecting actions based on the current Q-values.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The selected actions.
        '''
        ...


class BaseQNetwork(BaseRLModel, abc.ABC):
    '''
    Abstract base class for Q-Networks.

    One should either implement the ``forward`` method, or both `Q` and `action_Q` methods.

    Args:
        shape_dim (Tuple[int, ...]): The shape dimensions of the input state.
        num_actions (int): The number of actions the agent can take.
        gamma (float, optional): The discount factor for future rewards, defaults to 0.99.
        tau (int, optional): The number of steps to look ahead for target updates, defaults to 1.

    Attributes
        - \\_target (BaseQNetwork | None): The target network for the Q-learning algorithm. Used for evaluating the TD target :math:`y = r + \\arg\\max_{a'} Q(s', a')`.
    '''

    def __init__(self, shape_dim: Tuple[int, ...] | int, num_actions: int, *, gamma: float = 0.99, tau: int = 1):
        super(BaseQNetwork, self).__init__(shape_dim=shape_dim, gamma=gamma, tau=tau)
        if num_actions <= 0:
            raise ValueError('num_actions must be positive.')
        if (
            (isinstance(shape_dim, int) and shape_dim <= 0) or
            (isinstance(shape_dim, tuple) and any(d <= 0 for d in shape_dim))
        ):
            raise ValueError('shape_dim must be positive.')

        self._target: BaseQNetwork | None = None
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

    @property
    def target(self):
        '''
        The target network for the Q-learning algorithm. Used in double DQN. If not set, the current network is used.

        Returns:
            BaseQNetwork: The target network.
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
        '''
        Compute the state-value function V(s) for a given state s.

        .. math::
            V(s) = \\max_a Q(s, a)

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The state-value function V(s) for a given state s.

        Shapes:
            - state: (\\*, (state_dims,))
            - output: (\\*)
        '''
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

    def loss(
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
