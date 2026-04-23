from typing import List

import torch

from .. import BaseHook, LoopControl


class GradientClipHook(BaseHook):
    '''
    Hook to clip gradients before optimizer step.
    '''

    def __init__(
        self, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False
    ):
        self.error_if_nonfinite = error_if_nonfinite
        self.max_norm = max_norm
        self.norm_type = norm_type

    def before_optimizer_step(self) -> LoopControl | None:
        torch.nn.utils.clip_grad_norm_(
            self.parent.model.parameters(), self.max_norm, self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite
        )


class DefaultZeroGradHook(BaseHook):
    '''
    Hook to zero gradients before forward pass. Supports gradient accumulation.

    Args:
        per_step: Number of steps to accumulate gradients before optimizer step.
    '''

    def __init__(self, per_step: int = 1):
        self.per_step = per_step  # Single step case

    def before_forward(self) -> LoopControl | None:
        if self.parent.global_context.backward % self.per_step == 0:
            for optim in self.parent.optimizer:
                optim.zero_grad()

    def check_optimizer_step(self) -> LoopControl | None:
        if self.parent.global_context.backward % self.per_step != 0:
            return LoopControl.SKIP_EVENT


class MaxEpochHook(BaseHook):
    '''
    Hook to stop training after a certain number of epochs.

    Args:
        num_epochs: The maximum number of epochs to train.
    '''

    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

    def check_epoch(self) -> LoopControl | None:
        if self.parent.global_context.epoch >= self.num_epochs:
            return LoopControl.SKIP_STAGE


class MaxStepHook(BaseHook):
    '''
    Hook to stop training after a certain number of steps.

    Args:
        num_steps: The maximum number of steps to train.
    '''

    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def check_step(self) -> LoopControl | None:
        if self.parent.global_context.step >= self.num_steps:
            return LoopControl.SKIP_STAGE

class WeightedLossHook(BaseHook):
    '''
    Hook to compute weighted sum of multiple losses.

    Args:
        loss_weights: The weights for each loss.
    '''
    def __init__(self, loss_weights: List[float]):
        self.loss_weights = loss_weights
        self.loss_weights_tensor = None

    def before_stage(self) -> LoopControl | None:
        if self.loss_weights_tensor is None:
            self.loss_weights_tensor = torch.tensor(self.loss_weights)
        self.loss_weights_tensor = self.loss_weights_tensor.to(
            self.parent.device)

    def after_forward(self) -> LoopControl | None:
        if self.loss_weights_tensor is None:
            self.loss_weights_tensor = torch.tensor(
                self.loss_weights, device=self.parent.device
            )
        losses = self.parent.step_context['losses']
        weights = self.loss_weights_tensor.to(losses.dtype)
        assert losses is not None
        self.parent.step_context['loss'] = (losses @ weights).mean()

