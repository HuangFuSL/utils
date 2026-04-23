from typing import Literal

from .. import BaseHook, Evaluator, LoopControl


class EvaluateHook(BaseHook):
    '''
    Hook to perform evaluation after per step or per epoch.

    Args:
        evaluator: An instance of Evaluator to perform evaluation.
        eval_interval_step: The interval (in steps) at which to perform evaluation.
        eval_interval_epoch: The interval (in epochs) at which to perform evaluation.
        copy: Whether to copy the model weights from the trainer to the evaluator before evaluation. Model and Device hooks must be registered if set to True.
    '''
    def __init__(
        self, evaluator: Evaluator,
        eval_interval_step: int = 0, eval_interval_epoch: int = 0,
        copy: bool = True
    ):
        self.evaluator = evaluator
        self.eval_interval_step = eval_interval_step
        self.eval_interval_epoch = eval_interval_epoch
        self.copy = copy
        self.hyperparams_copied = False

    def _eval(self) -> None:
        if self.copy:
           self.evaluator.model.load_state_dict(self.parent.model.state_dict())
        else:
            self.evaluator.model_context.device = self.parent.device
            self.evaluator.model_context.model = self.parent.model
        if not self.hyperparams_copied:
            self.evaluator.global_context.hyperparams.update(
                self.parent.global_context.hyperparams
            )
            self.hyperparams_copied = True
        self.evaluator.evaluate()
        new_metrics = self.evaluator.get_metrics().copy()
        if self.parent.global_context.metrics:
            if self.parent.global_context.metrics[-1][1] == self.parent.global_context.step:
                # Avoid duplicate metrics for the same step
                self.parent.global_context.metrics[-1][-1].update(new_metrics)
                return
        self.parent.global_context.metrics.append((
            self.parent.global_context.epoch,
            self.parent.global_context.step,
            new_metrics
        ))

    def after_step(self) -> LoopControl | None:
        if self.eval_interval_step > 0 and \
            self.parent.global_context.step % self.eval_interval_step == 0:
            self._eval()

    def finalize_epoch(self) -> LoopControl | None:
        if self.eval_interval_epoch > 0 and \
            self.parent.global_context.epoch % self.eval_interval_epoch == 0:
            self._eval()

class EarlyStoppingHook(BaseHook):
    '''
    Hook to stop training early based on evaluation metrics.
    '''
    def __init__(
        self, monitor: str, min_delta: float = 0.0,
        patience: int = 5, mode: Literal['min', 'max'] = 'min'
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        if mode == 'min':
            self.best_score = float('inf')
        elif mode == 'max':
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")
        self.num_bad_evals = 0
        self.num_evaluations = 0

    def check_step(self) -> LoopControl | None:
        if self.num_evaluations == len(self.parent.global_context.metrics):
            # No new evaluation come in
            return None
        # Update best metric
        _, _, new_metrics = self.parent.global_context.metrics[-1]

        if self.monitor not in new_metrics:
            raise ValueError(f'Monitored metric {self.monitor} not found in evaluation metrics.')
        if not isinstance(new_metrics[self.monitor], (int, float)):
            raise TypeError(
                f'Expected monitored metric {self.monitor} to be a number, '
                f'but got {type(new_metrics[self.monitor])}.'
            )

        if self.mode == 'min':
            improved = new_metrics[self.monitor] < self.best_score - self.min_delta
            self.best_score = min(self.best_score, new_metrics[self.monitor])
        else:
            improved = new_metrics[self.monitor] > self.best_score + self.min_delta
            self.best_score = max(self.best_score, new_metrics[self.monitor])
        # Check patience
        if not improved:
            self.num_bad_evals += 1
        else:
            self.num_bad_evals = 0
        self.num_evaluations = len(self.parent.global_context.metrics)
        if self.num_bad_evals >= self.patience:
            return LoopControl.SKIP_STAGE
