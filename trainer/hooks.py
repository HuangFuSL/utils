'''
`utils.trainer.hooks` - Trainer hooks for various training functionalities.
'''

from typing import Any, Dict, List, Literal
import numpy as np
import torch
import os
import traceback
import shutil
import warnings

from torch.profiler import profile

from . import BaseHook, DeferHookExec, Trainer, LoopControl, Evaluator
from ..ctorch.device import get_best_device


class InitModelHook(BaseHook):
    '''
    Hook to initialize the model before training.

    Args:
        model_cls: The class of the model to be initialized.
        *args: Positional arguments to be passed to the model constructor.
        **kwargs: Keyword arguments to be passed to the model constructor.
    '''

    def __init__(self, model_cls, *args, **kwargs):
        self.model_cls = model_cls
        self.args = args
        self.kwargs = kwargs or {}

    def before_stage(self, trainer: 'Trainer') -> LoopControl | None:
        model = self.model_cls(*self.args, **self.kwargs)
        trainer.model_context.model = model


class InitOptimizerHook(BaseHook):
    '''
    Hook to initialize the optimizer before training.

    Args:
        optim_cls: The class of the optimizer to be initialized.
        **kwargs: Keyword arguments to be passed to the optimizer constructor.
    '''

    def __init__(self, optim_cls, **kwargs):
        self.optim_cls = optim_cls
        self.kwargs = kwargs

    def before_stage(self, trainer: 'Trainer') -> LoopControl | None:
        trainer.device # DeferHookExec if device is not initialized
        trainer.model_context.optimizer = [self.optim_cls(
            trainer.model.parameters(), **self.kwargs
        )]


class InitLRSchedulerHook(BaseHook):
    '''
    Hook to initialize the learning rate scheduler before training. The hook will be executed
    after the optimizer is initialized.
    '''

    def __init__(self, lr_scheduler_cls, **kwargs):
        self.lr_scheduler_cls = lr_scheduler_cls
        self.kwargs = kwargs

    def before_stage(self, trainer: 'Trainer') -> LoopControl | None:
        if len(trainer.optimizer) != 1:
            raise ValueError(
                'Length of lr_scheduler_cls must match length of optimizer.'
            )
        trainer.model_context.lr_scheduler = [
            self.lr_scheduler_cls(trainer.optimizer[0], **self.kwargs)
        ]


class InitDataloaderHook(BaseHook):
    '''
    Hook to initialize the dataloader before training.

    Args:
        dataloader_cls: The class of the dataloader to be initialized.
        *args: Positional arguments to be passed to the dataloader constructor.
        **kwargs: Keyword arguments to be passed to the dataloader constructor.
    '''

    def __init__(self, dataloader_cls, *args, **kwargs):
        self.dataloader_cls = dataloader_cls
        self.args = args
        self.kwargs = kwargs or {}

    def before_stage(self, trainer: 'Trainer') -> LoopControl | None:
        dataloader = self.dataloader_cls(*self.args, **self.kwargs)
        trainer.model_context.dataloader = dataloader


class InitDeviceHook(BaseHook):
    '''
    Hook to initialize the device. If no device is specified, the best available device will be used.

    Args:
        device: The device to be used.
    '''

    def __init__(self, device: torch.device | str | None = None):
        if device is None:
            device = get_best_device()
        self.device = torch.device(device)

    def before_stage(self, trainer: 'Trainer') -> LoopControl | None:
        trainer.model_context.device = self.device

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

    def before_optimizer_step(self, trainer: 'Trainer') -> LoopControl | None:
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), self.max_norm, self.norm_type,
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

    def before_forward(self, trainer: 'Trainer') -> LoopControl | None:
        if trainer.global_context.backward % self.per_step == 0:
            for optim in trainer.optimizer:
                optim.zero_grad()

    def check_optimizer_step(self, trainer: 'Trainer') -> LoopControl | None:
        if trainer.global_context.backward % self.per_step != 0:
            return LoopControl.SKIP_EVENT

class SuppressExceptionHook(BaseHook):
    def on_exception(self, trainer: 'Trainer') -> LoopControl | None:
        return LoopControl.SKIP_STEP


class MaxEpochHook(BaseHook):
    '''
    Hook to stop training after a certain number of epochs.

    Args:
        num_epochs: The maximum number of epochs to train.
    '''

    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

    def check_epoch(self, trainer: Trainer) -> LoopControl | None:
        if trainer.global_context.epoch >= self.num_epochs:
            return LoopControl.SKIP_STAGE


class MaxStepHook(BaseHook):
    '''
    Hook to stop training after a certain number of steps.

    Args:
        num_steps: The maximum number of steps to train.
    '''

    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def check_step(self, trainer: Trainer) -> LoopControl | None:
        if trainer.global_context.step >= self.num_steps:
            return LoopControl.SKIP_STAGE


class TqdmHook(BaseHook):
    '''
    Hook to display a progress bar using tqdm during training.

    Args:
        unit: The unit of progress to track. Can be 'data_step', 'update_step', or 'epoch'.
    '''

    def __init__(
        self, unit: Literal['data_step', 'update_step', 'epoch'] = 'data_step',
        metrics_keys: List[str] | None = None, floatfmt: str = '.4g'
    ):
        self.unit = unit
        self.metrics_keys = metrics_keys
        self.floatfmt = floatfmt
        self._last_metrics_idx = -1

    def _refresh_metrics_postfix(self, trainer: Trainer) -> None:
        if self.metrics_keys is None or not trainer.global_context.metrics:
            return
        metrics_log = trainer.global_context.metrics
        if len(metrics_log) - 1 == self._last_metrics_idx:
            return
        self._last_metrics_idx = len(metrics_log) - 1

        _, _, metrics = metrics_log[self._last_metrics_idx]
        if not isinstance(metrics, dict) or not metrics:
            return
        if self.metrics_keys is not None:
            metrics = {k: metrics[k] for k in self.metrics_keys if k in metrics}

        postfix = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                postfix[k] = format(v, self.floatfmt)
            else:
                postfix[k] = v

        self.tqdm.set_postfix(postfix, refresh=False)

    def before_stage(self, trainer: Trainer) -> LoopControl | None:
        import tqdm
        self.tqdm = tqdm.tqdm(unit=' ' + self.unit)

    def before_step(self, trainer: Trainer) -> LoopControl | None:
        if self.unit == 'data_step':
            self.tqdm.update(1)
            self._refresh_metrics_postfix(trainer)

    def after_optimizer_step(self, trainer: Trainer) -> LoopControl | None:
        if self.unit == 'update_step':
            self.tqdm.update(1)
            self._refresh_metrics_postfix(trainer)

    def after_epoch(self, trainer: Trainer) -> LoopControl | None:
        if self.unit == 'epoch':
            self.tqdm.update(1)
            self._refresh_metrics_postfix(trainer)

    def finalize_stage(self, trainer: Trainer) -> LoopControl | None:
        if hasattr(self, "tqdm"):
            self.tqdm.close()


class WeightedLossHook(BaseHook):
    '''
    Hook to compute weighted sum of multiple losses.

    Args:
        loss_weights: The weights for each loss.
    '''
    def __init__(self, loss_weights: List[float]):
        self.loss_weights = torch.tensor(loss_weights)

    def before_stage(self, trainer: Trainer) -> LoopControl | None:
        self.loss_weights = self.loss_weights.to(trainer.device)

    def after_forward(self, trainer: Trainer) -> LoopControl | None:
        losses = trainer.step_context['losses']
        weights = self.loss_weights.to(losses.dtype)
        assert losses is not None
        trainer.step_context['loss'] = (losses @ weights).mean()

class ProfileHook(BaseHook):
    '''
    Hook to profile each training step using torch.profiler.

    Args:
        activities: List of ProfilerActivity to be profiled.
        record_shapes: Whether to record shapes of the tensors.
        with_stack: Whether to record stack traces.
    '''

    def __init__(
        self, num_steps: int, export_path: str | None = None,
        **kwargs
    ):
        self.num_steps = num_steps
        self.export_path = export_path
        self.prof_kwargs = kwargs

    def before_stage(self, trainer: Trainer) -> LoopControl | None:
        trainer.global_context['profile'] = profile(
            experimental_config=torch._C._profiler._ExperimentalConfig(
                verbose=True),
            **self.prof_kwargs
        )
        trainer.global_context['profile'].__enter__()

    def check_step(self, trainer: Trainer) -> LoopControl | None:
        if trainer.global_context.step >= self.num_steps:
            return LoopControl.SKIP_STAGE

    def finalize_step(self, trainer: Trainer) -> LoopControl | None:
        trainer.global_context['profile'].step()

    def finalize_stage(self, trainer: Trainer) -> LoopControl | None:
        exc = trainer.exception
        exc_type = None if exc is None else type(exc)
        tb = None if exc is None else exc.__traceback__
        trainer.global_context['profile'].__exit__(exc_type, exc, tb)
        if self.export_path is not None:
            trainer.global_context['profile'].export_chrome_trace(f"{self.export_path}/trace.json")

class LoadCheckpointHook(BaseHook):
    '''
    Hook for loading model checkpoints **for evaluation**.

    Args:
        checkpoint_path: The path of the checkpoint to be loaded.
        step: The training step at which to load the checkpoint. If both step and epoch are provided, an error is raised.
        epoch: The training epoch at which to load the checkpoint. If both step and epoch are provided, an error is raised.
    '''

    def __init__(self, checkpoint_path: str, step: int = 0, epoch: int = 0):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoint path {checkpoint_path} does not exist.'
            )
        if step and epoch:
            raise ValueError('Only one of step or epoch should be non-zero.')

        if step:
            suffix = f'step_{step}'
        elif epoch:
            suffix = f'epoch_{epoch}'
        else:
            suffix = 'latest'
        checkpoint_path = os.path.join(checkpoint_path, suffix)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoint path {checkpoint_path} does not exist.'
            )

        self.checkpoint_path = checkpoint_path

    def before_stage(self, trainer: Trainer) -> LoopControl | None:
        # Check DeferHookExec conditions
        _ = trainer.model, trainer.device
        # Warn if loading from exception checkpoint
        if os.path.exists(os.path.join(self.checkpoint_path, 'exception.txt')):
            warnings.warn(
                f'Loading from exception checkpoint at {self.checkpoint_path}.'
            )
        # Only load model context and global context
        # Since epoch and step context are to be created during runtime
        device = trainer.device
        load_ctx = lambda filename: torch.load(
            os.path.join(self.checkpoint_path, filename), map_location=device
        )
        trainer.model_context.load_dict(load_ctx('model_context.pt'))
        trainer.global_context.load_dict(load_ctx('global_context.pt'))
        trainer.model_context.device = device

class ResumeCheckpointHook(BaseHook):
    '''
    Hook to load model checkpoints before training. If used together with InitHooks, this hook will be executed after them.

    Args:
        checkpoint_path: The base path of the checkpoint to be loaded.
        step: The training step at which to load the checkpoint. If both step and epoch are provided, an error is raised.
        epoch: The training epoch at which to load the checkpoint. If both step and epoch are provided, an error is raised.
    '''

    def __init__(self, checkpoint_path: str, step: int = 0, epoch: int = 0):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoint path {checkpoint_path} does not exist.'
            )
        if step and epoch:
            raise ValueError('Only one of step or epoch should be non-zero.')

        if step:
            suffix = f'step_{step}'
        elif epoch:
            suffix = f'epoch_{epoch}'
        else:
            suffix = 'latest'
        checkpoint_path = os.path.join(checkpoint_path, suffix)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoint path {checkpoint_path} does not exist.'
            )

        self.checkpoint_path = checkpoint_path


    def before_stage(self, trainer: Trainer) -> LoopControl | None:
        # Check DeferHookExec conditions
        _ = trainer.model, trainer.optimizer, trainer.device
        if trainer.model_context.lr_scheduler is None:
            # Strict resume here - must load as is
            if any(
                fname == 'has_lr_scheduler'
                for fname in os.listdir(self.checkpoint_path)
            ):
                raise DeferHookExec()
        else:
            if all(
                fname != 'has_lr_scheduler'
                for fname in os.listdir(self.checkpoint_path)
            ):
                warnings.warn(
                    'Checkpoint does not have lr_scheduler, but model_context has. '
                )
        # Warn if resuming from exception checkpoint
        if os.path.exists(os.path.join(self.checkpoint_path, 'exception.txt')):
            warnings.warn(
                f'Resuming from exception checkpoint at {self.checkpoint_path}.'
            )
        # Only load model context and global context
        # Since epoch and step context are to be created during runtime
        device = trainer.device
        load_ctx = lambda filename: torch.load(
            os.path.join(self.checkpoint_path, filename),
            map_location=device, weights_only=False
        )
        trainer.model_context.load_dict(load_ctx('model_context.pt'))
        trainer.global_context.load_dict(load_ctx('global_context.pt'))
        trainer.model_context.device = device


class SaveCheckpointHook(BaseHook):
    '''
    Hook to save model checkpoints at specified intervals.

    Args:
        checkpoint_dir: The directory where checkpoints will be saved.
        save_interval_step: The interval (in steps) at which to save checkpoints.
        save_interval_epoch: The interval (in epochs) at which to save checkpoints.
    '''

    def __init__(
        self, checkpoint_dir: str,
        save_interval_step: int = 0, save_interval_epoch: int = 0,
        overwrite: bool = False
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.save_interval_step = save_interval_step
        self.save_interval_epoch = save_interval_epoch
        self.overwrite = overwrite
        if os.path.exists(self.checkpoint_dir) and os.listdir(self.checkpoint_dir):
            if not self.overwrite:
                raise FileExistsError(
                    f'Non-empty checkpoint directory {self.checkpoint_dir} already exists.'
                )
            warnings.warn(
                f'Non-empty checkpoint directory {self.checkpoint_dir} already exists. '
                'Existing checkpoints will be removed.'
            )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, dir_path: str, trainer: Trainer) -> None:
        tmp_dir = dir_path + '_tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.makedirs(tmp_dir)
        # Save model
        save_ctx = lambda ctx, filename: torch.save(
            ctx, os.path.join(tmp_dir, filename)
        )
        save_ctx(trainer.model_context.to_dict(), 'model_context.pt')
        save_ctx(trainer.global_context.to_dict(), 'global_context.pt')
        # Handle lr_scheduler
        if trainer.model_context.lr_scheduler is not None:
            open(os.path.join(tmp_dir, 'has_lr_scheduler'), 'a').close()

        # Save exception info if any
        if trainer.exception is not None:
            exception_path = os.path.join(tmp_dir, 'exception.txt')
            with open(exception_path, 'w') as f:
                f.write(''.join(
                    traceback.format_exception(
                        type(trainer.exception),
                        trainer.exception,
                        trainer.exception.__traceback__
                    )
                ))

        # Atomic move
        os.rename(tmp_dir, dir_path)

        # Create a `latest` symlink if no exception
        if trainer.exception is None:
            latest_path = os.path.join(self.checkpoint_dir, 'latest')
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                os.remove(latest_path)
            latest_tmp = latest_path + '_tmp'
            if os.path.islink(latest_tmp) or os.path.exists(latest_tmp):
                os.remove(latest_tmp)
            latest_target = os.path.relpath(dir_path, self.checkpoint_dir)
            os.symlink(latest_target, latest_tmp)
            os.replace(latest_tmp, latest_path)

    def after_step(self, trainer: Trainer) -> LoopControl | None:
        if self.save_interval_step > 0 and \
           trainer.global_context.step % self.save_interval_step == 0:
            step_dir = os.path.join(
                self.checkpoint_dir, f'step_{trainer.global_context.step}'
            )
            self.save(step_dir, trainer)

    def finalize_epoch(self, trainer: Trainer) -> LoopControl | None:
        if self.save_interval_epoch > 0 and \
           trainer.global_context.epoch % self.save_interval_epoch == 0:
            epoch_dir = os.path.join(
                self.checkpoint_dir, f'epoch_{trainer.global_context.epoch}'
            )
            self.save(epoch_dir, trainer)

    def finalize_stage(self, trainer: Trainer) -> LoopControl | None:
        if trainer.exception is None:
            final_dir = os.path.join(self.checkpoint_dir, 'final')
            self.save(final_dir, trainer)

    def on_exception(self, trainer: Trainer) -> LoopControl | None:
        exception_dir = os.path.join(
            self.checkpoint_dir, 'exception'
        )
        self.save(exception_dir, trainer)


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

    def _eval(self, trainer: Trainer) -> None:
        if self.copy:
           self.evaluator.model.load_state_dict(trainer.model.state_dict())
        else:
            self.evaluator.model_context.device = trainer.device
            self.evaluator.model_context.model = trainer.model
        self.evaluator.evaluate()
        new_metrics = self.evaluator.get_metrics().copy()
        if trainer.global_context.metrics:
            if trainer.global_context.metrics[-1][1] == trainer.global_context.step:
                # Avoid duplicate metrics for the same step
                trainer.global_context.metrics[-1][-1].update(new_metrics)
                return
        trainer.global_context.metrics.append((
            trainer.global_context.epoch,
            trainer.global_context.step,
            new_metrics
        ))

    def after_step(self, trainer: Trainer) -> LoopControl | None:
        if self.eval_interval_step > 0 and \
            trainer.global_context.step % self.eval_interval_step == 0:
            self._eval(trainer)

    def finalize_epoch(self, trainer: Trainer) -> LoopControl | None:
        if self.eval_interval_epoch > 0 and \
            trainer.global_context.epoch % self.eval_interval_epoch == 0:
            self._eval(trainer)

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

    def check_step(self, trainer: Trainer) -> LoopControl | None:
        if self.num_evaluations == len(trainer.global_context.metrics):
            # No new evaluation come in
            return None
        # Update best metric
        _, _, new_metrics = trainer.global_context.metrics[-1]

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
        self.num_evaluations = len(trainer.global_context.metrics)
        if self.num_bad_evals >= self.patience:
            return LoopControl.SKIP_STAGE

class WandBHook(BaseHook):
    '''
    Hook to log training metrics to Weights and Biases (wandb).
    The hook must be registered with the least priority (largest priority number)

    Items will be logged:
    - Losses from ``step_context['loss']`` and ``step_context['losses']``, both are PyTorch tensors.
    - Metrics gathered from ``global_context.metrics[-1]``.
    - Learning rates from optimizer.param_groups.
    - Additional entries from step_context defined in `additional_entries`

    Args:
        project: The wandb project name.
        entity: The wandb entity (user or team) name.
        run_name: The name of the wandb run. If not provided, a random name will be generated.
        config: A dictionary of hyperparameters to be logged in wandb.
        flush_interval: The interval (in steps) at which to flush the logged data to wandb, to reduce device synchronization overhead.
        loss_keys: The names of the losses.
        optimizer_keys: The names of the optimizers.
    '''
    def __init__(
        self, project: str, entity: str | None = None,
        run_name: str | None = None, config: dict | None = None,
        flush_interval: int = 1000,
        loss_keys: List[str] | None = None,
        optimizer_keys: List[str] | None = None,
        additional_entries: Dict[str, str] | List[str] | None = None
    ):
        import wandb
        self.wandb = wandb
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config
        self.flush_interval = flush_interval
        self.loss_keys = loss_keys or []
        self.optimizer_keys = optimizer_keys or []
        self.num_evaluations = 0

        self.pending_data: Dict[int, Dict[str, Any]] = {}
        if isinstance(additional_entries, list):
            self.additional_entries = {entry: entry for entry in additional_entries}
        else:
            self.additional_entries = additional_entries or {}

    def before_stage(self, trainer: Trainer) -> LoopControl | None:
        self.wandb_run = self.wandb.init(
            project=self.project, entity=self.entity,
            name=self.run_name, config=self.config
        )

    def _write_data(self, step: int, key: str, value: Any):
        if torch.is_tensor(value):
            value = value.detach() # Detach but not sync
        self.pending_data.setdefault(step, {})[key] = value

    def _sync_tensor(self, data: Dict[str, Any]):
        for k, v in data.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    data[k] = v.item()
                elif v.dim() == 1:
                    # Histogram
                    hist = np.histogram(v.cpu().numpy(), bins='auto')
                    data[k] = self.wandb.Histogram(np_histogram=hist)
                elif v.dim() == 2 and v.shape[0] == 2:
                    # Line plot
                    data[k] = self.wandb.plot.line_series(
                        xs=v[0].cpu().numpy(), ys=v[1].cpu().numpy()
                    )
                elif v.dim() == 2 or (v.dim() == 3 and v.shape[0] in [1, 3, 4]):
                    data[k] = self.wandb.Image(v.cpu(), normalize=True) # CHW
                else:
                    raise ValueError(f'Unsupported tensor shape {v.shape} for logging.')
            else:
                data[k] = v
        return data

    def _flush(self):
        if hasattr(self, 'wandb_run'):
            for step in sorted(self.pending_data):
                data = self.pending_data[step]
                self.wandb_run.log(
                    self._sync_tensor(data), step=step, commit=True
                )
            self.pending_data.clear()

    def before_step(self, trainer: Trainer) -> LoopControl | None:
        # commit data
        # Some metrics are logged in finalize_epoch, which happens after finalize_step
        # So, here must flush in before_step
        if self.pending_data and trainer.global_context.step % self.flush_interval == 0:
            self._flush()

    def _add_metrics(self, trainer: Trainer):
        if len(trainer.global_context.metrics) == self.num_evaluations:
            # No new evaluation come in
            return
        *_, metrics = trainer.global_context.metrics[-1]
        step = trainer.global_context.step - 1 # Align with the step at which losses are computed
        for key, value in metrics.items():
            self._write_data(step, f'metrics/{key}', value)
        self.num_evaluations = len(trainer.global_context.metrics)

    def finalize_optimizer_step(self, trainer: Trainer) -> LoopControl | None:
        # Log learning rate
        step = trainer.global_context.step
        if self.optimizer_keys and len(trainer.optimizer) != len(self.optimizer_keys):
            raise ValueError('Unmatched number of optimizers.')
        for i, optim in enumerate(trainer.optimizer):
            name = self.optimizer_keys[i] if self.optimizer_keys else i
            for j, param_group in enumerate(optim.param_groups):
                self._write_data(step, f'lr/{name}/{j}', param_group['lr'])

    def finalize_backward(self, trainer: Trainer) -> LoopControl | None:
        step = trainer.global_context.step
        if 'loss' in trainer.step_context:
            self._write_data(step, 'loss', trainer.step_context['loss'])
        if 'losses' in trainer.step_context:
            losses = trainer.step_context['losses'].reshape(-1).detach()
            if self.loss_keys and len(self.loss_keys) != losses.numel():
                raise ValueError('Unmatched number of losses.')
            for i in range(losses.numel()):
                name = self.loss_keys[i] if self.loss_keys else i
                self._write_data(step, f'losses/{name}', losses[i])

    def finalize_step(self, trainer: Trainer) -> LoopControl | None:
        self._add_metrics(trainer)
        for entry, name in self.additional_entries.items():
            if entry in trainer.step_context:
                self._write_data(
                    trainer.global_context.step - 1, name, trainer.step_context[entry]
                )

    def finalize_epoch(self, trainer: Trainer) -> LoopControl | None:
        # Log epoch-level metrics if any
        self._add_metrics(trainer)

    def finalize_stage(self, trainer: Trainer) -> LoopControl | None:
        if hasattr(self, 'wandb_run'):
            self._flush()
            self.wandb_run.finish()
