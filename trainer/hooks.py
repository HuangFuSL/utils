'''
`utils.trainer.hooks` - Trainer hooks for various training functionalities.
'''

from typing import Any, List, Literal
import torch
import os
import json
import traceback
import shutil
import warnings

from torch.profiler import profile

from . import BaseTrainerHook, DeferHookExec, Trainer, TrainerControl
from ..ctorch.device import get_best_device


class InitModelHook(BaseTrainerHook):
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

    def before_train(self, trainer: 'Trainer') -> TrainerControl | None:
        model = self.model_cls(*self.args, **self.kwargs)
        trainer.context.model = model


class InitOptimizerHook(BaseTrainerHook):
    '''
    Hook to initialize the optimizer before training. The hook will be executed after the model is initialized and moved to the device.

    Args:
        optim_cls: The class of the optimizer to be initialized.
        **kwargs: Keyword arguments to be passed to the optimizer constructor.
    '''

    def __init__(self, optim_cls, **kwargs):
        self.optim_cls = optim_cls
        self.kwargs = kwargs

    def before_train(self, trainer: 'Trainer') -> TrainerControl | None:
        if trainer.context.model is None:
            raise DeferHookExec()
        optimizer = self.optim_cls(
            trainer.context.model.parameters(), **self.kwargs
        )
        trainer.context.optimizer = optimizer


class InitLRSchedulerHook(BaseTrainerHook):
    '''
    Hook to initialize the learning rate scheduler before training. The hook will be executed
    after the optimizer is initialized.
    '''

    def __init__(self, lr_scheduler_cls, **kwargs):
        self.lr_scheduler_cls = lr_scheduler_cls
        self.kwargs = kwargs

    def before_train(self, trainer: 'Trainer') -> TrainerControl | None:
        if trainer.context.optimizer is None:
            raise DeferHookExec()
        lr_scheduler = self.lr_scheduler_cls(
            trainer.context.optimizer, **self.kwargs
        )
        trainer.context.lr_scheduler = lr_scheduler


class InitDataloaderHook(BaseTrainerHook):
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

    def before_train(self, trainer: 'Trainer') -> TrainerControl | None:
        dataloader = self.dataloader_cls(*self.args, **self.kwargs)
        trainer.context.train_dataloader = dataloader


class InitDeviceHook(BaseTrainerHook):
    '''
    Hook to initialize the device before training. If no device is specified, the best available device will be used.

    Args:
        device: The device to be used for training.
    '''

    def __init__(self, device: torch.device | str | None = None):
        if device is None:
            device = get_best_device()
        self.device = torch.device(device)

    def before_train(self, trainer: 'Trainer') -> TrainerControl | None:
        if trainer.context.model is None:
            raise DeferHookExec()

        trainer.context.device = self.device
        trainer.context.model.to(trainer.device)

    def after_fetch(self, trainer: 'Trainer') -> TrainerControl | None:
        def _move_to_device(obj: Any, device: torch.device) -> Any:
            if isinstance(obj, (torch.Tensor, torch.nn.utils.rnn.PackedSequence)):
                return obj.to(device, non_blocking=True)
            elif isinstance(obj, (list, tuple)):
                return type(obj)(
                    _move_to_device(item, device) for item in obj
                )
            elif isinstance(obj, dict):
                return {
                    key: _move_to_device(value, device)
                    for key, value in obj.items()
                }
            else:
                return obj
        trainer.step_context.batch = _move_to_device(
            trainer.step_context.batch, trainer.device
        )


class GradientClipHook(BaseTrainerHook):
    '''
    Hook to clip gradients after backward pass.
    '''

    def __init__(
        self, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False
    ):
        self.error_if_nonfinite = error_if_nonfinite
        self.max_norm = max_norm
        self.norm_type = norm_type

    def after_backward(self, trainer: 'Trainer') -> TrainerControl | None:
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), self.max_norm, self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite
        )


class DefaultZeroGradHook(BaseTrainerHook):
    '''
    Hook to zero gradients before forward pass.
    '''

    def __init__(self, per_step: int = 1):
        self.per_step = per_step  # Single step case

    def before_forward(self, trainer: 'Trainer') -> TrainerControl | None:
        if trainer.context.current_global_data_step % self.per_step == 0:
            for optim in trainer.optimizer:
                optim.zero_grad()


class DefaultTrainModeHook(BaseTrainerHook):
    '''
    Hook to set the model to training mode before each step.
    '''

    def before_step(self, trainer: 'Trainer') -> TrainerControl | None:
        trainer.model.train()


class RaiseExceptionHook(BaseTrainerHook):
    def on_exception(self, trainer: 'Trainer') -> TrainerControl | None:
        if trainer.exception is not None:
            raise trainer.exception


class SuppressExceptionHook(BaseTrainerHook):
    def on_exception(self, trainer: 'Trainer') -> TrainerControl | None:
        return TrainerControl.SKIP_STEP


class MaxEpochHook(BaseTrainerHook):
    '''
    Hook to stop training after a certain number of epochs.

    Args:
        num_epochs: The maximum number of epochs to train.
    '''

    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs

    def before_epoch(self, trainer: Trainer) -> TrainerControl | None:
        if trainer.global_context.current_epoch >= self.num_epochs:
            return TrainerControl.STOP_TRAINING


class MaxStepHook(BaseTrainerHook):
    '''
    Hook to stop training after a certain number of steps.

    Args:
        num_steps: The maximum number of steps to train.
    '''

    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def before_step(self, trainer: Trainer) -> TrainerControl | None:
        if trainer.global_context.current_global_data_step >= self.num_steps:
            return TrainerControl.STOP_TRAINING


class TqdmHook(BaseTrainerHook):
    '''
    Hook to display a progress bar using tqdm during training.

    Args:
        unit: The unit of progress to track. Can be 'data_step', 'update_step', or 'epoch'.
    '''

    def __init__(self, unit: Literal['data_step', 'update_step', 'epoch'] = 'data_step'):
        import tqdm
        self.unit = unit
        self.tqdm = tqdm.tqdm(unit=' ' + self.unit)

    def after_data_step(self, trainer: Trainer) -> TrainerControl | None:
        if self.unit == 'data_step':
            self.tqdm.update(1)

    def after_update_step(self, trainer: Trainer) -> TrainerControl | None:
        if self.unit == 'update_step':
            self.tqdm.update(1)

    def after_epoch(self, trainer: Trainer) -> TrainerControl | None:
        if self.unit == 'epoch':
            self.tqdm.update(1)

    def after_train(self, trainer: Trainer) -> TrainerControl | None:
        self.tqdm.close()


class WeightedLossHook(BaseTrainerHook):
    '''
    Hook to compute weighted sum of multiple losses.

    Args:
        num_losses: The number of losses to be combined.
        loss_weights: The weights for each loss.
    '''
    def __init__(self, num_losses: int, loss_weights: List[float]):
        if len(loss_weights) != num_losses:
            raise ValueError(
                f'Length of loss_weights ({len(loss_weights)}) does not match num_losses ({num_losses}).'
            )
        self.num_losses = num_losses
        self.loss_weights = torch.tensor(loss_weights)

    def before_train(self, trainer: Trainer) -> TrainerControl | None:
        if trainer.context.device is None:
            raise DeferHookExec()
        self.loss_weights = self.loss_weights.to(trainer.context.device)

    def on_reduce_loss(self, trainer: Trainer) -> TrainerControl | None:
        losses = trainer.step_context.losses
        assert losses is not None
        trainer.step_context.loss = (losses @ self.loss_weights).mean()


class ProfileHook(BaseTrainerHook):
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
        self.prof = profile(
            experimental_config=torch._C._profiler._ExperimentalConfig(
                verbose=True),
            **kwargs
        )

    def before_train(self, trainer: Trainer) -> TrainerControl | None:
        trainer.context['profile'] = self.prof
        self.prof.__enter__()

    def before_step(self, trainer: Trainer) -> TrainerControl | None:
        if trainer.global_context.current_global_data_step > self.num_steps:
            return TrainerControl.STOP_TRAINING

    def after_train(self, trainer: Trainer) -> TrainerControl | None:
        self.prof.__exit__(None, None, None)
        if self.export_path is not None:
            self.prof.export_chrome_trace(f"{self.export_path}/trace.json")


class ResumeCheckpointHook(BaseTrainerHook):
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
        self.step = step
        self.epoch = epoch

    def before_train(self, trainer: Trainer) -> TrainerControl | None:
        if trainer.context.model is None:
            raise DeferHookExec()
        if trainer.context.optimizer is None:
            raise DeferHookExec()
        # For lr_scheduler, it can be None
        if trainer.context.lr_scheduler is None:
            if any(
                fname.startswith('lr_scheduler_') and fname.endswith('.pt')
                for fname in os.listdir(self.checkpoint_path)
            ):
                raise DeferHookExec()
        if trainer.context.train_dataloader is None:
            raise DeferHookExec()
        if trainer.context.device is None:
            raise DeferHookExec()
        # Load global context
        device = trainer.device
        context_path = os.path.join(self.checkpoint_path, 'global_context.json')
        with open(context_path, 'r') as f:
            context_dict = json.load(f)
        # The load_dict is in-place and recursively loads child contexts.
        # It creates child contexts without incrementing counters
        trainer.context.load_dict(context_dict)
        trainer.context.device = device # Write back the device
        # Load state dicts
        trainer.context.model.load_state_dict(torch.load(
            os.path.join(self.checkpoint_path, 'model.pt'), map_location=device
        ))
        for i, optim in enumerate(trainer.optimizer):
            optim.load_state_dict(torch.load(
                os.path.join(self.checkpoint_path, f'optimizer_{i}.pt'), map_location=device
            ))
        if trainer.lr_scheduler is not None:
            for i, lr_scheduler in enumerate(trainer.lr_scheduler):
                lr_scheduler.load_state_dict(torch.load(
                    os.path.join(
                        self.checkpoint_path, f'lr_scheduler_{i}.pt'
                    ), map_location=device
                ))
        # Fast-forward to the correct step
        if trainer.context._child is not None:
            for _ in range(trainer.epoch_context.current_data_step):
                next(trainer.epoch_context.dataloader_iter)

class SaveCheckpointHook(BaseTrainerHook):
    '''
    Hook to save model checkpoints at specified intervals.

    Args:
        checkpoint_dir: The directory where checkpoints will be saved.
        save_interval_step: The interval (in steps) at which to save checkpoints.
        save_interval_epoch: The interval (in epochs) at which to save checkpoints.
    '''

    def __init__(self, checkpoint_dir: str, save_interval_step: int = 0, save_interval_epoch: int = 0):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.save_interval_step = save_interval_step
        self.save_interval_epoch = save_interval_epoch
        if os.path.exists(self.checkpoint_dir):
            warnings.warn(
                f'Checkpoint directory {self.checkpoint_dir} already exists. '
                'Existing checkpoints may be overwritten.'
            )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, dir_path: str, trainer: Trainer) -> None:
        tmp_dir = dir_path + '_tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        os.makedirs(tmp_dir)
        # Save model
        model_path = os.path.join(tmp_dir, 'model.pt')
        torch.save(trainer.model.state_dict(), model_path)
        # Save optimizer
        for i, optim in enumerate(trainer.optimizer):
            optim_path = os.path.join(tmp_dir, f'optimizer_{i}.pt')
            torch.save(optim.state_dict(), optim_path)
        # Save lr_scheduler
        if trainer.lr_scheduler is not None:
            for i, lr_scheduler in enumerate(trainer.lr_scheduler):
                lr_scheduler_path = os.path.join(tmp_dir, f'lr_scheduler_{i}.pt')
                torch.save(lr_scheduler.state_dict(), lr_scheduler_path)
        # Save trainer context
        context_path = os.path.join(tmp_dir, 'global_context.json')
        with open(context_path, 'w') as f:
            json.dump(trainer.global_context.to_dict(), f)

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
            os.symlink(dir_path, latest_path)

    def after_step(self, trainer: Trainer) -> TrainerControl | None:
        if self.save_interval_step > 0 and \
           trainer.global_context.current_global_data_step % self.save_interval_step == 0:
            step_dir = os.path.join(
                self.checkpoint_dir,
                f'step_{trainer.global_context.current_global_data_step}'
            )
            self.save(step_dir, trainer)

    def after_epoch(self, trainer: Trainer) -> TrainerControl | None:
        if self.save_interval_epoch > 0 and \
           trainer.global_context.current_epoch % self.save_interval_epoch == 0:
            epoch_dir = os.path.join(
                self.checkpoint_dir,
                f'epoch_{trainer.global_context.current_epoch}'
            )
            self.save(epoch_dir, trainer)

    def after_train(self, trainer: Trainer) -> TrainerControl | None:
        final_dir = os.path.join(self.checkpoint_dir, 'final')
        self.save(final_dir, trainer)

    def on_exception(self, trainer: Trainer) -> TrainerControl | None:
        exception_dir = os.path.join(
            self.checkpoint_dir, 'exception'
        )
        self.save(exception_dir, trainer)
