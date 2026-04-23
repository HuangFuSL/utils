import os
import shutil
import traceback
import warnings

import torch

from .. import BaseHook, DeferHookExec, LoopControl


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

    def before_stage(self) -> LoopControl | None:
        # Check DeferHookExec conditions
        _ = self.parent.model, self.parent.device
        # Warn if loading from exception checkpoint
        if os.path.exists(os.path.join(self.checkpoint_path, 'exception.txt')):
            warnings.warn(
                f'Loading from exception checkpoint at {self.checkpoint_path}.'
            )
        # Only load model context and global context
        # Since epoch and step context are to be created during runtime
        device = self.parent.device
        load_ctx = lambda filename: torch.load(
            os.path.join(self.checkpoint_path, filename), map_location=device
        )
        self.parent.model_context.load_dict(load_ctx('model_context.pt'))
        self.parent.global_context.load_dict(load_ctx('global_context.pt'))
        self.parent.model_context.device = device

class ResumeCheckpointHook(BaseHook):
    '''
    Hook to load model checkpoints before training. If used together with InitHooks, this hook should be executed after them.

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


    def before_stage(self) -> LoopControl | None:
        # Check DeferHookExec conditions
        _ = self.parent.model, self.parent.optimizer, self.parent.device
        if self.parent.model_context.lr_scheduler is None:
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
        device = self.parent.device
        load_ctx = lambda filename: torch.load(
            os.path.join(self.checkpoint_path, filename),
            map_location=device, weights_only=False
        )
        self.parent.model_context.load_dict(load_ctx('model_context.pt'))
        self.parent.global_context.load_dict(load_ctx('global_context.pt'))
        self.parent.model_context.device = device


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

    def save(self, dir_path: str) -> None:
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
        save_ctx(self.parent.model_context.to_dict(), 'model_context.pt')
        save_ctx(self.parent.global_context.to_dict(), 'global_context.pt')
        # Handle lr_scheduler
        if self.parent.model_context.lr_scheduler is not None:
            open(os.path.join(tmp_dir, 'has_lr_scheduler'), 'a').close()

        # Save exception info if any
        if self.parent.exception is not None:
            exception_path = os.path.join(tmp_dir, 'exception.txt')
            with open(exception_path, 'w') as f:
                f.write(''.join(
                    traceback.format_exception(
                        type(self.parent.exception),
                        self.parent.exception,
                        self.parent.exception.__traceback__
                    )
                ))

        # Atomic move
        os.rename(tmp_dir, dir_path)

        # Create a `latest` symlink if no exception
        if self.parent.exception is None:
            latest_path = os.path.join(self.checkpoint_dir, 'latest')
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                os.remove(latest_path)
            latest_tmp = latest_path + '_tmp'
            if os.path.islink(latest_tmp) or os.path.exists(latest_tmp):
                os.remove(latest_tmp)
            latest_target = os.path.relpath(dir_path, self.checkpoint_dir)
            os.symlink(latest_target, latest_tmp)
            os.replace(latest_tmp, latest_path)

    def after_step(self) -> LoopControl | None:
        if self.save_interval_step > 0 and \
           self.parent.global_context.step % self.save_interval_step == 0:
            step_dir = os.path.join(
                self.checkpoint_dir, f'step_{self.parent.global_context.step}'
            )
            self.save(step_dir)

    def finalize_epoch(self) -> LoopControl | None:
        if self.save_interval_epoch > 0 and \
           self.parent.global_context.epoch % self.save_interval_epoch == 0:
            epoch_dir = os.path.join(
                self.checkpoint_dir, f'epoch_{self.parent.global_context.epoch}'
            )
            self.save(epoch_dir)

    def finalize_stage(self) -> LoopControl | None:
        if self.parent.exception is None:
            final_dir = os.path.join(self.checkpoint_dir, 'final')
            self.save(final_dir)

    def on_exception(self) -> LoopControl | None:
        exception_dir = os.path.join(
            self.checkpoint_dir, 'exception'
        )
        self.save(exception_dir)

