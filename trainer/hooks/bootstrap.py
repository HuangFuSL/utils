import torch

from .. import BaseHook, LoopControl
from ...ctorch.device import get_best_device

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

    def before_stage(self) -> LoopControl | None:
        model = self.model_cls(*self.args, **self.kwargs)
        self.parent.model_context.model = model


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

    def before_stage(self) -> LoopControl | None:
        self.parent.device # DeferHookExec if device is not initialized
        self.parent.model_context.optimizer = [self.optim_cls(
            self.parent.model.parameters(), **self.kwargs
        )]


class InitLRSchedulerHook(BaseHook):
    '''
    Hook to initialize the learning rate scheduler before training. The hook will be executed
    after the optimizer is initialized.
    '''

    def __init__(self, lr_scheduler_cls, **kwargs):
        self.lr_scheduler_cls = lr_scheduler_cls
        self.kwargs = kwargs

    def before_stage(self) -> LoopControl | None:
        if len(self.parent.optimizer) != 1:
            raise ValueError(
                'Length of lr_scheduler_cls must match length of optimizer.'
            )
        self.parent.model_context.lr_scheduler = [
            self.lr_scheduler_cls(self.parent.optimizer[0], **self.kwargs)
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

    def before_stage(self) -> LoopControl | None:
        dataloader = self.dataloader_cls(*self.args, **self.kwargs)
        self.parent.model_context.dataloader = dataloader


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

    def before_stage(self) -> LoopControl | None:
        self.parent.model_context.device = self.device
