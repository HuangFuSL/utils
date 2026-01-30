'''
`utils.trainer` - Trainer module with context management and hooks.
'''

import dataclasses
import enum
import bisect
from typing import Any, Callable, Dict, Iterator, List, Tuple

import torch

from ..ctorch.nn import Model

Optimizer = torch.optim.Optimizer
LRScheduler = torch.optim.lr_scheduler.LRScheduler


@dataclasses.dataclass
class GlobalContext():
    # Static fields
    _child: 'EpochContext | None' = None
    current_global_data_step: int = 0
    current_global_update_step: int = 0
    current_epoch: int = 0

    device: torch.device | None = None
    model: Model | None = None
    optimizer: Optimizer | List[Optimizer] | None = None
    lr_scheduler: LRScheduler | List[LRScheduler] | None = None

    train_dataloader: torch.utils.data.DataLoader | None = None

    objects: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_global_data_step': self.current_global_data_step,
            'current_global_update_step': self.current_global_update_step,
            'current_epoch': self.current_epoch,
            'device': str(self.device) if self.device is not None else None,
            'objects': self.objects,
            '_child': self._child.to_dict() if self._child is not None else None,
        }

    def load_dict(self, data: Dict[str, Any]):
        self.current_global_data_step = data['current_global_data_step']
        self.current_global_update_step = data['current_global_update_step']
        self.current_epoch = data['current_epoch']
        self.device = torch.device(data['device']) if data['device'] is not None else None
        self.objects = data['objects']

        if data['_child'] is not None:
            if self._child is None:
                self._child = EpochContext(_parent=self)
            self._child.load_dict(data['_child'])

    def __getitem__(self, key: str) -> Any:
        return self.objects[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.objects[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.objects.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        return self.objects.setdefault(key, default)

    @property
    def child(self) -> 'EpochContext':
        if self._child is not None:
            return self._child
        raise ValueError('EpochContext is not created yet.')

    def new_child(self) -> 'EpochContext':
        if self._child is not None:
            return self._child
        self._child = EpochContext(_parent=self)
        return self._child

    def increment_epoch(self):
        self.current_epoch += 1

    def finalize_child(self):
        if self._child is None:
            return
        self.child.finalize()
        del self._child
        self._child = None


@dataclasses.dataclass
class EpochContext():
    _parent: GlobalContext
    _child: 'StepContext | None' = None
    _current_dataloader_iter: Iterator[Any] | None = None

    current_data_step: int = 0
    current_update_step: int = 0

    objects: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_data_step': self.current_data_step,
            'current_update_step': self.current_update_step,
            'objects': self.objects,
            '_child': self._child.to_dict() if self._child is not None else None,
        }

    def load_dict(self, data: Dict[str, Any]):
        self.objects = data['objects']
        self.current_data_step = data['current_data_step']
        self.current_update_step = data['current_update_step']
        if data['_child'] is not None:
            if self._child is None:
                self._child = StepContext(_parent=self)
            self._child.load_dict(data['_child'])

    def __getitem__(self, key: str) -> Any:
        return self.objects[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.objects[key] = value

    @property
    def child(self) -> 'StepContext':
        if self._child is not None:
            return self._child
        raise ValueError('StepContext is not created yet.')

    def finalize(self):
        self._current_dataloader_iter = None
        self.objects.clear()

    def new_child(self) -> 'StepContext':
        if self._child is not None:
            return self._child
        self._child = StepContext(_parent=self)
        return self._child

    def increment_data_step(self):
        self.current_data_step += 1
        self._parent.current_global_data_step += 1

    def increment_update_step(self):
        self.current_update_step += 1
        self._parent.current_global_update_step += 1

    def finalize_child(self):
        if self._child is None:
            return
        self.child.finalize()
        del self._child
        self._child = None

    @property
    def dataloader_iter(self) -> Iterator[Any]:
        if self._current_dataloader_iter is None:
            if self._parent.train_dataloader is None:
                raise ValueError('train_dataloader is not set yet.')
            self._current_dataloader_iter = iter(self._parent.train_dataloader)
        return self._current_dataloader_iter


@dataclasses.dataclass
class StepContext():
    _parent: EpochContext

    batch: Any = None
    losses: torch.Tensor | None = None
    loss: torch.Tensor | None = None

    objects: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'objects': self.objects,
        }

    def load_dict(self, data: Dict[str, Any]):
        self.objects = data['objects']

    def __getitem__(self, key: str) -> Any:
        return self.objects[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.objects[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.objects.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        return self.objects.setdefault(key, default)

    def finalize(self):
        self.batch = None
        self.losses = None
        self.loss = None
        self.objects.clear()


class TrainerControl(enum.IntFlag):
    NONE = 0
    SKIP_LR_SCHEDULER_STEP = enum.auto()
    SKIP_OPTIMIZER_STEP = enum.auto()
    SKIP_STEP = enum.auto()
    STOP_TRAINING = enum.auto()

HookReturn = TrainerControl | None

class TrainEnd(Exception): pass
class EpochEnd(Exception): pass
class StepEnd(Exception): pass
class DeferHookExec(Exception): pass


@dataclasses.dataclass(slots=True)
class BaseTrainerHook():
    '''
    Base class for trainer hooks. All methods return NotImplemented by default.
    Subclasses can override the methods they need. Any method should check the criteria
    for its execution and raise DeferHookExec if the criteria are not met yet.

    Methods:
        before_train(trainer): Called before training starts.
        before_epoch(trainer): Called before each epoch starts.
        before_step(trainer): Called before each training step.
        before_fetch(trainer): Called before fetching data.
        after_fetch(trainer): Called after fetching data.
        before_forward(trainer): Called before the forward pass.
        after_forward(trainer): Called after the forward pass.
        before_backward(trainer): Called before the backward pass.
        after_backward(trainer): Called after the backward pass.
        after_data_step(trainer): Called after each data step.
        on_reduce_loss(trainer): Called when reducing the loss.
        before_optimizer_step(trainer): Called before the optimizer step.
        after_optimizer_step(trainer): Called after the optimizer step.
        after_update_step(trainer): Called after each update step.
        before_lr_scheduler_step(trainer): Called before the LR scheduler step.
        after_lr_scheduler_step(trainer): Called after the LR scheduler step.
        after_step(trainer): Called after each training step.
        after_epoch(trainer): Called after each epoch ends.
        after_train(trainer): Called after training ends.
        on_exception(trainer): Called when an exception occurs during training. The exception can be accessed via trainer.exception.
    '''
    def before_train(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def before_epoch(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def before_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def before_fetch(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_fetch(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def before_forward(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_forward(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def before_backward(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_backward(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_data_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def on_reduce_loss(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def before_optimizer_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_optimizer_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_update_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def before_lr_scheduler_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_lr_scheduler_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_step(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_epoch(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def after_train(self, trainer: 'Trainer') -> HookReturn: return NotImplemented
    def on_exception(self, trainer: 'Trainer') -> HookReturn: return NotImplemented


class Trainer():
    '''
    Trainer class to manage the training loop with hooks.

    Methods:
        train(): Start the training process.
        register_hook(hook, priority): Register a training hook with optional priority. All hooks must be registered before calling train().
    '''
    def __init__(self):
        self.hooks: List[Tuple[int, BaseTrainerHook]] = []
        self.context = GlobalContext()
        self.exception: Exception | None = None
        self._hook_sequence: Dict[str, List[Callable[['Trainer'], HookReturn]]] = {}

    @property
    def global_context(self) -> GlobalContext:
        return self.context

    @property
    def epoch_context(self) -> EpochContext:
        return self.context.child

    @property
    def step_context(self) -> StepContext:
        return self.context.child.child

    @property
    def device(self) -> torch.device:
        if self.context.device is None:
            raise ValueError('device is not set yet.')
        return self.context.device

    @property
    def model(self) -> Model:
        if self.context.model is None:
            raise ValueError('model is not set yet.')
        return self.context.model

    @property
    def optimizer(self) -> List[Optimizer]:
        if self.context.optimizer is None:
            raise ValueError('optimizer is not set yet.')
        if not isinstance(self.context.optimizer, list):
            return [self.context.optimizer]
        return self.context.optimizer

    @property
    def lr_scheduler(self) -> List[LRScheduler] | None:
        if self.context.lr_scheduler is None:
            return None
        if not isinstance(self.context.lr_scheduler, list):
            return [self.context.lr_scheduler]
        return self.context.lr_scheduler

    @property
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.context.train_dataloader is None:
            raise ValueError('train_dataloader is not set yet.')
        return self.context.train_dataloader

    def register_hook(self, hook: BaseTrainerHook, priority: int | None = None) -> None:
        if priority is None:
            priority = len(self.hooks)
        bisect.insort(self.hooks, (priority, hook), key=lambda x: x[0])

    def _ctrl(self, control: TrainerControl) -> TrainerControl:
        actions = (
            (TrainerControl.SKIP_STEP, StepEnd),
            (TrainerControl.STOP_TRAINING, TrainEnd),
        )
        for flag, exception in actions:
            if control & flag:
                raise exception()
        return control

    def _call_hooks(self, method_name: str) -> TrainerControl:
        control = TrainerControl.NONE
        if method_name in self._hook_sequence:
            hook_seq = self._hook_sequence[method_name]
            for hook_fn in hook_seq:
                ret = hook_fn(self)
                if ret is not None:
                    control |= ret
            return control

        exec_seq = []
        deferred_hooks: List[BaseTrainerHook] = [
            hook for _, hook in self.hooks]
        while deferred_hooks:
            new_deferred_hooks: List[BaseTrainerHook] = []
            for hook in deferred_hooks:
                method = getattr(hook, method_name)
                if method is getattr(BaseTrainerHook, method_name):
                    # Skip hooks that do not implement this method
                    continue
                try:
                    ret = method(self)
                except DeferHookExec:
                    new_deferred_hooks.append(hook)
                    continue
                if ret is NotImplemented:
                    # Skip hooks that do not implement this method
                    raise ValueError('Override method returned NotImplemented.')
                exec_seq.append(method)
                if ret is not None:
                    control |= ret
            if len(deferred_hooks) == len(new_deferred_hooks):
                # No progress made, break to avoid infinite loop
                raise RuntimeError(
                    f'Cannot resolve dependencies for hooks in {method_name}.'
                )
            deferred_hooks = new_deferred_hooks

        self._hook_sequence[method_name] = exec_seq
        return control

    def _train_step(self):
        try:
            ctrl = TrainerControl.NONE
            self.epoch_context.new_child()
            # Before step
            ctrl = self._ctrl(ctrl | self._call_hooks('before_step'))

            # Fetch data
            ctrl = self._ctrl(ctrl | self._call_hooks('before_fetch'))
            try:
                data = next(self.epoch_context.dataloader_iter)
            except StopIteration:
                raise EpochEnd()
            self.step_context.batch = data
            ctrl = self._ctrl(ctrl | self._call_hooks('after_fetch'))

            # Forward
            ctrl = self._ctrl(ctrl | self._call_hooks('before_forward'))
            if isinstance(self.step_context.batch, (list, tuple)):
                ret = self.model.loss(*self.step_context.batch)
            elif isinstance(self.step_context.batch, dict):
                ret = self.model.loss(**self.step_context.batch)
            else:
                ret = self.model.loss(self.step_context.batch)
            self.step_context.losses = ret
            ctrl = self._ctrl(ctrl | self._call_hooks('after_forward'))
            ctrl = self._ctrl(ctrl | self._call_hooks('on_reduce_loss'))

            # Backward
            ctrl = self._ctrl(ctrl | self._call_hooks('before_backward'))
            if self.step_context.loss is None:
                self.step_context.loss = self.step_context.losses.mean()
            self.step_context.loss.backward()
            ctrl = self._ctrl(ctrl | self._call_hooks('after_backward'))

            # If backward, increment step count
            self.epoch_context.increment_data_step()
            ctrl = self._ctrl(ctrl | self._call_hooks('after_data_step'))

            # Optimizer step
            ctrl = self._ctrl(ctrl | self._call_hooks("before_optimizer_step"))
            if ctrl & TrainerControl.SKIP_OPTIMIZER_STEP:
                raise StepEnd()  # No optimizer step -> no scheduler step
            for optim in self.optimizer:
                optim.step()
            ctrl = self._ctrl(ctrl | self._call_hooks("after_optimizer_step"))

            self.epoch_context.increment_update_step()
            ctrl = self._ctrl(ctrl | self._call_hooks('after_update_step'))

            # LR scheduler step
            if self.lr_scheduler is not None:
                ctrl = self._ctrl(ctrl | self._call_hooks(
                    'before_lr_scheduler_step'))
                if ctrl & TrainerControl.SKIP_LR_SCHEDULER_STEP:
                    raise StepEnd()  # No scheduler step
                for lr_scheduler in self.lr_scheduler:
                    lr_scheduler.step()
                ctrl = self._ctrl(ctrl | self._call_hooks(
                    'after_lr_scheduler_step'))
            # After step
        except (StepEnd, EpochEnd, TrainEnd) as e:
            if not isinstance(e, StepEnd):
                raise
        except Exception as e:
            self.exception = e
            ctrl = self._call_hooks("on_exception")
            if ctrl == TrainerControl.NONE:
                raise e
            else:
                self._ctrl(ctrl)
        finally:
            self.exception = None
            self._ctrl(self._call_hooks('after_step'))

    def _train_epoch(self):
        # Before epoch
        self._ctrl(self._call_hooks('before_epoch'))

        while True:
            try:
                self._train_step()
            except StepEnd:
                continue
            except EpochEnd:
                break
            except TrainEnd:
                raise
            finally:
                self.epoch_context.finalize_child()

        self._ctrl(self._call_hooks('after_epoch'))

    def _train(self):
        # Before train
        self._ctrl(self._call_hooks('before_train'))
        while True:
            try:
                self.global_context.new_child()
                self._train_epoch()
            except TrainEnd:
                break
            finally:
                self.global_context.increment_epoch()
                self.global_context.finalize_child()

        # After train
        self._ctrl(self._call_hooks('after_train'))

    def train(self):
        # Clear previous hook execution sequence
        self._hook_sequence.clear()
        try:
            self._train()
        except TrainEnd:
            return
        except StopIteration:
            return
