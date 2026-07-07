'''
`utils.trainer` - Trainer module with context management and hooks.
'''

import bisect
import dataclasses
import enum
import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, OrderedDict, Tuple, TypeVar

import torch

from ..ctorch.nn import Module

Optimizer = torch.optim.Optimizer
LRScheduler = torch.optim.lr_scheduler.LRScheduler

T = TypeVar('T')

# Controls and hooks

class LoopControl(enum.IntEnum):
    '''
    Control flags for loop execution.

    - NONE: No control action.
    - SKIP_STEP: Skip the rest of the current step.
    - SKIP_EPOCH: Skip the rest of the current epoch.
    - SKIP_STAGE: Skip all remaining stages.
    '''
    NONE = 0
    SKIP_EVENT = 1
    SKIP_STEP = 2
    SKIP_EPOCH = 3
    SKIP_STAGE = 4

    def __bool__(self) -> bool:
        return self != LoopControl.NONE

    def __or__(self, other: 'LoopControl | None') -> 'LoopControl':
        if other is None:
            ret = self
        else:
            ret = LoopControl(max(self.value, other.value))
        return ret


# Exceptions
class DeferHookExec(Exception):
    ''' Exception for resolving hook execution order. '''
    pass


class BaseHook():
    '''
    Base class for hooks. All methods return None by default.
    Subclasses can override the methods they need. Hooks can be called
    via hook(method_name), which will dispatch to the appropriate method.
    '''
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *a, **kw):
            object.__setattr__(self, "_track_hp", True)
            object.__setattr__(self, "_unresolved_hp", set())
            object.__setattr__(self, "_parent", None)
            try:
                original_init(self, *a, **kw)
            finally:
                object.__setattr__(self, "_track_hp", False)

        def block_subclass(cls) -> None:
            raise RuntimeError(f'{cls.__name__} cannot be subclassed.')

        cls.__init__ = wrapped_init
        cls.__init_subclass__ = classmethod(block_subclass)  # type: ignore

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if not object.__getattribute__(self, "_track_hp"):
            return
        if name.startswith("_"):
            return
        if HyperParam.check(value):
            object.__getattribute__(self, "_unresolved_hp").add(name)
        else:
            object.__getattribute__(self, "_unresolved_hp").discard(name)

    def __getattribute__(self, name: str) -> Any:
        if not object.__getattribute__(self, '_unresolved_hp'):
            return super().__getattribute__(name)
        if name.startswith('_'):
            return super().__getattribute__(name)
        attr = super().__getattribute__(name)
        unresolved_hp = super().__getattribute__("_unresolved_hp")
        if not unresolved_hp or name not in unresolved_hp:
            return attr
        if super().__getattribute__('_parent') is None:
            return attr
        resolved_attr = HyperParam.resolve(attr, self)
        object.__setattr__(self, name, resolved_attr)
        if not HyperParam.check(resolved_attr):
            unresolved_hp.discard(name)
        return resolved_attr

    @property
    def parent(self) -> 'BatchedModelLoop':
        if self._parent is None:
            raise ValueError('Hook is not attached to any loop.')
        return self._parent

    @parent.setter
    def parent(self, value: 'BatchedModelLoop'):
        self._parent = value

    def __contains__(self, item: str) -> bool:
        attr = getattr(type(self), item, None)
        return callable(attr)

    def __getitem__(self, method_name: str):
        return getattr(self, method_name)

    def __call__(self, method_name: str) -> LoopControl | None:
        return self[method_name]()

@dataclasses.dataclass
class BaseContext():
    '''
    Base context class for storing arbitrary torch-saveable objects.
    '''
    objects: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.objects[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.objects[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.objects

    def get(self, key: str, default: Any = None) -> Any:
        return self.objects.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        return self.objects.setdefault(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {'objects': self.objects}

    def load_dict(self, data: Dict[str, Any]):
        objects = data.get('objects', {})
        self.objects = objects

@dataclasses.dataclass
class ModelContext(BaseContext):
    '''
    Context to store model-related objects.
    '''
    device: torch.device | None = None
    model: Module | None = None
    optimizer: List[Optimizer] | None = None
    lr_scheduler: List[LRScheduler] | None = None
    dataloader: torch.utils.data.DataLoader | None = None
    # ModelContext does not use objects dict.

    def to_dict(self) -> Dict[str, Any]:
        return {
            'device': str(self.device) if self.device is not None else None,
            'model': self.model.state_dict() if self.model is not None else None,
            'optimizer': [
                opt.state_dict() for opt in self.optimizer
            ] if self.optimizer is not None else None,
            'lr_scheduler': [
                sched.state_dict() for sched in self.lr_scheduler
            ] if self.lr_scheduler is not None else None,
            'dataloader': None,  # Dataloader state is not saved
        } | super().to_dict()

    def load_dict(self, data: Dict[str, Any]):
        super().load_dict(data)
        if data.get('device') is not None:
            self.device = torch.device(data['device'])
        if self.model is not None and data.get('model') is not None:
            self.model.load_state_dict(data['model'])
        if self.optimizer is not None and data.get('optimizer') is not None:
            for opt, state in zip(self.optimizer, data['optimizer']):
                opt.load_state_dict(state)
        if self.lr_scheduler is not None and data.get('lr_scheduler') is not None:
            for sched, state in zip(self.lr_scheduler, data['lr_scheduler']):
                sched.load_state_dict(state)

@dataclasses.dataclass(slots=True, frozen=True)
class HyperParam():
    name: str

    @classmethod
    def resolve(cls, obj: Any, hook: 'BaseHook') -> Any:
        if isinstance(obj, cls):
            if obj.name in hook.parent.global_context.hyperparams:
                return hook.parent.global_context.hyperparams[obj.name]
            raise DeferHookExec()
        elif isinstance(obj, (list, set, frozenset)):
            return [cls.resolve(item, hook) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(cls.resolve(item, hook) for item in obj)
        elif isinstance(obj, dict):
            # Dict[str, Any] type
            return {
                key: cls.resolve(value, hook)
                for key, value in obj.items()
            }
        else:
            return obj

    @classmethod
    def check(cls, obj: Any) -> bool:
        if isinstance(obj, cls):
            return True
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return any(cls.check(item) for item in obj)
        elif isinstance(obj, dict):
            return any(cls.check(value) for value in obj.values())
        else:
            return False

@dataclasses.dataclass
class GlobalContext(BaseContext):
    '''
    Context to store global counter and metrics.
    '''
    epoch: int = 0
    fetch: int = 0
    forward: int = 0
    backward: int = 0
    update: int = 0
    scheduler_step: int = 0
    step: int = 0
    hyperparams: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Epoch, step, metric dict
    metrics: List[Tuple[int, int, Dict[str, Any]]] = dataclasses.field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'fetch': self.fetch,
            'forward': self.forward,
            'backward': self.backward,
            'update': self.update,
            'scheduler_step': self.scheduler_step,
            'metrics': self.metrics,
            'step': self.step,
            'hyperparams': self.hyperparams,
        } | super().to_dict()

    def load_dict(self, data: Dict[str, Any]):
        super().load_dict(data)
        self.epoch = data['epoch']
        self.fetch = data['fetch']
        self.forward = data['forward']
        self.backward = data['backward']
        self.update = data['update']
        self.scheduler_step = data['scheduler_step']
        self.metrics = data['metrics']
        self.step = data['step']
        self.hyperparams = data['hyperparams']

@dataclasses.dataclass
class EpochContext(BaseContext):
    ''' Context to store epoch-level information. '''
    pass

@dataclasses.dataclass
class StepContext(BaseContext):
    ''' Context to store step-level information. '''
    batch: Any = None

HookReturn = LoopControl | None

def _auto_call(fn: Callable, batch: Any) -> Any:
    ''' Automatically call fn with unpacked batch if needed. '''
    if isinstance(batch, (list, tuple)):
        return fn(*batch)
    elif isinstance(batch, dict):
        return fn(**batch)
    else:
        return fn(batch)

class _NestedLoopCore(BaseHook):
    ''' General core hook for training/evaluation loops. '''

    def __init__(self):
        super().__init__()
        self._track_hp = False  # Disable hyperparam tracking for core hooks

    def before_stage(self) -> LoopControl | None:
        self.parent.model # Raise DeferHookExec if model is not initialized
        self.parent.model.to(self.parent.device)

    def stage(self) -> HookReturn:
        while True:
            self.parent._epoch_context = EpochContext()
            self.parent._step_context = None
            ctrl = self.parent.launch_event('epoch', LoopControl.SKIP_EPOCH)
            if ctrl > LoopControl.SKIP_EPOCH:
                return ctrl

    def epoch(self) -> HookReturn:
        for _ in self.parent.dataloader:
            self.parent.global_context.fetch += 1
            self.parent._step_context = StepContext()
            self.parent.step_context.batch = _
            ctrl = self.parent.launch_event('step', LoopControl.SKIP_STEP)
            if ctrl > LoopControl.SKIP_STEP:
                return ctrl

    def before_step(self) -> LoopControl | None:
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
        self.parent.step_context.batch = _move_to_device(
            self.parent.step_context.batch, self.parent.device
        )

    def step(self) -> HookReturn:
        for event in self.parent.event_sequence:
            ctrl = self.parent.launch_event(event, LoopControl.SKIP_EVENT)
            if ctrl > LoopControl.SKIP_EVENT:
                return ctrl

class _TrainerCore(BaseHook):
    ''' Core hook for training loops. '''

    def __init__(self):
        super().__init__()
        self._track_hp = False  # Disable hyperparam tracking for core hooks


    def before_step(self) -> HookReturn:
        self.parent.model.train()

    def forward(self) -> HookReturn:
        if TYPE_CHECKING:
            assert self.parent.step_context.batch is not None
        self.parent.step_context['losses'] = _auto_call(
            self.parent.model.loss, self.parent.step_context.batch # type: ignore
        )

    def after_forward(self) -> HookReturn:
        self.parent.global_context.forward += 1

    def backward(self) -> HookReturn:
        if TYPE_CHECKING:
            assert 'losses' in self.parent.step_context
        if self.parent.step_context.get('loss', None) is None:
            self.parent.step_context['loss'] = self.parent.step_context['losses'].mean()
        self.parent.step_context['loss'].backward()

    def after_backward(self) -> HookReturn:
        self.parent.global_context.backward += 1

    def after_optimizer_step(self) -> HookReturn:
        self.parent.global_context.update += 1

    def after_lr_scheduler_step(self) -> HookReturn:
        self.parent.global_context.scheduler_step += 1

    def after_step(self) -> HookReturn:
        self.parent.global_context.step += 1

    def finalize_epoch(self) -> HookReturn:
        self.parent.global_context.epoch += 1

class _EvaluatorCore(BaseHook):
    ''' Core hook for evaluation loops. '''

    def __init__(self):
        super().__init__()
        self._track_hp = False  # Disable hyperparam tracking for core hooks

    def check_epoch(self) -> LoopControl | None:
        self.parent._epoch_context = EpochContext()

    def before_epoch(self) -> HookReturn:
        self.parent.model.eval()

    def forward(self) -> HookReturn:
        if TYPE_CHECKING:
            assert self.parent.step_context.batch is not None
        with torch.inference_mode():
            self.parent.step_context['result'] = _auto_call(
                self.parent.model.predict, self.parent.step_context.batch  # type: ignore
            )

    def after_forward(self) -> HookReturn:
        self.parent.global_context.forward += 1

    def after_epoch(self) -> HookReturn:
        return LoopControl.SKIP_STAGE

    def finalize_epoch(self) -> HookReturn:
        self.parent.model.train()
        self.parent.global_context.epoch += 1

class BatchedModelLoop():
    '''
    Class to manage batched model loops during training or evaluation.
    '''
    def __init__(self):
        # Contexts
        self.model_context: ModelContext = ModelContext()
        self._global_context = GlobalContext()
        self._epoch_context: EpochContext | None = None
        self._step_context: StepContext | None = None
        self.exception: Exception | None = None

        # Hooks and events
        self.event_sequence: List[str] = []
        self.hooks: List[Tuple[int, BaseHook]] = []
        self._hook_sequence: Dict[str, OrderedDict[str, Callable[[], LoopControl | None]]] = {}

        # Execution level
        self.levels = {
            'stage': LoopControl.SKIP_STAGE,
            'epoch': LoopControl.SKIP_EPOCH,
            'step': LoopControl.SKIP_STEP,
        }

        # Initialization
        self._register_hook(_NestedLoopCore(), priority=-2)
        self.reset_context()
        self.reset_hook_cache()

    def reset_context(self) -> None:
        self.model_context = ModelContext()
        self._global_context = GlobalContext()
        self._epoch_context = None
        self._step_context = None
        self.exception = None

    def reset_hook_cache(self) -> None:
        self._hook_sequence.clear()

    @property
    def global_context(self) -> GlobalContext:
        return self._global_context

    @property
    def epoch_context(self) -> EpochContext:
        if self._epoch_context is None:
            raise DeferHookExec()
        return self._epoch_context

    @property
    def step_context(self) -> StepContext:
        if self._step_context is None:
            raise DeferHookExec()
        return self._step_context

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        if self.model_context.dataloader is None:
            raise DeferHookExec()
        return self.model_context.dataloader

    @property
    def device(self) -> torch.device:
        if self.model_context.device is None:
            raise DeferHookExec()
        return self.model_context.device

    @property
    def model(self) -> Module:
        if self.model_context.model is None:
            raise DeferHookExec()
        return self.model_context.model

    @property
    def optimizer(self) -> List[Optimizer]:
        if self.model_context.optimizer is None:
            raise DeferHookExec()
        return self.model_context.optimizer

    @property
    def lr_scheduler(self) -> List[LRScheduler] | None:
        if self.model_context.lr_scheduler is None:
            return None
        return self.model_context.lr_scheduler

    def launch_event(self, event_name: str, threshold: LoopControl = LoopControl.NONE):
        ctrl = LoopControl.NONE
        try:
            try:
                ctrl |= self._call_hooks(f'check_{event_name}')
                if ctrl >= threshold:
                    return ctrl
                # By design,
                # before_*, *, after_*, finally_* hooks should not return SKIP_*
                # However, occasionally users may want to enforce skipping.
                ctrl |= self._call_hooks(f'before_{event_name}')
                if ctrl >= threshold:
                    return ctrl
                ctrl |= self._call_hooks(event_name)
                if ctrl >= threshold:
                    return ctrl
                ctrl |= self._call_hooks(f'after_{event_name}')
                if ctrl >= threshold:
                    return ctrl
            finally:
                ret = self._call_hooks(f'finalize_{event_name}')
                if ret:
                    warnings.warn(
                        'Ignoring LoopControl returned by finalize hook.'
                    )
        except Exception as e:
            # Restoration is not supported
            if self.exception is None:
                self.exception = e
                if self._call_hooks('on_exception'):
                    warnings.warn(
                        'Ignoring LoopControl returned by on_exception hook.'
                    )
            raise

        return ctrl

    def register_hook(self, hook: BaseHook, priority: int | None = None) -> None:
        '''
        Register a hook with an optional priority.

        Args:
            hook: An instance of BaseHook to register.
            priority: An optional integer priority. Lower values indicate higher priority. Defaults to appending at the end.
        '''
        if priority is None:
            priority = self.hooks[-1][0] + 1 if self.hooks else 0
        if priority < 0:
            raise ValueError('Priority must be non-negative.')
        self._register_hook(hook, priority)

    def _register_hook(self, hook: BaseHook, priority: int) -> None:
        hook.parent = self
        bisect.insort(self.hooks, (priority, hook), key=lambda x: x[0])

    def _call_hooks(self, method_name: str) -> LoopControl:
        control = LoopControl.NONE
        exec_seq = OrderedDict()
        deferred_methods = self._hook_sequence.get(
            method_name, {
                type(hook).__name__: hook[method_name]
                for _, hook in self.hooks if method_name in hook
            }
        )
        while deferred_methods:
            new_deferred_methods = {}
            for name, method in deferred_methods.items():
                try:
                    ret = method()
                except DeferHookExec:
                    new_deferred_methods[name] = method
                    continue
                exec_seq[name] = method
                if ret is not None:
                    control |= ret
            if len(deferred_methods) == len(new_deferred_methods):
                # No progress made, break to avoid infinite loop
                failed_hooks = ', '.join(new_deferred_methods.keys())
                raise RuntimeError(
                    f'Cannot resolve dependencies for {method_name} of hook {failed_hooks}.'
                )
            deferred_methods = new_deferred_methods

        self._hook_sequence[method_name] = exec_seq
        return control

    def run(
        self, level: Literal['stage', 'epoch', 'step'] = 'stage',
        cleanup: bool = True
    ) -> None:
        '''
        Run the specified level of the loop.
        '''
        if not self.event_sequence:
            raise ValueError('No events registered to run.')
        if cleanup:
            self.reset_context()
            self.reset_hook_cache()

        self.launch_event(level, self.levels[level])

class Trainer(BatchedModelLoop):
    '''
    Trainer class to manage training loops with hooks and contexts.
    '''
    def __init__(self):
        super().__init__()
        self.event_sequence = [
            'forward', 'backward', 'optimizer_step', 'lr_scheduler_step'
        ]
        self._register_hook(_TrainerCore(), priority=-1)

    def train(self):
        ''' Run the training loop. '''
        self.run('stage')

class Evaluator(BatchedModelLoop):
    '''
    Evaluator class to manage evaluation loops with hooks and contexts.

    After evaluation, metrics should be written to epoch_context['metrics']: Dict[str, Any] in `finalize_epoch` hook.
    '''

    def __init__(self):
        super().__init__()
        self.event_sequence = ['forward']
        self._initialized = False
        self._register_hook(_EvaluatorCore(), priority=-1)

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        ''' Interface to set evaluation metrics after evaluation. '''
        self.epoch_context['metrics'] = metrics

    def get_metrics(self) -> Dict[str, Any]:
        ''' Interface to get evaluation metrics after evaluation. '''
        if 'metrics' not in self.epoch_context:
            raise ValueError('Metrics have not been set for the current epoch.')
        return self.epoch_context['metrics']

    def evaluate(self):
        ''' Run the evaluation loop. '''
        if not self._initialized:
            self.run('stage', cleanup=False) # Keep context for evaluation
        else:
            self.run('epoch', cleanup=False)
        self._initialized = True
