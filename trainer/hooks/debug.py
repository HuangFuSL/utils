import cProfile
import os
import profile
import pstats
from typing import Literal

import torch
from torch.profiler import profile as torch_profile

from .. import BaseHook, LoopControl


class SuppressExceptionHook(BaseHook):
    def on_exception(self) -> LoopControl | None:
        return LoopControl.SKIP_STEP

class PythonProfileHook(BaseHook):
    '''
    Hook to profile each training step using cProfile
    '''
    def __init__(
        self, num_steps: int | None = None,
        export_path: str | None = None, write_to: str = 'python_profile',
        kernel: Literal['c', 'python'] = 'c'
    ):
        self.num_steps = num_steps
        self.export_path = export_path
        self.write_to = write_to
        self.kernel = kernel


    def before_stage(self) -> LoopControl | None:
        if self.kernel == 'c':
            self.parent.global_context[self.write_to] = cProfile.Profile()
        else:
            self.parent.global_context[self.write_to] = profile.Profile()
        self.parent.global_context[self.write_to].enable()

    def check_step(self) -> LoopControl | None:
        if self.num_steps is not None and \
            self.parent.global_context.step >= self.num_steps:
            return LoopControl.SKIP_STAGE

    def finalize_stage(self) -> LoopControl | None:
        self.parent.global_context[self.write_to].disable()
        if self.export_path is not None:
            if not os.path.exists(self.export_path):
                os.makedirs(self.export_path)
            with open(f"{self.export_path}/profile.txt", 'w') as f:
                ps = pstats.Stats(self.parent.global_context[self.write_to], stream=f)
                ps.strip_dirs().sort_stats('tottime').print_stats()

class TorchProfileHook(BaseHook):
    '''
    Hook to profile each training step using torch.profiler.

    Args:
        activities: List of ProfilerActivity to be profiled.
        record_shapes: Whether to record shapes of the tensors.
        with_stack: Whether to record stack traces.
    '''

    def __init__(
        self, num_steps: int | None = None, export_path: str | None = None,
        write_to: str = 'torch_profile',
        **kwargs
    ):
        self.num_steps = num_steps
        self.export_path = export_path
        self.write_to = write_to
        self.prof_kwargs = kwargs

    def before_stage(self) -> LoopControl | None:
        self.parent.global_context[self.write_to] = torch_profile(
            experimental_config=torch._C._profiler._ExperimentalConfig(
                verbose=True),
            **self.prof_kwargs
        )
        self.parent.global_context[self.write_to].__enter__()

    def check_step(self) -> LoopControl | None:
        if self.num_steps is not None and \
            self.parent.global_context.step >= self.num_steps:
            return LoopControl.SKIP_STAGE

    def finalize_step(self) -> LoopControl | None:
        self.parent.global_context[self.write_to].step()

    def finalize_stage(self) -> LoopControl | None:
        exc = self.parent.exception
        exc_type = None if exc is None else type(exc)
        tb = None if exc is None else exc.__traceback__
        self.parent.global_context[self.write_to].__exit__(exc_type, exc, tb)
        if self.export_path is not None:
            self.parent.global_context[self.write_to].export_chrome_trace(
                f"{self.export_path}/trace.json"
            )
