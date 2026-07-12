from __future__ import annotations

import asyncio
import dataclasses
import enum
import multiprocessing
import multiprocessing.synchronize
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict

import gymnasium
import torch

from .data import Trajectory
from .interface import RewardMapping, _default_shape, EnvironmentInfo
from .model import BaseRLModel

if TYPE_CHECKING:
    mp = multiprocessing
    torch_mp = torch.multiprocessing
else:
    if sys.platform == 'darwin' or sys.platform.startswith('win'):
        mp = multiprocessing.get_context('spawn')
        torch_mp = torch.multiprocessing.get_context('spawn')
    else:
        mp = multiprocessing.get_context('forkserver')
        torch_mp = torch.multiprocessing.get_context('forkserver')

class Status(enum.IntEnum):
    STEP = 0
    DONE = 1
    ALL_DONE = 2
    SHOULD_EXIT = 3

@dataclasses.dataclass(init=False)
class _SyncObjs():
    # Barriers
    env_ready: multiprocessing.synchronize.Barrier
    model_ready: multiprocessing.synchronize.Barrier
    action_ready: multiprocessing.synchronize.Barrier
    state_ready: multiprocessing.synchronize.Barrier
    trajectory_done: multiprocessing.synchronize.Barrier
    # Queues
    task_queue: torch.multiprocessing.JoinableQueue
    # Events
    buffer_event: multiprocessing.synchronize.Event

    # Shared tensor
    buffer: torch.Tensor
    status: torch.Tensor

    def __init__(
        self, num_workers: int, buffer: torch.Tensor, status: torch.Tensor
    ):
        # Env + Model uses the following
        self.model_ready = mp.Barrier(num_workers + 1)
        self.action_ready = mp.Barrier(num_workers + 1)
        self.state_ready = mp.Barrier(num_workers + 1)

        # Env + Main uses the following
        self.env_ready = mp.Barrier(num_workers + 1)
        self.trajectory_done = mp.Barrier(num_workers + 1)

        # Model + Main uses the following
        self.task_queue = torch_mp.JoinableQueue()
        self.buffer_event = mp.Event()

        # Shared tensor
        self.buffer = buffer
        self.status = status

    def abort(self):
        self.model_ready.abort()
        self.action_ready.abort()
        self.state_ready.abort()
        self.env_ready.abort()
        self.trajectory_done.abort()


class EnvProcess(mp.Process):
    def __init__(self, worker_id: int, sync: _SyncObjs, env_info: EnvironmentInfo):
        super().__init__(daemon=True)
        self.sync = sync
        self.worker_id = worker_id
        self.env_info = env_info
        self.done = False
        self.step = 0

    def sync_step(self):
        status = self.sync.status[self.worker_id]
        status[Status.STEP] = min(self.step, self.env_info.max_len - 1)
        status[Status.DONE] = int(self.done)

    def run(self):
        # Stage 1: Initialize environment and signal readiness
        torch.set_num_threads(1)
        env = buffer = status = buffer_np = None
        try:
            sd = self.env_info.slice_dict
            buffer = self.sync.buffer[self.worker_id]
            status = self.sync.status[self.worker_id]
            buffer_np = buffer.numpy()
            env = self.env_info.env_fn()
            self.sync.env_ready.wait()

            while True:
                # Step 1: Wait for model
                try:
                    self.sync.model_ready.wait()
                except threading.BrokenBarrierError:
                    break
                if status[Status.SHOULD_EXIT].item():
                    break

                # Step 2: After model is ready, reset the environment, signal state
                self.step, self.done = 0, status[Status.DONE].item()
                if not self.done:
                    buffer_np[self.step, sd['state']] = env.reset()[0].flatten()
                self.sync.state_ready.wait()

                while True:
                    # Step 3: Wait for action
                    self.sync.action_ready.wait()
                    if status[Status.ALL_DONE].item():
                        break # Remember: Break after action_ready
                    if self.done:
                        if self.step < self.env_info.max_len:
                            self.step += 1
                        self.sync_step()
                        self.sync.state_ready.wait()
                        continue

                    action = buffer_np[self.step, sd['action']] \
                        .reshape(*self.env_info.action_shape) \
                        .astype(self.env_info.action_dtype_np)

                    # Step 4: Step the environment, signal state
                    next_state, reward, terminated, truncated, info = env.step(action)
                    self.step += 1
                    if self.step < self.env_info.max_len:
                        buffer_np[self.step, sd['state']] = next_state.flatten()
                    self.done = bool(terminated or truncated or self.step >= self.env_info.max_len)
                    self.sync_step()
                    self.sync.state_ready.wait()

                    # Step 5: Write other info to buffer
                    buffer_np[self.step - 1, sd['next_state']] = next_state.flatten()
                    buffer_np[self.step - 1, sd['reward']] = reward
                    buffer_np[self.step - 1, sd['term']] = int(terminated)
                    buffer_np[self.step - 1, sd['trunc']] = int(truncated)

                # After exit
                self.sync.trajectory_done.wait()
        except Exception:
            self.sync.abort()
            raise
        finally:
            del buffer_np
            del buffer
            del status
            if env is not None:
                env.close()

class ModelProcess(mp.Process):
    def __init__(
        self, model: BaseRLModel,
        sync: _SyncObjs,
        env_info: EnvironmentInfo,
        device: torch.device | None = None,
        shared_params: Dict[str, Any] | None = None,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.shared_params = shared_params
        self.device = device if device is not None else model.device
        self.sync = sync
        self.env_info = env_info

    def batch_act(self, model: BaseRLModel, **act_kwargs):
        buffer, status = self.sync.buffer, self.sync.status
        n = int((status[:, Status.DONE] == 0).sum().item())

        sd = self.env_info.slice_dict
        state_shape = self.env_info.state_shape
        action_numel = self.env_info.action_numel

        if n > 0:
            step = int(status[:, Status.STEP].min().item())
            states = buffer[:, step, sd['state']] \
                .to(model.device).reshape(-1, *state_shape)

            with torch.no_grad():
                actions, log_pis = model.act(states, **act_kwargs)

            a_sl, lp_sl = sd['action'], sd['log_pi']
            buffer[:, step, a_sl].copy_(actions.to(torch.float32).reshape(-1, action_numel))
            self.sync.action_ready.wait()
            buffer[:, step, lp_sl].copy_(log_pis.reshape(-1, 1))
        else:
            status[:, Status.ALL_DONE] = 1
            self.sync.action_ready.wait()
        return n == 0

    def run(self):
        if self.device.type != 'cpu':
            torch.set_num_threads(1)
        self.model.to(self.device)

        sd = self.env_info.slice_dict
        buffer, status = self.sync.buffer, self.sync.status

        try:
            while True:
                obj = self.sync.task_queue.get()

                if obj is None:
                    status[:, Status.SHOULD_EXIT] = 1
                    self.sync.model_ready.wait() # Signal env workers to exit
                    self.sync.task_queue.task_done()
                    break

                state_dict, num_trajectories, act_kwargs = obj
                if state_dict is None:
                    state_dict = self.shared_params
                if state_dict is not None:
                    self.model.load_state_dict({
                        k: v.to(self.device)
                        for k, v in state_dict.items()
                    }, strict=True)
                # Use the model's current parameters by default.

                left = 0
                while left < num_trajectories:
                    right = min(left + buffer.shape[0], num_trajectories)
                    batch_size = right - left

                    buffer[:batch_size, :, sd['term']].zero_()
                    buffer[:batch_size, :, sd['trunc']].zero_()
                    buffer[:batch_size, :, sd['reward']].zero_()
                    buffer[:batch_size, :, sd['log_pi']].zero_()
                    status.zero_()
                    status[:batch_size, Status.DONE] = 0
                    status[batch_size:, Status.DONE] = 1
                    self.sync.model_ready.wait()

                    while True:
                        self.sync.state_ready.wait()
                        if self.batch_act(self.model, **act_kwargs):
                            break

                    left = right

                    self.sync.buffer_event.wait() # Wait for main to read the buffer
                    self.sync.buffer_event.clear()
                self.sync.task_queue.task_done()
        except Exception:
            self.sync.abort()
            raise
        finally:
            del buffer
            del status


class SyncedEnvPool():
    def __init__(
        self,
        init_model: BaseRLModel,
        env_fn: Callable[[], gymnasium.Env],
        num_workers: int = 1,
        inference_device: torch.device | str | None = None,
        enable_shared_params: bool = False,
        max_len: int | None = None
    ):
        if num_workers < 0:
            raise ValueError(f'num_workers must be >= 0, got {num_workers}')
        elif num_workers == 0:
            num_workers = max(mp.cpu_count() // 2, 1)
        self.num_workers = num_workers
        if inference_device is None:
            inference_device = init_model.device
        self.inference_device = torch.device(inference_device)
        if enable_shared_params:
            self.shared_params = init_model.state_dict()
            for k, v in self.shared_params.items():
                self.shared_params[k] = v.cpu()
                self.shared_params[k].share_memory_()
        else:
            self.shared_params = None

        # Parse env:
        self.env_info = EnvironmentInfo(env_fn=env_fn, max_len=max_len)
        B, L, D = self.num_workers, self.env_info.max_len, self.env_info.buffer_dim
        self.B, self.L, self.D = B, L, D

        # Allocate buffer
        self.buffer = torch.zeros((B, L, D), dtype=torch.float32)
        self.status_buffer = torch.zeros((B, len(Status)), dtype=torch.int32)
        self.buffer.share_memory_()
        self.status_buffer.share_memory_()

        # Start processes
        self.sync = _SyncObjs(B, self.buffer, self.status_buffer)
        args = (self.sync, self.env_info)
        self._async_lock = asyncio.Lock()
        self.model_worker = ModelProcess(
            init_model, *args,
            device=self.inference_device,
            shared_params=self.shared_params
        )
        self.env_workers = [EnvProcess(i, *args) for i in range(B)]
        for p in self.env_workers + [self.model_worker]:
            p.start()
        try:
            self.sync.env_ready.wait(30)
            assert self._check_alive()
        except Exception:
            self.sync.abort()
            self.close()
            raise RuntimeError('Environment workers failed to initialize.')

    def run_episode(
        self,
        model: BaseRLModel,
        num_trajectories: int = 1,
        reward_shape: RewardMapping = _default_shape,
        output_device: torch.device | None = None,
        **act_kwargs
    ):
        output_buffer = self._pre_process(model, num_trajectories, output_device, **act_kwargs)
        for _ in self._batch_gen(output_buffer):
            self.sync.trajectory_done.wait()
        return self._post_process(output_buffer, model, reward_shape)

    async def async_run_episode(
        self, model: BaseRLModel,
        num_trajectories: int = 1,
        reward_shape: RewardMapping = _default_shape,
        output_device: torch.device | None = None, **act_kwargs
    ):
        async with self._async_lock:
            output_buffer = self._pre_process(model, num_trajectories, output_device, **act_kwargs)
            for _ in self._batch_gen(output_buffer):
                await asyncio.to_thread(self.sync.trajectory_done.wait)
            return self._post_process(output_buffer, model, reward_shape)

    def _check_alive(self):
        return self.model_worker.is_alive() and \
            all(p.is_alive() for p in self.env_workers)

    def _pre_process(
        self, model: BaseRLModel, num_trajectories: int,
        output_device: torch.device | None = None,
        **act_kwargs
    ):
        if not self._check_alive():
            raise RuntimeError('One or more worker processes have exited unexpectedly.')
        if num_trajectories <= 0:
            raise ValueError(f'num_trajectories must be > 0, got {num_trajectories}')
        device = output_device if output_device is not None else model.device
        output_buffer = torch.zeros(
            (num_trajectories, self.L, self.D),
            dtype=torch.float32, device=device
        )
        if (
            model.device.type != 'cuda'
            or self.inference_device.type != 'cuda'
            or model.device.index != self.inference_device.index
        ) and self.shared_params is not None:
            state_dict = model.state_dict()
            for name in state_dict:
                self.shared_params[name].copy_(state_dict[name])
            put = None
        else:
            put = model.state_dict()
        self.sync.task_queue.put((put, num_trajectories, act_kwargs))
        return output_buffer

    def _batch_gen(self, output_buffer: torch.Tensor):
        B, N = self.B, output_buffer.shape[0]
        left = 0
        while left < N:
            right = min(left + B, N)
            batch_size = right - left

            yield # Breakpoint for trajectory_done barrier
            output_buffer[left:right] = self.buffer[:batch_size]
            left = right
            self.sync.buffer_event.set() # Signal model process to continue

    def _post_process(
        self,
        buffer: torch.Tensor,
        model: BaseRLModel,
        reward_shape: RewardMapping
    ):
        B, L, D = buffer.shape
        sd = self.env_info.slice_dict
        state_shape = self.env_info.state_shape
        action_shape = self.env_info.action_shape
        action_dtype = self.env_info.action_dtype

        raw_rewards = buffer[..., sd['reward']].sum(dim=(1, 2))

        Trajectory._shape_reward(buffer, reward_shape, state_shape, action_shape, sd)
        buffer = Trajectory._tau_step(buffer, model.tau, model.gamma, sd)

        trajectories = []
        done_flag = ((
            buffer[:, :, sd['term']] + buffer[:, :, sd['trunc']]
        ) > 0).squeeze(-1).to(torch.long)
        has_done = done_flag.any(dim=1)
        traj_len = done_flag.argmax(dim=1) + 1
        traj_len[~has_done] = L
        traj_len = traj_len.tolist()
        total_rewards = raw_rewards.tolist()

        for i, (length, r) in enumerate(zip(traj_len, total_rewards)):
            trajectories.append(Trajectory(
                buffer[i, :length],
                state_shape, action_shape, action_dtype, r
            ))

        return trajectories

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        if self.shared_params is not None:
            del self.shared_params
        self.sync.buffer_event.set()
        self.sync.task_queue.put(None)
        self.model_worker.join(5)
        while self.sync.task_queue.qsize() > 0:
            self.sync.task_queue.get()
            self.sync.task_queue.task_done()
        while True:
            try:
                self.sync.task_queue.task_done()
            except:
                break
        self.sync.task_queue.join()
        if self.model_worker.is_alive() or any(p.is_alive() for p in self.env_workers):
            self.sync.abort()
            for p in self.env_workers:
                if p.is_alive():
                    p.terminate()
                    p.join(1)
            self.model_worker.terminate()
            self.model_worker.join(1)
        del self.buffer
        del self.status_buffer

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()
