from typing import Any, AsyncGenerator, Callable, Dict, Generator, Iterable, List, Tuple

import gymnasium
import torch
import enum
import numpy as np

from .interface import EnvironmentInfo, RewardMapping, _default_shape
from .model import BaseRLModel
from .data import Trajectory
from ...mp_sync import OneToMany, ManyToOne, mp, shared_tensor


class WorkerSignal(enum.IntEnum):
    ACTION = 1
    RESET = 2
    EXIT = 3


class WorkerResponse(enum.IntEnum):
    NORMAL = 1
    DONE = 2


class ResetSignal(Exception):
    pass


class ExitSignal(Exception):
    pass


class _SyncObjs():
    def __init__(
        self,
        n_workers: int,
        env_info: EnvironmentInfo,
    ):
        self.n_workers = n_workers
        self.state_shape = env_info.state_shape
        self.action_shape = env_info.action_shape
        self.action_dtype = env_info.action_dtype

        # Per-worker current-step buffer
        self.states = shared_tensor((n_workers, *self.state_shape))
        self.actions = shared_tensor((n_workers, *self.action_shape), self.action_dtype)
        self.log_pi = shared_tensor((n_workers,))

        # Model -> Worker synchronization
        self.worker_task = OneToMany(n_workers)

        # Input
        self.task_queue = mp.Queue()

        # Output
        self.output_buffer = shared_tensor((
            n_workers, env_info.max_len, env_info.buffer_dim
        ))
        self.output_len = shared_tensor((n_workers,), torch.int32)
        self.main_task = ManyToOne(n_workers)


class EnvProcess(mp.Process):
    def __init__(self, worker_id: int, sync: _SyncObjs, env_info: EnvironmentInfo):
        super().__init__(daemon=True)
        self.sync = sync
        self.worker_id = worker_id
        self.env_info = env_info
        self.output_buffer = self.sync.output_buffer[worker_id]

    def get_action(self) -> Tuple[np.ndarray, np.ndarray]:
        sig = self.sync.worker_task.wait_do(self.worker_id)
        match WorkerSignal(sig):
            case WorkerSignal.ACTION:
                action = self.sync.actions[self.worker_id].numpy().copy()
                log_pi = self.sync.log_pi[self.worker_id].numpy().copy()
                return action, log_pi
            case WorkerSignal.RESET:
                raise ResetSignal()
            case WorkerSignal.EXIT:
                raise ExitSignal()

    def set_state(self, state: np.ndarray | None):
        if state is None:
            self.sync.worker_task.done(self.worker_id, WorkerResponse.DONE)
        else:
            np_view = self.sync.states.numpy()
            np_view[self.worker_id] = state
            self.sync.worker_task.done(self.worker_id, WorkerResponse.NORMAL)

    def run(self):
        torch.set_num_threads(1)
        env = None
        sd = self.env_info.slice_dict
        traj_done = False
        try:
            output_buffer_np = self.output_buffer.numpy()
            env = self.env_info.env_fn()
            while True:
                step = 0
                state, _ = env.reset()
                if traj_done:
                    self.sync.main_task.wait_done(self.worker_id)
                    traj_done = False

                self.output_buffer.zero_()
                output_buffer_np[step, sd['state']] = state.flatten()
                self.set_state(state)
                try:
                    while True:
                        action, log_pi = self.get_action()
                        next_state, reward, term, trunc, _ = env.step(action)
                        output_buffer_np[step, sd['next_state']] = next_state.flatten()
                        output_buffer_np[step, sd['action']] = action.flatten()
                        output_buffer_np[step, sd['reward']] = reward
                        output_buffer_np[step, sd['term']] = term
                        output_buffer_np[step, sd['trunc']] = trunc | ((step + 1) >= self.env_info.max_len)
                        output_buffer_np[step, sd['log_pi']] = log_pi
                        step += 1
                        if term or trunc or step >= self.env_info.max_len:
                            self.sync.output_len[self.worker_id] = step
                            self.sync.main_task.do(self.worker_id)
                            traj_done = True
                            self.set_state(None)
                            break
                        else:
                            output_buffer_np[step, sd['state']] = next_state.flatten()
                            self.set_state(next_state)
                except ResetSignal:
                    continue
                except ExitSignal:
                    break
        except:
            raise
        finally:
            if env is not None:
                env.close()


class ModelProcess(mp.Process):
    def __init__(
        self,
        model: BaseRLModel,
        num_workers: int,
        sync: _SyncObjs,
        env_info: EnvironmentInfo,
        device: torch.device | None = None,
        shared_params: Dict[str, Any] | None = None,
        min_batch: int = 1,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.min_batch = min_batch
        self.num_workers = num_workers
        self.shared_params = shared_params
        self.device = device if device is not None else model.device
        self.sync = sync
        self.env_info = env_info

        # Task status tracking
        self.remaining_trajs = 0
        # Worker status tracking
        self.reset_workers = set()

    def get_state(
        self, min_batch: int | None
    ):
        result, _ = self.sync.worker_task.wait_done(min_batch)
        done_workers, alive_workers = [], []
        for i, f in result:
            if f == WorkerResponse.DONE:
                done_workers.append(i)
            elif f == WorkerResponse.NORMAL:
                alive_workers.append(i)
        alive_idx = self.sync.states.new_tensor(alive_workers, dtype=torch.long)
        states = self.sync.states.index_select(0, alive_idx)
        self.reset_workers.update(done_workers)
        self.remaining_trajs -= len(done_workers)
        return states, alive_workers, alive_idx

    def set_action(
        self, action: torch.Tensor, log_pi: torch.Tensor,
        alive_idx: torch.Tensor, alive_ids: Iterable[int]
    ):
        self.sync.actions.index_copy_(0, alive_idx, action.cpu())
        self.sync.log_pi.index_copy_(0, alive_idx, log_pi.cpu())
        self.sync.worker_task.do(alive_ids, WorkerSignal.ACTION)

    def set_reset(self):
        if not self.reset_workers:
            return
        self.sync.worker_task.do(self.reset_workers, WorkerSignal.RESET)
        self.reset_workers.clear()

    def set_exit(self):
        self.sync.worker_task.do(None, WorkerSignal.EXIT)

    def cleanup_stale(self, pending: Iterable[int] = ()):
        # Clean up any stale trajectories that have not been collected
        cleaned = set()
        pending = set(pending)

        while pending:
            worker_id, _ = self.sync.main_task.wait_do()
            self.sync.main_task.done(worker_id)
            cleaned.add(worker_id)
            pending.discard(worker_id)

        while True:
            task = self.sync.main_task.nowait_do()
            if task is None:
                break

            worker_id, _ = task
            self.sync.main_task.done(worker_id)
            cleaned.add(worker_id)

        return cleaned

    def run(self):
        if self.device.type != 'cpu':
            torch.set_num_threads(1)
        self.model.to(self.device)

        while True:
            obj = self.sync.task_queue.get()
            if obj is None:
                # Wake up all workers to exit
                cleaned_workers = self.cleanup_stale()
                self.get_state(None)
                self.reset_workers -= cleaned_workers
                self.cleanup_stale(self.reset_workers)
                self.set_exit()
                break
            (state_dict, num_trajs, act_kwargs, reset) = obj
            if state_dict is None:
                state_dict = self.shared_params
            if state_dict is not None:
                self.model.load_state_dict({
                    k: v.to(self.device)
                    for k, v in state_dict.items()
                }, strict=True)
            # Notify workers
            if reset:
                self.reset_workers.update(range(self.num_workers))
                self.set_reset()
                self.remaining_trajs = num_trajs
            else:
                self.remaining_trajs += num_trajs
            min_batch = min(self.remaining_trajs, self.min_batch)

            while self.remaining_trajs > 0:
                states, alive_workers, alive_idx = self.get_state(min_batch)
                self.set_reset()
                if alive_workers:
                    action, log_pi = self.model.act(states.to(self.device), **act_kwargs)
                    self.set_action(action, log_pi, alive_idx, alive_workers)

        del self.model


class AsyncEnvPool():
    def __init__(
        self,
        init_model: BaseRLModel,
        env_fn: Callable[[], gymnasium.Env],
        num_workers: int = 1,
        inference_device: torch.device | str | None = None,
        enable_shared_params: bool = False,
        max_len: int | None = None,
        min_batch: int = 1,
    ):
        self.env_info = EnvironmentInfo(env_fn=env_fn, max_len=max_len)
        self.sync = _SyncObjs(num_workers, self.env_info)
        if inference_device is None:
            inference_device = init_model.device
        self.inference_device = torch.device(inference_device)
        self.env_workers = [
            EnvProcess(i, self.sync, self.env_info)
            for i in range(num_workers)
        ]
        for w in self.env_workers:
            w.start()

        if enable_shared_params:
            self.shared_params = {
                k: v.share_memory_()
                for k, v in init_model.state_dict().items()
            }
        else:
            self.shared_params = None

        self.model_worker = ModelProcess(
            init_model, num_workers, self.sync, self.env_info,
            device=torch.device(
                inference_device) if inference_device is not None else None,
            shared_params=self.shared_params,
            min_batch=min(min_batch, num_workers)
        )
        self.model_worker.start()
        self.tau, self.gamma = init_model.tau, init_model.gamma

    def _check_alive(self):
        return self.model_worker.is_alive() and \
            all(p.is_alive() for p in self.env_workers)

    def _pre_process(
        self, model: BaseRLModel | None, num_trajectories: int, reset: bool = True,
        **act_kwargs
    ):
        if not self._check_alive():
            raise RuntimeError("One or more worker processes have exited unexpectedly.")
        if num_trajectories <= 0:
            raise ValueError(f'num_trajectories must be > 0, got {num_trajectories}')
        if model is None:
            put = None
        elif (
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

        self.sync.task_queue.put((put, num_trajectories, act_kwargs, reset))

    def _get_trajectory(
        self, worker_id: int,
        model: BaseRLModel | None, reward_shape: RewardMapping,
        output_device: torch.device | None = None
    ):
        output_len = self.sync.output_len[worker_id].item()
        if output_device is None or output_device.type == 'cpu':
            buf_clone = self.sync.output_buffer[worker_id, :output_len].clone()
        else:
            buf_clone = self.sync.output_buffer[worker_id, :output_len].to(
                output_device)
        total_reward = buf_clone[
            :, self.env_info.slice_dict['reward']
        ].sum().item()
        self.sync.main_task.done(worker_id)
        traj = Trajectory(
            buf_clone,
            self.env_info.state_shape,
            self.env_info.action_shape,
            self.env_info.action_dtype,
            total_reward
        )
        if reward_shape is not _default_shape:
            traj.shape_reward(reward_shape)
        return traj.tau_step(self.tau, self.gamma)

    def get_trajectory(
        self, model: BaseRLModel | None, reward_shape: RewardMapping,
        output_device: torch.device | None = None
    ):
        worker_id, _ = self.sync.main_task.wait_do()
        return self._get_trajectory(worker_id, model, reward_shape, output_device)

    async def get_trajectory_async(
        self, model: BaseRLModel | None, reward_shape: RewardMapping,
        output_device: torch.device | None = None
    ):
        worker_id, _ = await self.sync.main_task.async_wait_do()
        return self._get_trajectory(worker_id, model, reward_shape, output_device)

    def yield_episode(
        self,
        model: BaseRLModel | None,
        num_trajectories: int = 1,
        reward_shape: RewardMapping = _default_shape,
        output_device: torch.device | None = None,
        reset: bool = True,
        **act_kwargs,
    ) -> Generator[Trajectory, None, None]:
        self._pre_process(model, num_trajectories, reset, **act_kwargs)
        for _ in range(num_trajectories):
            yield self.get_trajectory(model, reward_shape, output_device)

    async def yield_episode_async(
        self,
        model: BaseRLModel | None,
        num_trajectories: int = 1,
        reward_shape: RewardMapping = _default_shape,
        output_device: torch.device | None = None,
        reset: bool = True,
        **act_kwargs,
    ) -> AsyncGenerator[Trajectory, None]:
        self._pre_process(model, num_trajectories, reset, **act_kwargs)
        for _ in range(num_trajectories):
            yield await self.get_trajectory_async(model, reward_shape, output_device)

    def run_episode(
        self,
        model: BaseRLModel | None,
        num_trajectories: int = 1,
        reward_shape: RewardMapping = _default_shape,
        output_device: torch.device | None = None,
        reset: bool = True,
        **act_kwargs,
    ) -> List[Trajectory]:
        return [
            *self.yield_episode(
                model, num_trajectories, reward_shape, output_device, reset, **act_kwargs
            )
        ]

    async def run_episode_async(
        self,
        model: BaseRLModel | None,
        num_trajectories: int = 1,
        reward_shape: RewardMapping = _default_shape,
        output_device: torch.device | None = None,
        reset: bool = True,
        **act_kwargs
    ) -> List[Trajectory]:
        ret = []
        async for traj in self.yield_episode_async(
            model, num_trajectories, reward_shape, output_device, reset, **act_kwargs
        ):
            ret.append(traj)
        return ret

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        if hasattr(self, 'shared_params') and self.shared_params is not None:
            del self.shared_params
        if not hasattr(self, 'sync'):
            return
        self.sync.task_queue.put(None)
        self.model_worker.join(5)
        if self.model_worker.is_alive() or any(p.is_alive() for p in self.env_workers):
            for p in self.env_workers:
                if p.is_alive():
                    p.terminate()
                    p.join(1)
            self.model_worker.terminate()
            self.model_worker.join(1)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        self.close()
