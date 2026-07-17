from __future__ import annotations

import asyncio
import multiprocessing
import queue
import sys
from typing import TYPE_CHECKING, Iterable, List, Tuple

import torch

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


def shared_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.zeros(shape, dtype=dtype, device='cpu')
    tensor.share_memory_()
    return tensor


class OneToMany():
    '''
    A synchronization primitive that allows one master process to communicate with many worker processes.

    Args:
        n (int): The number of worker processes.
    '''
    def __init__(self, n: int):
        self.n = n
        self.flag = mp.RawArray('i', n * 2)
        self.do_cond = mp.Condition(mp.Lock())
        self.done_cond = mp.Condition(mp.Lock())
        self.done_count = mp.RawValue('i', 0)

    def _do_index(self, worker: int) -> int:
        return worker

    def _done_index(self, worker: int) -> int:
        return worker + self.n

    # Master -> Worker
    def do(self, worker: int | Iterable[int] | None = None, value: int = 1):
        '''
        Called by the master process to notify one or more worker processes to perform a task.

        Args:
            worker (int | List[int] | None): The index of the workers to notify. If None, notify all workers.
            value (int): The task index for the workers to perform. Default is 1.
        '''
        with self.do_cond:
            if isinstance(worker, int):
                self.flag[self._do_index(worker)] = value
            elif worker is None:
                self.flag[:self.n] = [value] * self.n
            else:
                for w in worker:
                    self.flag[self._do_index(w)] = value
            self.do_cond.notify_all()

    def nowait_do(self, worker: int) -> int | None:
        '''
        Called by the worker process to check for a task notification from the master process without blocking.

        Args:
            worker (int): The index of the worker process.

        Returns:
            int | None: The task index that the worker process should perform, or None if no task is available.
        '''
        with self.do_cond:
            if self.flag[self._do_index(worker)]:
                ret = self.flag[self._do_index(worker)]
                self.flag[self._do_index(worker)] = 0
                return ret
            else:
                return None

    def wait_do(self, worker: int) -> int:
        '''
        Called by the worker process to wait for a task notification from the master process.

        Args:
            worker (int): The index of the worker process.

        Returns:
            int: The task index that the worker process should perform.
        '''
        with self.do_cond:
            self.do_cond.wait_for(lambda: bool(self.flag[self._do_index(worker)]))
            ret = self.flag[self._do_index(worker)]
            self.flag[self._do_index(worker)] = 0
            return ret

    async def async_wait_do(self, worker: int):
        '''
        Called by the worker process to asynchronously wait for a task notification from the master process.

        Args:
            worker (int): The index of the worker process.
        '''
        return await asyncio.to_thread(self.wait_do, worker)

    # Worker -> Master
    def done(self, worker: int, result: int = 1):
        '''
        Called by the worker process to notify the master process that it has completed its task.

        Args:
            worker (int): The index of the worker process that has completed its task.
            result (int): The result of the completed task. Default is 1.
        '''
        if not result:
            return
        with self.done_cond:
            self.done_count.value += (1 - bool(self.flag[self._done_index(worker)]))
            self.flag[self._done_index(worker)] = result
            self.done_cond.notify()

    def nowait_done(self) -> List[Tuple[int, int]]:
        '''
        Called by the master process to instantly collect the completed workers

        Returns:
            List[Tuple[int, int]]: A list of tuples containing the indices of the worker processes that have completed their tasks and the results of those tasks.
        '''
        with self.done_cond:
            done = [i for i in enumerate(self.flag[self.n:]) if i[1]]
            if done:
                self.done_count.value = 0
                for w in done:
                    self.flag[self._done_index(w[0])] = 0
            return done


    def wait_done(self, min_workers: int | None = 1, timeout: float | None = None) -> Tuple[List[Tuple[int, int]], bool]:
        '''
        Called by the master process to wait for a minimum number of worker processes to complete their tasks.

        Args:
            min_workers (int): The minimum number of worker processes to wait for. Default is 1.
            timeout (float | None): The maximum time to wait for the workers to complete their tasks. If None, wait indefinitely. Default is None.

        Returns:
            Tuple[List[Tuple[int, int]], bool]: A tuple containing a list of tuples with the indices of the worker processes that have completed their tasks and the results of those tasks, and a boolean indicating whether the wait was successful (True) or timed out (False).
        '''
        if min_workers is None:
            min_workers = self.n
        with self.done_cond:
            success = self.done_cond.wait_for(
                lambda: self.done_count.value >= min_workers,
                timeout=timeout
            )
            done = [i for i in enumerate(self.flag[self.n:]) if i[1]]
            self.done_count.value = 0
            for w in done:
                self.flag[self._done_index(w[0])] = 0
            return done, success

    async def async_wait_done(self, min_workers: int = 1, timeout: float | None = None) -> Tuple[List[Tuple[int, int]], bool]:
        '''
        Called by the master process to asynchronously wait for a minimum number of worker processes to complete their tasks.

        Args:
            min_workers (int): The minimum number of worker processes to wait for. Default is 1.
            timeout (float | None): The maximum time to wait for the workers to complete their tasks. If None, wait indefinitely. Default is None.

        Returns:
            Tuple[List[Tuple[int, int]], bool]: A tuple containing a list of tuples with the indices of the worker processes that have completed their tasks and the results of those tasks, and a boolean indicating whether the wait was successful (True) or timed out (False).
        '''
        return await asyncio.to_thread(self.wait_done, min_workers=min_workers, timeout=timeout)

class ManyToOne():
    '''
    A synchronization primitive that allows many master processes to communicate with one worker process.

    Args:
        n (int): The number of master processes.
    '''
    def __init__(self, n: int):
        self.n = n
        self.do_lock = mp.Lock()
        self.do_queue = mp.Queue(maxsize=n)
        self.do_flag = mp.RawArray('i', n)
        self.done_flag = mp.RawArray('i', n)
        self.done_cond = mp.Condition(mp.Lock())

    def do(self, master: int, value: int = 1):
        '''
        Called by the master process to notify the worker process to perform a task.

        Args:
            master (int): The index of the master process.
            value (int): The task index for the worker process to perform. Default is 1.
        '''
        with self.do_lock:
            if self.do_flag[master] + self.done_flag[master]:
                raise RuntimeError(f'Master {master} has an uncompleted task.')
            self.do_queue.put((master, value), block=False)
            self.do_flag[master] = 1
            with self.done_cond:
                self.done_flag[master] = 0

    def _do(self, block: bool = True) -> Tuple[int, int]:
        worker_id, task = self.do_queue.get(block=block)
        return worker_id, task

    def nowait_do(self) -> Tuple[int, int] | None:
        '''
        Called by the worker process to check for a task notification from any of the master processes without blocking.

        Returns:
            Tuple[int, int] | None: Tuple of master index and task index if a task is available, otherwise None.
        '''
        try:
            return self._do(block=False)
        except queue.Empty:
            return None

    def wait_do(self) -> Tuple[int, int]:
        '''
        Called by the worker process to wait for a task notification from any of the master processes.

        Returns:
            Tuple[int, int]: Tuple of master index and task index.
        '''
        return self._do(block=True)

    async def async_wait_do(self) -> Tuple[int, int]:
        '''
        Called by the worker process to asynchronously wait for a task notification from any of the master processes.

        Returns:
            Tuple[int, int]: Tuple of master index and task index.
        '''
        return await asyncio.to_thread(self._do, block=True)

    def done(self, master: int, result: int = 1):
        '''
        Called by the worker process to notify the master process that it has completed its task.

        Args:
            master (int): The index of the master process.
        '''
        if not result:
            return
        with self.do_lock:
            with self.done_cond:
                self.do_flag[master] = 0
                self.done_flag[master] = result
                self.done_cond.notify_all()

    def nowait_done(self, master: int) -> int:
        '''
        Called by the master process to check if the worker process has completed its task without blocking.

        Args:
            master (int): The index of the master process.

        Returns:
            int: The result of the completed task, or 0 if the task is not yet completed.
        '''
        with self.done_cond:
            if self.done_flag[master]:
                ret = self.done_flag[master]
                self.done_flag[master] = 0
                return ret
            return 0

    def wait_done(self, master: int) -> int:
        '''
        Called by the master process to wait for the worker process to complete its task.

        Args:
            master (int): The index of the master process.
        '''
        with self.done_cond:
            self.done_cond.wait_for(lambda: self.done_flag[master])
            ret = self.done_flag[master]
            self.done_flag[master] = 0
            return ret

    async def async_wait_done(self, master: int) -> int:
        '''
        Called by the master process to asynchronously wait for the worker process to complete its task.

        Args:
            master (int): The index of the master process.
        '''
        return await asyncio.to_thread(self.wait_done, master)
