'''
run_nb.py - Script to execute multiple instances of a Jupyter notebook in parallel with controlled frequency and resource management.
'''
from __future__ import annotations

import argparse
import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import re
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, List

import nbformat
from nbclient import NotebookClient

# Ensure this script is not imported as a module
if __name__ != '__main__':
    raise ImportError('Do not import this script; run it as a program instead.')

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run multiple instances of a Jupyter notebook in parallel.')
    parser.add_argument('-f', '--notebook', type=str, help='Path to the notebook template.', required=True)
    parser.add_argument('-t', '--tasks', type=int, help='Total number of notebook instances to run.', default=1)
    parser.add_argument('-w', '--workers', type=int, help='Maximum number of parallel workers. Defaults to 1.', default=1)
    parser.add_argument('--cwd', type=str, default=None, help='Working directory for notebook execution. Defaults to the notebook template directory.')
    parser.add_argument('-i', '--interval', type=float, default=20.0, help='Minimum interval (in seconds) between starting new notebook instances.')
    parser.add_argument('--timeout', type=int, default=None, help='Execution timeout for each notebook instance (in seconds).')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level.')
    parser.add_argument('--raise-on-error', dest='raise_on_error', action='store_true', help='Raise exceptions on notebook instance execution errors. Does not affect the main loop. (default)')
    parser.add_argument('--ignore-errors', dest='raise_on_error', action='store_false', help='Ignore notebook execution errors and continue execution of that instance. Does not affect the main loop.')
    parser.set_defaults(raise_on_error=True)
    return parser

@dataclasses.dataclass
class Config():
    notebook_template_name: str
    execution_cwd: str | None = None

    task_count: int = 1
    num_workers: int = 1
    task_interval: float = 20.0

    execution_timeout: int | None = None
    raise_on_error: bool = True

    log_level: int | str = logging.INFO

    @classmethod
    def from_cli(cls, args: List[str] | None = None) -> Config:
        parser = get_parser()
        parsed_args = parser.parse_args(args)
        return cls(
            notebook_template_name=parsed_args.notebook,
            execution_cwd=parsed_args.cwd,
            task_count=parsed_args.tasks,
            num_workers=parsed_args.workers,
            task_interval=parsed_args.interval,
            execution_timeout=parsed_args.timeout,
            raise_on_error=parsed_args.raise_on_error,
            log_level=parsed_args.log_level,
        )

    def __post_init__(self):
        if not self.task_count >= 1:
            raise ValueError('task_count must be at least 1')
        if not self.num_workers >= 1:
            raise ValueError('num_workers must be at least 1')
        if not self.task_interval >= 1:
            raise ValueError('task_interval must be at least 1')
        if not self.notebook_template_path.exists():
            raise ValueError(f'Notebook template not found: {self.notebook_template_path}')
        if not self.cwd.exists() or not self.cwd.is_dir():
            raise ValueError(f'Working directory is invalid: {self.cwd}')

        self.num_workers = min(self.num_workers, self.task_count)
        self.num_workers = min(self.num_workers, mp.cpu_count())

        if isinstance(self.log_level, str):
            self.log_level = getattr(logging, self.log_level.upper(), logging.INFO)

        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        logging.info(f'Config initialized:\n{self}')
        logging.info(f'Notebook: {self.notebook_template_path}')
        logging.info(f'Python executable: {sys.executable}')
        logging.info(f'Working directory: {self.cwd}')
        logging.info(f'Task count: {self.task_count}')
        logging.info(f'Num workers: {self.num_workers}')
        logging.info(f'Task interval: {self.task_interval}s')
        if self.execution_timeout is not None:
            logging.info(f'Execution timeout: {self.execution_timeout}s')
        else:
            logging.info('Execution timeout: None (no timeout)')
        logging.info(f'Raise on error: {self.raise_on_error}')

    @property
    def script_dir(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def notebook_template_path(self) -> Path:
        path = Path(self.notebook_template_name)
        if path.is_absolute():
            return path.resolve()
        else:
            return (self.script_dir / path).resolve()

    @property
    def cwd(self) -> Path:
        if self.execution_cwd is not None:
            path = Path(self.execution_cwd)
            if path.is_absolute():
                return path.resolve()
            else:
                return (self.script_dir / path).resolve()
        else:
            return self.notebook_template_path.parent.resolve()


@dataclasses.dataclass
class AsyncNotebookExecutor():
    index: int
    notebook_path: Path
    config: Config
    env: Dict[str, str] = dataclasses.field(default_factory=dict)

    task: asyncio.Task[None] | None = dataclasses.field(init=False, default=None)

    def inject_env(self, nb: nbformat.NotebookNode):
        injected_cell = nbformat.v4.new_code_cell(
            source=('\n'.join([
                '# This cell is automatically injected to set up environment variables.',
                'import os',
                *[f"os.environ[{k!r}] = {v!r}" for k, v in self.env.items()]
            ])),
            metadata={
                'tags': ['run-nb-injected-env'],
            },
        )
        nb.cells.insert(0, injected_cell)

    async def execute_notebook(self):
        nb_path = self.notebook_path
        with nb_path.open('r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        self.inject_env(nb)

        def on_notebook_start(notebook):
            logging.info(f'Worker {self.index}: kernel started for {nb_path.name}')
        def on_notebook_complete(notebook):
            logging.info(f'Worker {self.index}: kernel finished for {nb_path.name}')
        def on_cell_execute(cell, cell_index):
            logging.info(f'Worker {self.index}: executing cell {cell_index} in {nb_path.name}')
        def on_cell_complete(cell, cell_index):
            logging.info(f'Worker {self.index}: completed cell {cell_index} in {nb_path.name}')
        def on_cell_error(cell, cell_index, execute_reply):
            logging.warning(f'Worker {self.index}: error in cell {cell_index} of {nb_path.name}.')
            logging.warning(f'Worker {self.index}: error details: {execute_reply}')


        client = NotebookClient(
            nb, km=None,
            timeout=self.config.execution_timeout,
            resources={'metadata': {'path': str(self.config.cwd)}},
            allow_errors=not self.config.raise_on_error,
            on_notebook_start=on_notebook_start,
            on_notebook_complete=on_notebook_complete,
            on_cell_execute=on_cell_execute,
            on_cell_complete=on_cell_complete,
            on_cell_error=on_cell_error,
        )

        try:
            await client.async_execute()
        except Exception as e:
            logging.error(f'Worker {self.index}: execution failed for {nb_path.name}: {e}')
            logging.error(traceback.format_exc())
            raise
        finally:
            with nb_path.open('w', encoding='utf-8') as f:
                nbformat.write(nb, f)

    def start(self) -> asyncio.Task[None]:
        if self.task is not None:
            raise RuntimeError('Task is already started')
        self.task = asyncio.create_task(self._start())
        return self.task

    async def _start(self):
        logging.info(f'Worker {self.index}: created {self.notebook_path.name}')
        await self.execute_notebook()
        logging.info(f'Worker {self.index}: finished {self.notebook_path.name}')

    def is_alive(self) -> bool:
        return self.task is not None and not self.task.done()

    def cancel(self):
        if self.task is not None:
            self.task.cancel()

class MainExecutor():
    def __init__(self, config: Config):
        self._last_start_time = 0.0
        self.config = config
        self.clients: Dict[int, AsyncNotebookExecutor] = {}
        self.notebook_semaphore = asyncio.Semaphore(1)

    async def next_notebook(self, sem: asyncio.Semaphore) -> Path:
        notebook_template_path = self.config.notebook_template_path
        async with sem:
            parent = notebook_template_path.parent
            stem = notebook_template_path.stem
            suffix = notebook_template_path.suffix
            pattern = re.compile(
                rf'^{re.escape(stem)}_(\d+){re.escape(suffix)}$')
            max_index = 0
            for item in parent.iterdir():
                if not item.is_file():
                    continue
                match = pattern.match(item.name)
                if match:
                    max_index = max(max_index, int(match.group(1)))
            target = parent / f'{stem}_{max_index + 1}{suffix}'
            shutil.copy2(notebook_template_path, target)
            return target

    @property
    def alive_clients(self) -> List[AsyncNotebookExecutor]:
        return [v for v in self.clients.values() if v.is_alive()]

    @property
    def unfinished(self):
        return any([
            len(self.clients) < self.config.task_count,
            len(self.alive_clients) > 0
        ])

    @property
    def has_free_worker(self):
        return all([
            len(self.alive_clients) < self.config.num_workers,
            len(self.clients) < self.config.task_count
        ])

    @property
    def ready(self):
        elapsed = time.monotonic() - self._last_start_time
        return elapsed >= self.config.task_interval

    def start_next(self, index: int, notebook_path: Path):
        client = AsyncNotebookExecutor(
            index=index,
            notebook_path=notebook_path,
            config=self.config,
            env={
                'automated': '1',
                'uuid': str(uuid.uuid4()),
            }
        )
        self.clients[index] = client
        client.start()
        self._last_start_time = time.monotonic()

    async def shutdown_all(self):
        for _ in self.clients.values():
            _.cancel()

        tasks = [c.task for c in self.clients.values() if c.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self):
        index = 0
        while self.unfinished:
            if self.has_free_worker and self.ready:
                new_instance = await self.next_notebook(self.notebook_semaphore)
                self.start_next(index, new_instance)
                index += 1
                continue

            # Wait for any task to complete
            unfinished_tasks = [
                _.task for _ in self.alive_clients if _.task is not None
            ]
            # Frequency control: at least wait 1s to avoid busy loop
            if len(unfinished_tasks) < self.config.num_workers:
                sleep = asyncio.create_task(asyncio.sleep(max(1,
                    self.config.task_interval -
                    (time.monotonic() - self._last_start_time)
                )))
                unfinished_tasks.append(sleep)
            else:
                sleep = None

            if unfinished_tasks:
                done, _ = await asyncio.wait(
                    unfinished_tasks, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                done = set()
            if sleep is not None and not sleep.done():
                sleep.cancel()
                try:
                    await sleep
                except asyncio.CancelledError:
                    pass
            for d in done:
                if d is sleep:
                    continue
                try:
                    d.result()
                except asyncio.CancelledError:
                    logging.warning('Worker task was cancelled.')
                except Exception:
                    logging.exception('Worker task failed.')

    async def run(self):
        try:
            await self._run()
        except (KeyboardInterrupt, asyncio.CancelledError):
            logging.warning('Interrupted, shutting down all workers...')
            await self.shutdown_all()
            raise
        except Exception as e:
            logging.error(f'Main executor failed: {e}')
            logging.error(traceback.format_exc())
            await self.shutdown_all()
            raise

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        raise RuntimeError(
            'This script is not supported on Windows due to limitations in process management.'
        )
    config = Config.from_cli()
    tempfile_path = config.script_dir / f'.{config.notebook_template_path.stem}_run_nb.lock'
    if tempfile_path.exists():
        raise RuntimeError(f'A same instance is already running.')
    else:
        tempfile_path.touch()
    atexit.register(lambda: tempfile_path.unlink(missing_ok=True))

    executor = MainExecutor(config)
    asyncio.run(executor.run())
