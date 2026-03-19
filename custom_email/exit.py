import atexit
import dataclasses
import datetime
import functools
import linecache
import os
import smtplib
import socket
import sys
import threading
import traceback
from email.message import EmailMessage
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, ClassVar, Iterable, List, Sequence, Tuple, Type

import __main__
from jinja2 import Environment, FileSystemLoader

try:
    # IPython test
    from IPython.core.getipython import get_ipython
    ip = get_ipython()
    IPYTHON = ip is not None
except ImportError:
    IPYTHON = False


def attach_exception_handler(
    func: Callable[[
        Type[BaseException], BaseException,
        TracebackType | None, threading.Thread | None
    ], Any]
) -> Callable[[], None]:
    if IPYTHON:
        # In IPython, sys.excepthook is not used. We need to patch IPython's exception event handler
        ip = get_ipython()
        assert ip is not None

        def ipython_wrapper(result):
            exc = result.error_in_exec or result.error_before_exec
            if exc is None:
                return

            func(type(exc), exc, exc.__traceback__, None)

        ip.events.register('post_run_cell', ipython_wrapper)

        def detach():
            ip.events.unregister('post_run_cell', ipython_wrapper)
    else:
        old_hook = sys.excepthook

        @functools.wraps(old_hook)
        def sys_wrapper(exc_type, exc_value, exc_traceback):
            try:
                func(exc_type, exc_value, exc_traceback, None)
            finally:
                old_hook(exc_type, exc_value, exc_traceback)

        sys.excepthook = sys_wrapper

        def detach():
            sys.excepthook = old_hook

    return detach

def attach_thread_exception_handler(
    func: Callable[[
        Type[BaseException], BaseException | None,
        TracebackType | None, threading.Thread | None
    ], Any]
) -> Callable[[], None]:
    old_hook = threading.excepthook

    @functools.wraps(old_hook)
    def thread_wrapper(args: threading.ExceptHookArgs):
        try:
            func(args.exc_type, args.exc_value, args.exc_traceback, args.thread)
        finally:
            old_hook(args)

    threading.excepthook = thread_wrapper

    def detach():
        threading.excepthook = old_hook

    return detach

@dataclasses.dataclass(slots=True, init=False)
class GlobalInfo():
    _initialized: ClassVar[bool] = False
    _instance: ClassVar['GlobalInfo | None'] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()

    start_time: datetime.datetime
    time_info: 'TimeInfo | None'
    process_info: 'ProcessInfo'
    exception_info: List['ExceptionInfo']
    thread_exception_info: List['ExceptionInfo']

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = object.__new__(cls)
            return cls._instance

    def __init__(self, *args, **kwargs):
        with type(self)._lock:
            if type(self)._initialized:
                return

            self.start_time = datetime.datetime.now().astimezone()
            self.time_info = None
            self.process_info = ProcessInfo.make()
            self.exception_info = []
            self.thread_exception_info: List['ExceptionInfo'] = []

            attach_exception_handler(ExceptionInfo.make)
            attach_thread_exception_handler(ExceptionInfo.make)
            type(self)._initialized = True

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()

    def exit(self):
        with self:
            self.time_info = TimeInfo.make()
            return dataclasses.asdict(self)

def _get_tb_context(tb: TracebackType) -> Tuple[List[str], int]:
    while tb.tb_next is not None:
        tb = tb.tb_next

    context_lines = 5

    frame = tb.tb_frame
    filename = frame.f_code.co_filename
    lineno = tb.tb_lineno or frame.f_lineno or 1

    lines = linecache.getlines(filename, frame.f_globals)

    if not lines:
        summary = traceback.extract_tb(tb, limit=1)[-1]
        if summary.line:
            return [summary.line], 0
        return [], 0

    start = max(1, lineno - context_lines)
    end = min(len(lines), lineno + context_lines)

    snippet = [line.rstrip('\n') for line in lines[start - 1:end]]
    error_line_index = lineno - start
    return snippet, error_line_index

@dataclasses.dataclass(slots=True)
class ProcessInfo():
    entry_filename: str
    entry_path: str
    cwd: str
    pid: int
    ppid: int | None
    hostname: str
    executable: str = dataclasses.field(init=False, default=sys.executable)
    version: str = dataclasses.field(init=False, default=sys.version)
    argv: List[str] = dataclasses.field(
        init=False, default_factory=lambda: list(sys.argv)
    )

    @classmethod
    def make(cls):
        raw = getattr(__main__, '__file__', None) or sys.argv[0] or ''
        if not raw:
            entry_filename = '<interactive>'
            entry_path = '<interactive>'
        else:
            try:
                entry_path_obj =  Path(raw).expanduser().resolve()
            except OSError:
                entry_path_obj = Path(raw)
            finally:
                entry_filename = entry_path_obj.name
                entry_path = str(entry_path_obj)

        try:
            ppid = os.getppid()
        except Exception:
            ppid = None

        return ProcessInfo(
            entry_filename=entry_filename,
            entry_path=entry_path,
            cwd=os.getcwd(),
            pid=os.getpid(),
            ppid=ppid,
            hostname=socket.gethostname(),
        )

@dataclasses.dataclass(slots=True)
class TimeInfo():
    wall_time: datetime.timedelta
    current_time: datetime.datetime

    @classmethod
    def make(cls):
        now = datetime.datetime.now().astimezone()
        elapsed = now - GlobalInfo().start_time
        return TimeInfo(wall_time=elapsed, current_time=now)


@dataclasses.dataclass(slots=True)
class ExceptionInfo():
    exc_type: str
    exc_message: str
    traceback_text: str

    filename: str
    lineno: int | None
    func_name: str | None

    code_context: List[str]
    error_line_index: int
    thread_name: str | None = None
    thread_id: int | None = None

    exc_time: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now().astimezone()
    )
    wall_time: datetime.timedelta = dataclasses.field(
        default_factory=lambda: datetime.datetime.now().astimezone() - GlobalInfo().start_time
    )

    # Match with sys.excepthook arguments for easy construction
    @classmethod
    def make(
        cls, exc_type, exc_value, exc_traceback,
        thread: threading.Thread | None = None
    ):
        if exc_traceback is not None:
            frames = traceback.extract_tb(exc_traceback)
            last_frame = frames[-1]
            filename = last_frame.filename
            lineno = last_frame.lineno
            func_name = last_frame.name
            code_context, error_line_index = _get_tb_context(
                exc_traceback
            )
        else:
            filename = '<unknown>'
            lineno = 0
            func_name = None
            code_context = []
            error_line_index = 0

        ret = ExceptionInfo(
            exc_type=exc_type.__name__ if exc_type else '<unknown>',
            exc_message=str(exc_value) if exc_value else '',
            traceback_text=''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback)),
            filename=filename,
            lineno=lineno,
            func_name=func_name,
            code_context=code_context,
            error_line_index=error_line_index,
            thread_name=thread.name if thread else None,
            thread_id=thread.ident if thread else None
        )
        with GlobalInfo() as global_info:
            if thread is not None:
                global_info.thread_exception_info.append(ret)
            else:
                global_info.exception_info.append(ret)
        return ret

def register(
    mailhost: str,
    fromaddr: str,
    toaddrs: Iterable[str],
    credentials: Tuple[str, str],
    port: int = 465,
    timeout: float | None = None,
    *,
    subject: str = 'Program Exit Notification',
    report_at_success: bool = False,
    html_template: str = 'exit.html.j2',
    template_search_paths: Sequence[str | Path] | str | Path | None = None,
):
    GlobalInfo() # Initialize the singleton instance
    if template_search_paths is None:
        template_search_paths = Path(__file__).parent / 'templates'
    loader = FileSystemLoader(template_search_paths)
    jinja_env = Environment(loader=loader, autoescape=True)
    template = jinja_env.get_template(html_template)

    def exit_handler():
        if not report_at_success and \
            not GlobalInfo().exception_info and \
            not GlobalInfo().thread_exception_info:
            return

        with GlobalInfo() as global_info:
            html = template.render(**global_info.exit())

        _kwargs = {}
        _kwargs['host'] = mailhost
        _kwargs['port'] = port
        if timeout is not None:
            _kwargs['timeout'] = timeout
        server = smtplib.SMTP_SSL(**_kwargs)
        server.login(*credentials)

        msg = EmailMessage()
        msg['From'] = fromaddr
        msg['To'] = ', '.join(toaddrs)
        msg['Subject'] = subject
        msg.set_content('Please view it in an HTML-capable client.')
        msg.add_alternative(html, subtype='html')
        server.send_message(msg)
        server.quit()

    atexit.register(exit_handler)
