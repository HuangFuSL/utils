'''
`utils.email.elogging` - Email-aware logging handlers and formatters.

This module provides a logging `Handler` that delivers selected log records
via email, using an external HTML template rendered by Jinja2.
'''

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import logging
import queue
import smtplib
import sys
import threading
import time
import traceback
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader


class EmailLogMode(enum.Enum):
    IMMEDIATE = 'immediate'
    DIGEST = 'digest'


class EmailThread(threading.Thread):
    def __init__(
        self,
        incoming_queue: queue.Queue[EmailMessage | None],
        mailhost: str,
        credentials: Tuple[str, str],
        port: int = 465,
        timeout: float | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._kwargs = {}
        self._kwargs['host'] = mailhost
        self._kwargs['port'] = port
        if timeout is not None:
            self._kwargs['timeout'] = timeout

        self.credentials = tuple(credentials)  # type: ignore[assignment]
        self.port = port
        self.timeout = timeout
        self._server: smtplib.SMTP_SSL | None = None
        self._queue = incoming_queue

    def _ensure_connection(self) -> smtplib.SMTP_SSL:
        if self._server is not None:
            try:
                self._server.noop()
                return self._server
            except Exception:
                try:
                    self._server.close()
                except Exception:
                    pass
                finally:
                    self._server = None

        server = smtplib.SMTP_SSL(**self._kwargs)
        server.login(*self.credentials)
        self._server = server
        return server

    def _close_connection(self) -> None:
        if self._server is None:
            return
        try:
            self._server.quit()
        except Exception:
            try:
                self._server.close()
            except Exception:
                pass
        finally:
            self._server = None

    def run(self) -> None:
        try:
            while True:
                msg = self._queue.get()
                try:
                    if msg is None:
                        break
                    server = self._ensure_connection()
                    server.send_message(msg)
                except Exception:
                    self._close_connection()
                    print('Failed to send email:', file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                finally:
                    self._queue.task_done()
        finally:
            self._close_connection()

@dataclasses.dataclass(frozen=True)
class BufferedLog():
    created_timestamp: float
    created: str
    levelno: int
    levelname: str
    logger_name: str
    pathname: str
    lineno: int
    message: str
    exc: str | None = None

    @property
    def key(self) -> Tuple[int, str]:
        return (self.levelno, f'{self.pathname}:{self.lineno}')

    def __gt__(self, other: 'BufferedLog | None'):
        if other is None:
            return True
        if self.levelno != other.levelno:
            return self.levelno > other.levelno
        return self.created_timestamp > other.created_timestamp

    @classmethod
    def from_record(cls, record: logging.LogRecord, formatted_msg: str):
        exc = None
        if record.exc_info:
            if isinstance(record.exc_info, BaseException):
                exc_value = record.exc_info
                exc_type = type(exc_value)
                exc_tb = exc_value.__traceback__
            elif isinstance(record.exc_info, tuple):
                exc_type, exc_value, exc_tb = record.exc_info
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

            if exc_type not in (None, type(None)):
                exc = ''.join(
                    traceback.format_exception(exc_type, exc_value, exc_tb)
                )

        return BufferedLog(
            created_timestamp=record.created,
            created=time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(record.created)
            ),
            levelno=record.levelno,
            levelname=record.levelname,
            logger_name=record.name,
            pathname=record.pathname,
            lineno=record.lineno,
            message=formatted_msg,
            exc=exc,
        )

    def to_dict(self):
        return dataclasses.asdict(self)


class TemplateEmailHandler(logging.Handler):
    '''
    Logging handler that sends log records via email using Jinja2 templates.

    This handler is intended for scenarios where:

    - Only *some* levels should generate email (for example, WARNING+),
    - High-severity logs (for example, ERROR+) should be sent immediately,
    - Lower-severity logs can be batched and periodically summarized.

    The handler does **not** try to be a generic replacement for all logging;
    it is specifically tuned for email delivery and rate control.

    Args:
        mailhost (str): SMTP server hostname.
        fromaddr (str): Sender email address.
        toaddrs (Iterable[str]): Iterable of recipient email addresses.
        credentials (Tuple[str, str]): Tuple of ``(username, password)`` used for SMTP authentication.
        port (int, optional): SMTP server port. Defaults to 465 for SSL.
        subject (str): Base subject line for emails.
        timeout (float | None): Timeout for SMTP operations in seconds. If None, no timeout is set.
        mail_level (int): Records with levelno >= ``mail_level`` are eligible for email. These records are cached and collapsed.
        immediate_level (int): Records with levelno >= ``immediate_level`` are sent immediately.
        html_template (str): Name of the Jinja2 template used for the HTML body.
        template_search_paths (Sequence[Pathlike] | Pathlike | None): List of filesystem paths to search for the HTML template.
    '''

    def __init__(
        self,
        mailhost: str,
        fromaddr: str,
        toaddrs: Iterable[str],
        credentials: Tuple[str, str],
        port: int = 465,
        timeout: float | None = None,
        *,
        subject: str = 'Log Message',
        mail_level: int = logging.WARNING,
        immediate_level: int = logging.ERROR,
        html_template: str = 'logging.html.j2',
        template_search_paths: Sequence[str | Path] | str | Path | None = None,
    ) -> None:
        super().__init__()
        self.queue = queue.Queue()
        self.fromaddr = fromaddr
        self.toaddrs = list(toaddrs)
        self.email_thread = EmailThread(
            incoming_queue=self.queue, mailhost=mailhost, credentials=credentials, port=port, timeout=timeout
        )

        self.subject = subject
        if mail_level > immediate_level:
            immediate_level = mail_level

        self.mail_level = mail_level
        self.immediate_level = immediate_level

        if template_search_paths is None:
            template_search_paths = Path(__file__).parent / 'templates'
        loader = FileSystemLoader(template_search_paths)
        self.jinja_env = Environment(loader=loader, autoescape=True)
        self.html_template_name = html_template

        self.main_record: BufferedLog | None = None
        self.summary_counter: Dict[Tuple[int, str], int] = collections.Counter()
        self.summary_latest: Dict[Tuple[int, str], BufferedLog] = {}

        self.email_thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self.email_thread.is_alive():
                return
            if record.levelno < self.mail_level:
                return
            new_record = copy.copy(record)
            new_record.exc_info = None
            new_record.exc_text = None
            message = self.format(new_record)

            new_log = BufferedLog.from_record(record, message)
            if new_log > self.main_record:
                self.main_record = new_log
            self.summary_counter[new_log.key] += 1
            self.summary_latest[new_log.key] = new_log

            if record.exc_info or record.levelno >= self.immediate_level:
                self._send(EmailLogMode.IMMEDIATE)
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        return

    def close(self) -> None:
        self.acquire()
        try:
            try:
                self._send(EmailLogMode.DIGEST)
            finally:
                self.queue.put(None)
                self.email_thread.join()
                super().close()
        finally:
            self.release()

    def _build_html(
        self, mode: EmailLogMode, main_record: BufferedLog,
        summary_counter: Dict[Tuple[int, str], int],
        summary_latest: Dict[Tuple[int, str], BufferedLog]
    ) -> str:
        template = self.jinja_env.get_template(self.html_template_name)
        main_record_dict = main_record.to_dict() | {
            'count': summary_counter.get(main_record.key, 0)
        }

        remaining_records = sorted([
            {'count': summary_counter.get(k, 0), **v.to_dict()}
            for k, v in summary_latest.items() if k != main_record.key
        ], key=lambda r: (r['levelno'], r['created_timestamp']), reverse=True)

        return template.render(**{
            'mode': mode.value, 'subject': self.subject,
            'total_raw_records': sum(summary_counter.values()),
            'main_record': main_record_dict,
            'remaining_records': remaining_records,
        })

    def _send(self, mode: EmailLogMode) -> None:
        # Build html in lock to ensure consistency
        active_main: BufferedLog | None = None
        active_count: Dict[Tuple[int, str], int] = {}
        active_latest: Dict[Tuple[int, str], BufferedLog] = {}

        try:
            if self.main_record is None:
                return

            active_main = self.main_record
            active_count = self.summary_counter.copy()
            active_latest = self.summary_latest.copy()
            self.main_record = None
            self.summary_counter.clear()
            self.summary_latest.clear()
            if mode == EmailLogMode.IMMEDIATE:
                sub = f'{self.subject} [{active_main.levelname}] at {active_main.logger_name}'
            else:
                sub = f'{self.subject} (digest, {sum(active_count.values())} records)'
            html = self._build_html(
                mode=mode, main_record=active_main,
                summary_counter=active_count, summary_latest=active_latest
            )

            msg = EmailMessage()
            msg['Subject'] = sub
            msg['From'] = self.fromaddr
            msg['To'] = ', '.join(self.toaddrs)

            msg.set_content('Please view it in an HTML-capable client.')
            msg.add_alternative(html, subtype='html')

            self.queue.put(msg)
        except Exception:
            if active_main is not None and active_main > self.main_record:
                self.main_record = active_main
            for k, v in active_count.items():
                self.summary_counter[k] += v
            for k, v in active_latest.items():
                if v > self.summary_latest.get(k):
                    self.summary_latest[k] = v
            raise
