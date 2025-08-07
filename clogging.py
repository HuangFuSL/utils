'''
`utils.clogging` - Colored Logging Formatter

A Python module that provides a colored logging formatter for the `logging` module.
This module defines a base colored formatter class and a default implementation
that applies specific colors to different logging levels and fields.
It allows for easy customization of log message styles, making it easier to
distinguish between different log levels in console output.
'''

import logging
import copy
from typing import Any, Dict, List, Tuple

from .cprint import cprefix, RESET


class BaseColoredFormatter(logging.Formatter):
    '''
    Add color to log messages based on their level.

    This formatter allows customization of log message styles based on their
    logging level. It supports coloring the level name and optionally the
    message itself. The styles can be customized through the `get_color` method,
    which should return a argument dictionary compatible with `cprint.cprint`.

    Args:
        fmt (str): The format string for the log messages.
        datefmt (str | None): The format string for the date in log messages.
        style (str): The character used in the format string (default is '%').
    (same as logging.Formatter)
    '''

    def __init__(
        self,
        fmt: str = '%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt: str | None = '%Y-%m-%d %H:%M:%S',
        style: str = '%'
    ) -> None:
        level_colors = {
            level: self.get_color('base', level)
            for level in (
                logging.DEBUG, logging.INFO, logging.WARNING,
                logging.ERROR, logging.CRITICAL
            )
        }
        self._reset_code = {
            k: RESET + (cprefix(**v) if v is not None else '')
            for k, v in level_colors.items()
        }
        self._color_cache: Dict[Tuple[str, int], str] = {}
        super().__init__(fmt=fmt, datefmt=datefmt, style=style) # type: ignore

    @property
    def field_names(self) -> List[str]:
        '''
        Returns the list of field names that can be colored.
        This can be overridden in subclasses to specify which fields are
        available.

        `'base'` is a special field that can be used to apply a default color
        to the template.

        Returns:
            List[str]: A list of field names that can be colored.
        '''
        return [
            'levelname', 'name', 'asctime', 'filename', 'lineno',
            'funcName', 'base', 'message'
        ]

    def get_color(self, field_name: str, level: int) -> Dict[str, Any] | None:
        '''
        Returns the color format for a given field name and logging level. This
        method should be overridden in subclasses to provide specific color
        styles for different fields and levels.

        Args:
            field_name (str): The name of the field to color.
            level (int): The logging level for which to get the color.

        Returns:
            Dict[str, Any] | None: A dictionary of color attributes compatible with `utils.cprint.cprint`.
        '''
        return None

    def _color_code(self, field_name: str, level: int) -> str:
        '''
        Returns the color code for a given field name and logging level.
        Caches the result to avoid recomputing it for the same field and level.
        '''
        key = (field_name, level)
        if key not in self._color_cache:
            color = self.get_color(field_name, level)
            self._color_cache[key] = cprefix(**color) if color is not None \
                else self._reset_code[level]
        return self._color_cache[key]

    def _format_field(self, record: logging.LogRecord, field_name: str):
        '''
        Formats a specific field of the log record with its color code.
        '''
        if field_name == 'message':
            # Special handling for the message field to ensure it is colored correctly
            record.msg = f'{self._color_code(field_name, record.levelno)}{record.msg}{self._reset_code[record.levelno]}'
        elif hasattr(record, field_name):
            value = getattr(record, field_name)
            color_code = self._color_code(field_name, record.levelno)
            record.__setattr__(field_name, f'{color_code}{value}{self._reset_code[record.levelno]}')

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        '''
        Formats the time of the log record.
        If datefmt is provided, it formats the time accordingly.
        Otherwise, it uses the default format.

        Args:
            record (logging.LogRecord): The log record to format.
            datefmt (str | None): The format string for the date.

        Returns:
            str: The formatted time string with color applied.
        '''
        return f'{self._color_code("asctime", record.levelno)}{super().formatTime(record, datefmt)}{self._reset_code[record.levelno]}'

    def format(self, record: logging.LogRecord) -> str:
        '''
        Formats the log record with colors applied to the specified fields.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with colors applied.
        '''
        new_record = copy.deepcopy(record)

        # Format the level name with color
        for field_name in self.field_names:
            self._format_field(new_record, field_name)

        formatted = super().format(new_record)

        return f'{self._reset_code[new_record.levelno]}{formatted}{RESET}'

class DefaultColoredFormatter(BaseColoredFormatter):
    '''
    Default colored formatter with predefined styles for log levels.
    '''

    def get_color(self, field_name: str, level: int) -> Dict[str, Any] | None:
        match (field_name, level):
            case (_, logging.CRITICAL):
                return {'fg': 'bright_white', 'bg': 'red', 'bf': True}
            case ('asctime', _):
                return {'fg': 'green', 'bf': True}
            case ('levelname', logging.DEBUG):
                return {'fg': 'bright_black'}
            case ('levelname', logging.INFO):
                return {'fg': 'green'}
            case ('levelname', logging.WARNING):
                return {'fg': 'yellow'}
            case ('levelname', logging.ERROR):
                return {'fg': 'red', 'bf': True}
            case ('levelname', logging.CRITICAL):
                return {'fg': '#FFFFFF', 'bg': 'red', 'bf': True}
            case ('name', _):
                return {'fg': 'bright_cyan'}
            case ('filename', _) | ('funcName', _) | ('lineno', _):
                return {'fg': 'bright_magenta'}
            case _:
                return None
