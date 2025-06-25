'''
parser.py

Author: HuangFuSL
Date: 2025-06-25

A utility to automatically parse command line arguments into a dataclass instance.
This module provides a decorator `@auto_cli` that can be applied to a dataclass.
After applying the decorator, you can call `parse_args()` on the dataclass to
parse command line arguments and return an instance of the dataclass.

1. A special field `--config` is added to allow loading configuration from a
file in JSON, YAML, or TOML format.
2. When a field is specified in both the command line arguments and the
configuration file, the command line argument takes precedence.
'''

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
import sys
import typing
from typing import Any, Dict, Sequence, Type, TypeVar, Union, Optional

T = TypeVar('T')

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import tomllib
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

class Const(enum.Enum):
    MISSING_IN_CLI = -1

def _strip_optional(tp: Any) -> Any:
    '''Union[..., None] ➜ ...'''
    if typing.get_origin(tp) is Union:
        non_none = [t for t in typing.get_args(tp) if t is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    if typing.get_origin(tp) is Optional:
        return _strip_optional(typing.get_args(tp)[0])
    return tp


def _infer_argtype(tp: Any):
    tp = _strip_optional(tp)
    origin = typing.get_origin(tp)
    if tp in (int, float, str, bool):
        return tp
    if origin in (list, tuple, set):
        return str

    return str


def _convert_value(raw: str, tp):
    ''' Convert a string value to the specified type. '''
    tp = _strip_optional(tp)
    origin = typing.get_origin(tp)

    if tp in (int, float, str):
        return tp(raw)
    if tp is bool:
        # Never really reach heres
        return raw.lower() not in {'false', '0', 'no'}
    if origin in (list, tuple, set, dict):
        return json.loads(raw)
    # Any other type, just return as is
    return raw


def auto_cli(cls: Type[T] | None = None, /, **decorator_kw):
    ''' Decorate and inject a `parse_args` method into a dataclass.'''

    def wrap(datacls: Type[T]) -> Type[T]:
        _type_hints: Dict[str, Any] = typing.get_type_hints(
            datacls, globalns=vars(sys.modules[datacls.__module__])
        )

        @classmethod
        def parse_args(cls_, argv: Sequence[str] | None = None) -> T:
            '''
            Parse command line arguments into a dataclass instance.

            - Basic types: int/float/str
            - bool: --flag / --no-flag
            - Optional[T]
            - list/tuple/set/dict: Parse as JSON strings
            - Configuration file: --config <path>
            '''
            parser = argparse.ArgumentParser(
                description=decorator_kw.get('description', cls_.__name__)
            )

            for f in dataclasses.fields(cls_):
                name = f'--{f.name}'
                argtype = _infer_argtype(_type_hints[f.name])
                default = f.default if f.default is not dataclasses.MISSING else None
                help_text = f.metadata.get('help', '')

                if argtype is bool:
                    # bool ➜ --flag / --no-flag
                    parser.add_argument(
                        f'--no-{f.name}', dest=f.name, action='store_false',
                        help=help_text + ' (set to False)'
                    )
                    parser.add_argument(
                        name, dest=f.name, action='store_true',
                        help=help_text + ' (set to True)'
                    )
                else:
                    parser.add_argument(
                        name,
                        type=argtype if argtype is not str else str,
                        default=Const.MISSING_IN_CLI,
                        help=help_text +
                        (f' (default: {default})' if default is not None else '')
                    )
            parser.add_argument(
                '--config',
                type=str, default='',
                help='Path to the configuration file'
            )

            ns = parser.parse_args(argv)
            kw = {}
            # Handle config file
            if ns.config:
                with open(ns.config, 'r') as f:
                    content = f.read()
                suffix = os.path.splitext(ns.config)[1].lower()
                if suffix in ('.json',):
                    config_data = json.loads(content)
                elif suffix in ('.yaml', '.yml'):
                    if not YAML_AVAILABLE:
                        raise ImportError('YAML support is not available. Install PyYAML to use YAML files.')
                    config_data = yaml.load(content, Loader=yaml.SafeLoader)
                elif suffix in ('.toml',):
                    if not TOML_AVAILABLE:
                        raise ImportError('TOML support is not available. Install tomllib to use TOML files.')
                    config_data = tomllib.loads(content)
                else:
                    raise ValueError('Unsupported config file format. Use .json, .yaml, or .toml.')
                if not isinstance(config_data, dict):
                    raise ValueError('Config file must contain a dictionary.')
                # Override all all command line arguments with config file values
                kw = config_data
            for f in dataclasses.fields(cls_):
                val = getattr(ns, f.name)
                if val is Const.MISSING_IN_CLI:
                    if f.name in kw:
                        continue
                    if f.default is dataclasses.MISSING:
                        raise ValueError(f'Missing required argument: {f.name}')
                    val = f.default
                else:
                    type_ = _type_hints[f.name]
                    # JSON deserialization for complex types
                    if isinstance(val, str) \
                        and _infer_argtype(type_) is str \
                        and typing.get_origin(_strip_optional(type_)) in (list, tuple, set, dict):
                        val = _convert_value(val, type_)
                kw[f.name] = val
            return cls_(**kw)

        datacls.parse_args = parse_args
        return datacls

    return wrap if cls is None else wrap(cls)

__all__ = ['auto_cli']
