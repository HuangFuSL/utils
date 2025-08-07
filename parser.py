'''
`utils.parser`

A utility to automatically parse command line arguments into a dataclass instance.
This module provides a decorator ``@auto_cli`` that can be applied to a dataclass.
After applying the decorator, you can call ``parse_args()`` on the dataclass to
parse command line arguments and return an instance of the dataclass.

#. A special field ``--config`` is added to allow loading configuration from a file in JSON, YAML, or TOML format.
#. When a field is specified in both the command line arguments and the configuration file, the command line argument takes precedence.
'''

from __future__ import annotations

import argparse
import dataclasses
import enum
import functools
import json
import os
import sys
import typing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Sequence, Type, TypeVar, Union, cast, overload, runtime_checkable

T = TypeVar('T', covariant=True)

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


def _build_parser(*parsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    '''
    Gather multiple argument parsers into a single one.
    '''
    parser = argparse.ArgumentParser(parents=parsers)
    parser.add_argument(
        '--config',
        type=str, default='',
        help='Path to the configuration file'
    )
    return parser

def _handle_config_file(ns: argparse.Namespace) -> Dict[str, Any]:
    if ns.config:
        with open(ns.config, 'r') as f:
            content = f.read()
        suffix = os.path.splitext(ns.config)[1].lower()
        if suffix in ('.json',):
            config_data = json.loads(content)
        elif suffix in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                raise ImportError(
                    'YAML support is not available. Install PyYAML to use YAML files.')
            config_data = yaml.load(content, Loader=yaml.SafeLoader)
        elif suffix in ('.toml',):
            if not TOML_AVAILABLE:
                raise ImportError(
                    'TOML support is not available. Install tomllib to use TOML files.')
            config_data = tomllib.loads(content)
        else:
            raise ValueError(
                'Unsupported config file format. Use .json, .yaml, or .toml.')
        if not isinstance(config_data, dict):
            raise ValueError('Config file must contain a dictionary.')
        # Override all all command line arguments with config file values
        kw = config_data
        return kw
    return {}

if TYPE_CHECKING:
    @runtime_checkable
    class _DECORATED(Any, Protocol[T]):
        @classmethod
        def get_parser(cls, prefix: str = '') -> argparse.ArgumentParser: ...

        @classmethod
        def parse_namespace(
            cls, ns: argparse.Namespace, kw: Dict[str, Any] | None = None,
            prefix: str = ''
        ) -> T: ...

        @classmethod
        def parse_args(cls, argv: Sequence[str] | None = None) -> T: ...
else:
    @runtime_checkable
    class _DECORATED(Protocol[T]):
        @classmethod
        def get_parser(cls, prefix: str = '') -> argparse.ArgumentParser: ...

        @classmethod
        def parse_namespace(
            cls, ns: argparse.Namespace, kw: Dict[str, Any] | None = None,
            prefix: str = ''
        ) -> T: ...

        @classmethod
        def parse_args(cls, argv: Sequence[str] | None = None) -> T: ...


@overload
def auto_cli(
    cls: Type[T], /, **decorator_kw: Any
) -> Type[_DECORATED[T]]:
    ...

@overload
def auto_cli(
    cls: None = None, /, **decorator_kw: Any
) -> Callable[[Type[T]], Type[_DECORATED[T]]]:
    ...

def auto_cli(cls: Type[T] | None = None, /, **decorator_kw) -> Type[_DECORATED[T]] | Callable[[Type[T]], Type[_DECORATED[T]]]:
    '''
    Automatically generates an argument parser for a dataclass with type hints.
    Field types, including basic types, and ``Optional`` are automatically
    recognized and converted corresponding argument types. Complex types like
    lists and dictionaries can be input as strings in json format.

    Apart from the fields of the dataclass, ``auto_cli`` also adds a ``--config``
    argument to the parser for loading configuration settings from a json, yaml,
    or toml file. Values from the configuration file have a lower priority than
    command line arguments.

    The decorated function will get the following new class methods:

    - ``get_parser(prefix: str = '') -> argparse.ArgumentParser``:
        Returns an `argparse.ArgumentParser` instance with arguments based on
        the dataclass fields. The `prefix` argument is prepended to each
        argument name. The parser returned does not include the ``--config``
        argument for loading configuration files.

        Example:

        .. code-block:: python

            @auto_cli
            @dataclasses.dataclass
            class YourClass:
                name: str = 'DefaultName'
                age: int = 25

            parser = YourClass.get_parser()
            parser_prefix = YourClass.get_parser(prefix='man')

        The generated `parser` will accept the following arguments:

        * ``--name``: The name argument with a default value of 'DefaultName'.
        * ``--age``: The age argument with a default value of 25.

        The generated ``parser_prefix`` will accept the following arguments:

        * ``--man-name``: The name argument with a default value of 'DefaultName'.
        * ``--man-age``: The age argument with a default value of 25.

    - ``parse_namespace(ns: argparse.Namespace, kw: Dict[str, Any] = None, prefix: str = '') -> 'YourClass'``:
        Parses an ``argparse.Namespace`` instance and a kwargs dictionary into an
        instance of the dataclass. Prefix is prepended to each argument name.
        The ``kw`` dictionary is loaded from a config file if specified in ``ns``.
        A ``ValueError`` is raised if a required argument is missing from both
        ``ns`` and ``kw``, and no default value is provided.
    - ``parse_args(argv: List[str] = None) -> 'YourClass'``:
        Parses command line arguments and returns an instance of the dataclass.
        If ``argv`` is ``None``, it uses ``sys.argv[1:]``. The method also handles
        config files specified in the command line arguments.

    Returns:
        Type[cls]: The decorated class.
    '''

    @functools.wraps(auto_cli)
    def wrap(datacls: Type[T]) -> Type[_DECORATED[T]]:
        _type_hints: Dict[str, Any] = typing.get_type_hints(
            datacls, globalns=vars(sys.modules[datacls.__module__])
        )

        @classmethod
        def get_parser(cls_, prefix: str = '') -> argparse.ArgumentParser:
            '''
            Get the argument parser for this dataclass.
            '''
            parser = argparse.ArgumentParser(add_help=False)

            for f in dataclasses.fields(cls_):
                if prefix:
                    name = f'{prefix}_{f.name}'
                    argname = f'--{prefix}-{f.name}'
                    no_argname = f'--no-{prefix}-{f.name}'
                else:
                    name = f.name
                    argname = f'--{f.name}'
                    no_argname = f'--no-{f.name}'
                argtype = _infer_argtype(_type_hints[f.name])
                if f.default is not dataclasses.MISSING:
                    default = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    default = f.default_factory()
                else:
                    default = None

                help_text = f.metadata.get('help', '')

                if argtype is bool:
                    # bool ➜ --flag / --no-flag
                    parser.add_argument(
                        no_argname, dest=name, action='store_false', default=Const.MISSING_IN_CLI,
                        help=help_text + ' (set to False)'
                    )
                    parser.add_argument(
                        argname, dest=name, action='store_true',
                        default=Const.MISSING_IN_CLI,
                        help=help_text + ' (set to True)'
                    )
                else:
                    parser.add_argument(
                        argname,
                        dest=name,
                        type=argtype if argtype is not str else str,
                        default=Const.MISSING_IN_CLI,
                        help=help_text +
                        (f' (default: {default})' if default is not None else '')
                    )
            return parser

        @classmethod
        def parse_namespace(
            cls_, ns: argparse.Namespace, kw: Dict[str, Any] | None = None,
            prefix: str = ''
        ) -> T:
            '''
            Parse an argparse.Namespace into a dataclass instance.
            '''
            if kw is None:
                kw = {}
            new_kw = {}  # Avoid modifying the original dict
            for f in dataclasses.fields(cls_):
                if prefix:
                    name = f'{prefix}_{f.name}'
                else:
                    name = f.name
                val = getattr(ns, name, Const.MISSING_IN_CLI)
                if val is Const.MISSING_IN_CLI:
                    if name in kw:
                        val = kw[name]
                    elif f.default is not dataclasses.MISSING:
                        val = f.default
                    elif f.default_factory is not dataclasses.MISSING:
                        val = f.default_factory()
                    else:
                        raise ValueError(f'Missing required argument: {name}')
                type_ = _type_hints[f.name]
                # JSON deserialization for complex types
                if isinstance(val, str):
                    val = _convert_value(val, type_)
                new_kw[f.name] = val
            return cls_(**new_kw)

        datacls.get_parser = get_parser
        datacls.parse_namespace = parse_namespace

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
            parser = _build_parser(cls_.get_parser())

            ns = parser.parse_args(argv)
            kw = _handle_config_file(ns)
            for name in kw:
                type_ = _type_hints.get(name, str)
                if isinstance(kw[name], str):
                    kw[name] = _convert_value(kw[name], type_)

            return cls_.parse_namespace(ns, kw)

        datacls.parse_args = parse_args

        if TYPE_CHECKING:
            class _TYPE_HINTS(_DECORATED[datacls],datacls): ...
            return _TYPE_HINTS

        return datacls

    return wrap if cls is None else wrap(cls)

def get_all_parser(
    dataclass = None, **dataclasses
):
    '''
    Returns a combined ``argparse.ArgumentParser`` that merges the parsers of
    multiple dataclasses into one, and then adds a ``--config`` argument for
    loading configuration files.

    The dataclass provided as a positional argument is treated as unprefixed,
    while the others are prefixed with their keyword argument names.

    Args:
        dataclass (Optional[Any]): A single dataclass to include in the parser.
        **dataclasses (Any): Additional dataclasses to include in the parser.

    Returns:
        argparse.ArgumentParser: An argument parser that includes all specified dataclasses.
    '''
    dataclasses = dataclasses.copy()
    if dataclass is not None:
        dataclasses[''] = dataclass
    parsers = []
    for name, datacls in dataclasses.items():
        parsers.append(
            datacls.get_parser(prefix=name)
        )
    return _build_parser(*parsers)

def parse_all_args(
    cli_args: List[str] | Any | None = None, dataclass: Any | None = None, **dataclasses: Any
) -> Dict[str, Any]:
    '''
    Parse command line arguments into a dictionary of dataclass instances.
    Each dataclass is identified by its name in the `dataclasses` argument.

    The ``parse_all_args`` function accepts two types of input:

    1. ``parse_all_args(cli_args, dataclass, **dataclasses)``: where ``cli_args`` is
       a list of command line arguments, ``dataclass`` is the unprefixed dataclass,
       and ``dataclasses`` are additional prefixed dataclasses.
    2. ``parse_all_args(dataclass, **dataclasses)``: where ``dataclass`` is the
       unprefixed dataclass and ``dataclasses`` are additional prefixed dataclasses.
       ``sys.argv[1:]`` is used as the command line arguments.

    The result is a dictionary where the keys are the prefixes of the
    dataclasses and the values are instances of those dataclasses, parsed from
    the command line arguments. The unprefixed dataclass is stored under the
    key ``''``.

    Example::

        @auto_cli
        @dataclasses.dataclass
        class ClassMain:
            name: str = 'MainName'
            age: int = 30

        @auto_cli
        @dataclasses.dataclass
        class ClassAdditional:
            address: str = 'DefaultAddress'
            phone: str = '1234567890'

        parser = get_all_parser(ClassMain, additional=ClassAdditional)

    The generated parser will accept the following arguments:

    * ``--name``: The name argument with a default value of ``'MainName'``.
    * ``--age``: The age argument with a default value of ``30``.
    * ``--additional-address``: The address argument with a default value of ``'DefaultAddress'``.
    * ``--additional-phone``: The phone argument with a default value of ``'1234567890'``.

    Or the following configuration files:

    .. code-block:: yaml

        name: 'MainName'
        age: 30
        additional_address: 'DefaultAddress'
        additional_phone: '1234567890'

    Args:
        cli_args (List[str] | Any | None): Command line arguments to parse. If None, uses ``sys.argv[1:]``.
        dataclass (Any | None): A single dataclass to include in the parsing.
        **dataclasses (Any): Additional dataclasses to include in the parsing.

    Returns:
        Dict[str, Any]: A dictionary where keys are dataclass names and values are instances of those dataclasses.
    '''
    pass_down = None
    if isinstance(cli_args, list) or cli_args is None:
        if cli_args is None:
            cli_args = sys.argv[1:]
        pass_down = cli_args
    else:
        pass_down = None
        dataclass = cli_args

    parser = get_all_parser(dataclass, **dataclasses)
    ns = parser.parse_args(pass_down)
    kw = _handle_config_file(ns)
    result = {}
    for name, datacls in dataclasses.items():
        result[name] = datacls.parse_namespace(ns, kw, prefix=name)
    if dataclass is not None:
        result[''] = dataclass.parse_namespace(ns, kw, prefix='')
    return result


__all__ = ['auto_cli', 'parse_all_args', 'get_all_parser']
