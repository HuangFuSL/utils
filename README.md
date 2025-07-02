# utils
Utilities for Python projects

## Usage

First mount this repo as a submodule under the source tree of your project. For example if your project is in `src/` directory.

```bash
git submodule add https://github.com/HuangFuSL/utils src/utils
```

Then you can import the modules in your Python code:

```python
from utils import cprint
```

Use the following script to update the submodule:

```bash
git submodule update --remote --merge
```

## `cprint.py` - Colorful print functions

* **Function:** `cprint()`

    Colorful print function with support for foreground and background colors,
    styles (bold, italic, underline, strikethrough), and fallback to lower color
    grades if the terminal does not support the requested color depth.

    Parameters:
    - `*obj`: Objects to print.
    - `fg`: Foreground color (color name, xterm-256color index, RGB tuple, or hex code).
    - `bg`: Background color (same format as fg).
    - `fallback`: Whether to use fallback colors if the terminal does not support
                the requested color depth.
    - `bf`: Whether to use bold text.
    - `it`: Whether to use italic text.
    - `us`: Whether to underline the text.
    - `st`: Whether to use strikethrough text (not supported in all terminals).
    - `reset`: String to append after the formatted text (default is ANSI reset code).

    The following parameters are inherited from the built-in print function:
    - `sep`: Separator between objects.
    - `end`: String appended after the last object.
    - `file`: A file-like object (default is sys.stdout).
    - `flush`: Whether to forcibly flush the output buffer.

## `clogging.py` - Colorful loggers

* **Class:** `BaseColoredFormatter`

    Add color to log messages based on their level.

    This formatter allows customization of log message styles based on their
    logging level. It supports coloring the level name and optionally the
    message itself. The styles can be customized through the `get_color` method,
    which should return a argument dictionary compatible with `cprint.cprint`.

    Parameters:
    - fmt: The format string for the log messages.
    - datefmt: The format string for the date in log messages.
    - style: The style character used in the format string (default is '%').
    (same as logging.Formatter)

* **Class:** `DefaultColoredFormatter`

    A colored formatter that

    * Shows full critical messages in bolded white text on a red background.
    * Time is shown in bolded green text.
    * Level names are shown in gray, green, yellow and bolded red, for DEBUG, INFO,
      WARNING and ERROR respectively.
    * Logger names are shown in cyan text.
    * Files, line numbers and function names are shown in magenta text.

## `ctorch.py` - Torch utilities

* **Class:** `Module`

    A base class for PyTorch modules that provides a special property `device`.

* **Function:** `pad_packed_sequence_right`

    Like `torch.nn.utils.rnn.pad_packed_sequence` but right-aligns the sequences.

* **Function:** `pack_padded_sequence_right`

    Like `torch.nn.utils.rnn.pack_padded_sequence` but accepts right-aligned sequences.

* **Function:** `unpad_sequence_right`

    Like `torch.nn.utils.rnn.unpad_sequence` but accepts right-aligned sequences.

* **Function:** `get_key_padding_mask_left`

    Builds a key padding mask for left-aligned sequences.

* **Function:** `get_key_padding_mask_right`

    Builds a key padding mask for right-aligned sequences.

* **Function:** `get_tensor_memory_size`

    Returns the memory size of a tensor in bytes.

* **Function:** `get_model_memory_size`

    Returns the memory size of a PyTorch model in bytes.

## `cstats.py` - Statistics utilities in PyTorch

* **Function:** `log_norm_pdf`

    Computes the log probability density function of a multivariate normal distribution.

    Parameters:
    - `x`: Input tensor.
    - `mean`: Mean vector.
    - `Sigma`: Covariance matrix (can be a scalar, diagonal, or full matrix).
    - `logSigma`: Logarithm of the covariance matrix (optional).
    - `batch_first`: Whether the first dimension of `x` is the batch size.

    Returns:
    - Log probability density values for each input sample.

* **Function:** `norm_pdf`

    Same as `log_norm_pdf` but returns the probability density function values instead of log values.

## `parser.py` - Automatic argument parser

* **Decorator:** `auto_cli`

    Automatically generates an argument parser for a dataclass with type hints.
    Field types, including basic types, and `Optional` are automatically
    recognized and converted corresponding argument types. Complex types like
    lists and dictionaries can be input as strings in json format.

    Apart from the fields of the dataclass, `auto_cli` also adds a `--config`
    argument to the parser for loading configuration settings from a json, yaml,
    or toml file. Values from the configuration file have a lower priority than
    command line arguments.

    The decorated function will get the following new class methods:

    - `get_parser(prefix: str = '') -> argparse.ArgumentParser`:
        Returns an `argparse.ArgumentParser` instance with arguments based on
        the dataclass fields. The `prefix` argument is prepended to each
        argument name. The parser returned does not include the `--config`
        argument for loading configuration files.

        Example:
        ```python
        @auto_cli
        @dataclasses.dataclass
        class YourClass:
            name: str = 'DefaultName'
            age: int = 25

        parser = YourClass.get_parser()
        parser_prefix = YourClass.get_parser(prefix='man')
        ```

        The generated `parser` will accept the following arguments:

        * `--name`: The name argument with a default value of 'DefaultName'.
        * `--age`: The age argument with a default value of 25.

        The generated `parser_prefix` will accept the following arguments:

        * `--man-name`: The name argument with a default value of 'DefaultName'.
        * `--man-age`: The age argument with a default value of 25.

    - `parse_namespace(ns: argparse.Namespace, kw: Dict[str, Any] = None, prefix: str = '') -> 'YourClass'`:
        Parses an `argparse.Namespace` instance and a kwargs dictionary into an
        instance of the dataclass. Prefix is prepended to each argument name.
        The `kw` dictionary is loaded from a config file if specified in `ns`.
        A `ValueError` is raised if a required argument is missing from both
        `ns` and `kw`, and no default value is provided.
    - `parse_args(argv: List[str] = None) -> 'YourClass'`:
        Parses command line arguments and returns an instance of the dataclass.
        If `argv` is `None`, it uses `sys.argv[1:]`. The method also handles
        config files specified in the command line arguments.

* **Function:** `get_all_parser(dataclass, **dataclasses) -> argparse.ArgumentParser`

    Returns a combined `argparse.ArgumentParser` that merges the parsers of
    multiple dataclasses into one, and then adds a `--config` argument for
    loading configuration files.

    The dataclass provided as a positional argument is treated as unprefixed,
    while the others are prefixed with their keyword argument names.

* **Function:** `parse_all_args() -> Dict[str, 'YourClass']`

    The `parse_all_args` function accepts two types of input:

    1. `parse_all_args(cli_args, dataclass, **dataclasses)`: where `cli_args` is
    a list of command line arguments, `dataclass` is the unprefixed dataclass,
    and `dataclasses` are additional prefixed dataclasses.
    2. `parse_all_args(dataclass, **dataclasses)`: where `dataclass` is the
    unprefixed dataclass and `dataclasses` are additional prefixed dataclasses.
    `sys.argv[1:]` is used as the command line arguments.

    The result is a dictionary where the keys are the prefixes of the
    dataclasses and the values are instances of those dataclasses, parsed from
    the command line arguments. The unprefixed dataclass is stored under the
    key `''`.

    Example:
    ```python
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
    ```

    The generated parser will accept the following arguments:

    * `--name`: The name argument with a default value of 'MainName'.
    * `--age`: The age argument with a default value of 30.
    * `--additional-address`: The address argument with a default value of 'DefaultAddress'.
    * `--additional-phone`: The phone argument with a default value of '1234567890'.

    Or the following configuration files:

    ```yaml
    name: 'MainName'
    age: 30
    additional_address: 'DefaultAddress'
    additional_phone: '1234567890'
    ```
