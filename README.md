# utils
Utilities for Python projects

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
