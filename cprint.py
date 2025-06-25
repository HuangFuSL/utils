'''
cprint.py

Author: HuangFuSL
Date: 2025-06-24

A Python module for colorful terminal output with support for ANSI escape codes,
xterm-256color, and true color (24-bit RGB). It provides functions to format
text with foreground and background colors, styles (bold, italic, underline,
strikethrough), and fallback to lower color grades if the terminal does not
support the requested color depth.
'''

import os
import warnings
from typing import Tuple

ESC = '\033['
RESET = '\033[0m'
# ANSI 8 Colors
BASE_COLOR = [
    'black', 'red', 'green',  'yellow', 'blue',  'magenta', 'cyan', 'white'
]
COLOR_CODES = {}
for i, name in enumerate(BASE_COLOR):
    COLOR_CODES[name] = 30 + i                    # Foreground
    COLOR_CODES[f'bright_{name}'] = 90 + i        # Bright foreground
    COLOR_CODES[f'bg_{name}'] = 40 + i            # Background
    COLOR_CODES[f'bg_bright_{name}'] = 100 + i    # Bright background

# Detect terminal support for colors
# In number of bits
if os.environ.get("COLORTERM", '').lower() in ("truecolor", "24bit"):
    COLOR_GRADE = 24
elif os.environ.get("TERM", '').lower() in ("xterm-256color", "screen-256color"):
    COLOR_GRADE = 8
else:
    COLOR_GRADE = 4


_XTERM_LEVELS = (0, 95, 135, 175, 215, 255)
_RGB_4BIT = (
    (0, 0, 0),         # 0 black
    (128, 0, 0),       # 1 red
    (0, 128, 0),       # 2 green
    (128, 128, 0),     # 3 yellow
    (0, 0, 128),       # 4 blue
    (128, 0, 128),     # 5 magenta
    (0, 128, 128),     # 6 cyan
    (192, 192, 192),   # 7 white/gray
    (128, 128, 128),   # 8 bright black (gray)
    (255, 0, 0),       # 9 bright red
    (0, 255, 0),       # 10 bright green
    (255, 255, 0),     # 11 bright yellow
    (0, 0, 255),       # 12 bright blue
    (255, 0, 255),     # 13 bright magenta
    (0, 255, 255),     # 14 bright cyan
    (255, 255, 255),   # 15 bright white
)

def _distance(c1: Tuple[int, ...], c2: Tuple[int, ...]) -> float:
    ''' Calculate the Euclidean distance between two RGB tuples. '''
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def _nearest_level(v: int) -> int:
    ''' Return the nearest xterm level for a given RGB value (0-255). '''
    return min(range(6), key=lambda i: abs(_XTERM_LEVELS[i] - v))


def _cube_index(r: int, g: int, b: int) -> Tuple[int, Tuple[int, int, int]]:
    ''' Mapping RGB to xterm-256color cube index. '''
    ri, gi, bi = map(_nearest_level, (r, g, b))
    idx = 16 + 36 * ri + 6 * gi + bi
    return idx, (_XTERM_LEVELS[ri], _XTERM_LEVELS[gi], _XTERM_LEVELS[bi])


def _gray_index(r: int, g: int, b: int) -> Tuple[int, Tuple[int, int, int]]:
    ''' Map RGB to xterm-256color gray index. '''
    y = round((r + g + b) / 3)
    k = max(0, min(23, round((y - 8) / 10)))
    idx = 232 + k
    gval = 8 + 10 * k
    return idx, (gval, gval, gval)


def rgb_to_xterm256(r: int, g: int, b: int) -> int:
    '''
    Map RGB values (0-255) to xterm-256color index (0-255).
    '''
    idx_cube, rgb_cube = _cube_index(r, g, b)
    idx_gray, rgb_gray = _gray_index(r, g, b)

    # 计算平方欧氏距离
    dist_cube = _distance((r, g, b), rgb_cube)
    dist_gray = _distance((r, g, b), rgb_gray)

    return idx_gray if dist_gray < dist_cube else idx_cube


def xterm256_to_rgb(idx: int) -> Tuple[int, int, int]:
    '''
    Map xterm-256color index (0-255) to RGB tuple (0-255).
    '''
    if idx < 16: # System colors
        return _RGB_4BIT[idx]
    if idx < 232: # Cube colors
        idx -= 16
        r = _XTERM_LEVELS[idx // 36]
        g = _XTERM_LEVELS[(idx // 6) % 6]
        b = _XTERM_LEVELS[idx % 6]
        return r, g, b

    gray = 8 + 10 * (idx - 232)
    return gray, gray, gray


def xterm256_to_4bit(idx: int) -> str:
    '''
    Map xterm-256color index (0-255) to 4-bit ANSI color name.
    '''
    r, g, b = xterm256_to_rgb(idx)
    best = min(
        range(16),
        key=lambda j: _distance((r, g, b), _RGB_4BIT[j])
    )
    return [
        *BASE_COLOR, *[f'bright_{c}' for c in BASE_COLOR]
    ][best]

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    '''
    Convert a hex color code in the format #RRGGBB or #RGB to an RGB tuple.
    '''
    if not hex_color.startswith('#'):
        raise ValueError('Hex color code must start with #')
    if len(hex_color) == 7:  # #RRGGBB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    elif len(hex_color) == 4:  # #RGB
        r = int(hex_color[1] * 2, 16)
        g = int(hex_color[2] * 2, 16)
        b = int(hex_color[3] * 2, 16)
    else:
        raise ValueError('Hex color code must be in #RRGGBB or #RGB format')
    return r, g, b

def _fallback_color(
    color: str | int | Tuple[int, int, int] | None, grade: int
) -> str | int | Tuple[int, int, int] | None:
    ''' Return a fallback color based on the current terminal's color support. '''
    if color is None:
        return None
    if grade <= 8:
        # Convert true color to xterm-256color
        if isinstance(color, str) and color.startswith('#'):
            color = hex_to_rgb(color)
        if isinstance(color, (tuple, list)):
            color = rgb_to_xterm256(*color)
    if grade <= 4:
        # Convert xterm-256color to 4-bit ANSI color name
        if isinstance(color, int):
            color = xterm256_to_4bit(color)
    return color


def _encode(
    color: str | int | Tuple[int, int, int] | None, is_bg=False, fallback: bool = True
):
    ''' Return the ANSI escape code for a given color. '''
    if color is None:
        return None, 4
    if fallback:
        color = _fallback_color(color, COLOR_GRADE)
        return _encode(color, is_bg=is_bg, fallback=False)
    # 1) ANSI color or hex color
    if isinstance(color, str):
        if color.startswith('#'):
            color_tuple = hex_to_rgb(color)
            return _encode(color_tuple, is_bg=is_bg, fallback=fallback)
        if is_bg and not color.startswith('bg_'):
            key = f'bg_{color}'
        else:
            key = color
        code = COLOR_CODES.get(key)
        if code is None:
            raise ValueError(f'Unknown color: {color}')
        return str(code), 4
    # 2) xterm-256color
    if isinstance(color, int):
        if not 0 <= color <= 255:
            raise ValueError('xterm color index should in range 0–255')
        return f'{48 if is_bg else 38};5;{color}', 8
    # 3) RGB tuple
    if isinstance(color, (tuple, list)) and len(color) == 3:
        r, g, b = color
        if not all(0 <= v <= 255 for v in (r, g, b)):
            raise ValueError('RGB values should be in range 0–255')
        return f'{48 if is_bg else 38};2;{r};{g};{b}', 24
    raise TypeError('Color must be str, int, tuple or None')

def cprefix(
    fg: str | int | Tuple[int, int, int] | None = None,
    bg: str | int | Tuple[int, int, int] | None = None,
    fallback: bool = True,
    bf: bool = False,
    dim: bool = False,
    it: bool = False,
    us: bool = False,
    st: bool = False,
) -> str:
    '''
    Generate an ANSI escape code prefix for formatting text with colors and styles.

    Parameters:
    - fg: Foreground color (color name, xterm-256color index, RGB tuple, or hex code).
    - bg: Background color (same format as fg).
    - fallback: Whether to use fallback colors if the terminal does not support
                the requested color depth.
    - bf: Whether to use bold text.
    - it: Whether to use italic text.
    - us: Whether to underline the text.
    - st: Whether to use strikethrough text (not supported in all terminals).

    Returns:
    A string containing the ANSI escape code prefix.
    '''

    seq = []
    if bf:
        seq.append('1')
    if dim:
        seq.append('2')
    if it:
        seq.append('3')
    if us:
        seq.append('4')
    if st:
        seq.append('9')

    fg_code, fg_grade = _encode(fg, is_bg=False, fallback=fallback)
    bg_code, bg_grade = _encode(bg, is_bg=True, fallback=fallback)

    if fg_grade > COLOR_GRADE or bg_grade > COLOR_GRADE:
        msg = f'The terminal does not support the requested color depth: ' \
              f'fg={fg_grade}, bg={bg_grade}.'
        warnings.warn(msg, RuntimeWarning, stacklevel=0)

    if fg_code:
        seq.append(fg_code)
    if bg_code:
        seq.append(bg_code)

    # Construct the ANSI escape sequence
    return ESC + ';'.join(seq) + 'm' if seq else ''


def cformat(
    text: str, *,
    fg: str | int | Tuple[int, int, int] | None = None,
    bg: str | int | Tuple[int, int, int] | None = None,
    fallback: bool = True,
    bf: bool = False,
    dim: bool = False,
    it: bool = False,
    us: bool = False,
    st: bool = False,
    reset: str = RESET
) -> str:
    '''
    Format a string with ANSI escape codes for colors and styles.

    Parameters:
    - text: The text to format.
    - fg: Foreground color (color name, xterm-256color index, RGB tuple, or hex code).
    - bg: Background color (same format as fg).
    - fallback: Whether to use fallback colors if the terminal does not support
                the requested color depth.
    - bf: Whether to use bold text.
    - it: Whether to use italic text.
    - us: Whether to underline the text.
    - st: Whether to use strikethrough text (not supported in all terminals).
    - reset: String to append after the formatted text (default is ANSI reset code).

    Returns:
    A formatted string with ANSI escape codes.
    '''
    prefix = cprefix(
        fg=fg, dim=dim, bg=bg, fallback=fallback,
        bf=bf, it=it, us=us, st=st
    )
    if not prefix:
        return text
    return f'{prefix}{text}{reset}'

def cprint(
    *obj,
    fg: str | int | Tuple[int, int, int] | None = None,
    bg: str | int | Tuple[int, int, int] | None = None,
    fallback: bool = True,
    bf: bool = False,
    dim: bool = False,
    it: bool = False,
    us: bool = False,
    st: bool = False,
    reset: str = RESET,
    sep: str = ' ',
    end: str = '\n',
    file=None,
    flush: bool = False,
):
    '''
    Colorful print function with support for foreground and background colors,
    styles (bold, italic, underline, strikethrough), and fallback to lower color
    grades if the terminal does not support the requested color depth.

    Parameters:
    - obj: Objects to print.
    - fg: Foreground color (color name, xterm-256color index, RGB tuple, or hex code).
    - bg: Background color (same format as fg).
    - fallback: Whether to use fallback colors if the terminal does not support
                the requested color depth.
    - bf: Whether to use bold text.
    - it: Whether to use italic text.
    - us: Whether to underline the text.
    - st: Whether to use strikethrough text (not supported in all terminals).
    - reset: String to append after the formatted text (default is ANSI reset code).

    The following parameters are inherited from the built-in print function:
    - sep: Separator between objects.
    - end: String appended after the last object.
    - file: A file-like object (default is sys.stdout).
    - flush: Whether to forcibly flush the output buffer.
    '''
    print(
        *[
            cformat(
                str(_), fg=fg, dim=dim, bg=bg, fallback=fallback,
                bf=bf, it=it, us=us, st=st, reset=reset
            ) for _ in obj
        ],
        sep=sep, end=end, file=file, flush=flush
    )

__all__ = ['cprefix', 'cprint', 'cformat', 'RESET']
