'''
`utils.cpolars.inspect` - Functions for inspecting and analyzing Polars DataFrames.
'''

import warnings
from typing import Any, Dict, List, Literal

import polars as pl


def to_latex(
    df: pl.DataFrame,
    cols: List[str] | None = None,
    skip_null: bool = False,
    max_rows: int | None = None,
    position: Dict[
        str, Literal['left', 'center', 'right']
    ] | Literal['left', 'center', 'right'] = 'left',
    unpack_containers: bool = False,
    formatters: Dict[str, str] | None = None
) -> str:
    '''
    Convert a Polars DataFrame to a LaTeX-formatted booktabs tabular environment.

    Notice:
    - This function DOES NOT escape LaTeX special characters in the output, so directly using it may lead to compilation errors or unintended formatting if column names or data contains LaTeX special characters. You may want to preprocess your DataFrame to escape these characters if needed.
    - For large DataFrames, consider setting ``max_rows`` to a smaller value to avoid generating excessively long LaTeX code that may be difficult to compile or display.
    - By default, no vertical lines are included in the output, as per booktabs style guidelines. You can customize the output by modifying the returned string if you want to include vertical lines or other LaTeX formatting.

    Args:
        df (pl.DataFrame): The input DataFrame to convert.
        cols (List[str] | None): The list of columns to include in the output. If None, all columns are included.
        skip_null (bool): Whether to skip rows with null values in the specified columns, or to include them with empty cells.
        max_rows (int | None): The maximum number of rows to include in the output, behaves identically to ``DataFrame.head()``. If None, all rows are included.
        position (Dict[str, Literal['left', 'center', 'right']] | Literal['left', 'center', 'right']): The alignment for each column. Can be a single value applied to all columns or a dictionary specifying alignment for each column. For any unspecified column, defaults to 'left' alignment.
        unpack_containers (bool): Whether to unpack container types (e.g., lists, structs) when formatting using custom formatters.
        formatters (Dict[str, str] | None): Custom format strings for specific columns, where the key is the column name and the value is a format string compatible with Python's ``str.format()``.

    Returns:
        str: A LaTeX-formatted string representing the DataFrame.
    '''
    # Validate input parameters
    if cols is not None:
        cols = list(dict.fromkeys(cols))
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f'Columns {missing} do not exist in DataFrame')
    else:
        cols = df.columns
    if not cols:
        raise ValueError('No columns selected for output')

    if isinstance(position, str):
        position = {col: position for col in cols}
    else:
        missing_pos = [col for col in position if col not in cols]
        if missing_pos:
            raise ValueError(
                f'Columns {missing_pos} in position dict are not selected for output')
    if any(pos not in ['left', 'center', 'right'] for pos in position.values()):
        raise ValueError(
            'Position values must be one of "left", "center", or "right".')
    formatters = formatters or {}
    missing_fmt = [col for col in formatters if col not in cols]
    if missing_fmt:
        raise ValueError(
            f'Columns {missing_fmt} in formatters dict are not selected for output')

    # Select specified range
    df = df.select(cols)
    if skip_null:
        df = df.drop_nulls(cols)
    if max_rows is not None and max_rows < len(df):
        df = df.head(max_rows)
    if len(df) > 500:
        warnings.warn(
            f'Output has {len(df)} rows, which may be too large to display. Consider setting ``max_rows`` to a smaller value.', stacklevel=2
        )

    align_map = {k: k[0] for k in ['left', 'center', 'right']}
    line_formatter = ' & '.join(['{}'] * len(cols)) + r' \\'
    header = line_formatter.format(*cols)
    col_spec = ''.join([
        align_map.get(position.get(c, 'left'), 'l') for c in cols
    ])

    # Unpack container types if needed
    def _format_cell(
        value: Any, col: str, unpack: bool, dtype: pl.DataType
    ) -> str:
        try:
            fmt = formatters.get(col, '{}')
            match unpack, dtype:
                case True, pl.List() | pl.Array():
                    return fmt.format(*value)
                case True, pl.Struct():
                    return fmt.format(**value)
                case True, pl.Object():
                    if hasattr(value, '__dict__'):
                        return fmt.format(**vars(value))
                    return fmt.format(value)
                case _:
                    return fmt.format(value)
        except Exception as e:
            warnings.warn(
                f'Error formatting cell in column {col} with value {value}: {e}. Falling back to str formatting.', stacklevel=2
            )
            return str(value)

    map_fns = {
        c: lambda x, _c=c: _format_cell(
            x, _c, unpack_containers, df.schema[_c],
        )
        for c in cols
    }
    map_exprs = {
        c: pl.col(c).map_elements(map_fns[c], return_dtype=pl.String)
        for c in cols if c in formatters
    }

    df_formatted = df.select(pl.format(line_formatter, *[
        map_exprs.get(c, pl.col(c).cast(pl.String)).fill_null('').alias(c)
        for c in cols
    ]).alias('result')).to_series().str.join('\n').item()  # Tested: '' if empty

    return \
f'''
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header}
\\midrule
{df_formatted}
\\bottomrule
\\end{{tabular}}
'''.strip()


def to_markdown(
    df: pl.DataFrame,
    cols: List[str] | None = None,
    skip_null: bool = False,
    max_rows: int | None = None,
    position: Dict[
        str, Literal['left', 'center', 'right']
    ] | Literal['left', 'center', 'right'] = 'left',
    unpack_containers: bool = False,
    formatters: Dict[str, str] | None = None
) -> str:
    '''
    Convert a Polars DataFrame to a Markdown-formatted string.

    Args:
        df (pl.DataFrame): The input DataFrame to convert.
        cols (List[str] | None): The list of columns to include in the output. If None, all columns are included.
        skip_null (bool): Whether to skip rows with null values in the specified columns, or to include them with empty cells.
        max_rows (int | None): The maximum number of rows to include in the output, behaves identically to ``DataFrame.head()``. If None, all rows are included.
        position (Dict[str, Literal['left', 'center', 'right']] | Literal['left', 'center', 'right']): The alignment for each column. Can be a single value applied to all columns or a dictionary specifying alignment for each column. For any unspecified column, defaults to 'left' alignment.
        unpack_containers (bool): Whether to unpack container types (e.g., lists, structs) when formatting using custom formatters.
        formatters (Dict[str, str] | None): Custom format strings for specific columns, where the key is the column name and the value is a format string compatible with Python's ``str.format()``.

    Returns:
        str: A Markdown-formatted string representing the DataFrame.
    '''
    # Validate input parameters
    if cols is not None:
        cols = list(dict.fromkeys(cols))
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f'Columns {missing} do not exist in DataFrame')
    else:
        cols = df.columns
    if not cols:
        raise ValueError('No columns selected for output')

    if isinstance(position, str):
        position = {col: position for col in cols}
    else:
        missing_pos = [col for col in position if col not in cols]
        if missing_pos:
            raise ValueError(f'Columns {missing_pos} in position dict are not selected for output')
    if any(pos not in ['left', 'center', 'right'] for pos in position.values()):
        raise ValueError('Position values must be one of "left", "center", or "right".')
    formatters = formatters or {}
    missing_fmt = [col for col in formatters if col not in cols]
    if missing_fmt:
        raise ValueError(f'Columns {missing_fmt} in formatters dict are not selected for output')

    # Select specified range
    df = df.select(cols)
    if skip_null:
        df = df.drop_nulls(cols)
    if max_rows is not None and max_rows < len(df):
        df = df.head(max_rows)
    if len(df) > 500:
        warnings.warn(
            f'Output has {len(df)} rows, which may be too large to display. Consider setting ``max_rows`` to a smaller value.', stacklevel=2
        )

    align_map = {
        'left': ':---',
        'center': ':---:',
        'right': '---:'
    }
    line_formatter = ' {} '.join(['|'] * (len(cols) + 1))
    header = line_formatter.format(*[
        _.replace('\\', '\\\\').replace('|', '\\|').replace('\n', '<br>') \
        for _ in cols
    ])
    separator = line_formatter.format(*[
        align_map.get(position.get(c, 'left'), ':---') for c in cols
    ])

    # Unpack container types if needed
    def _format_cell(
        value: Any, col: str, unpack: bool, dtype: pl.DataType
    ) -> str:
        try:
            fmt = formatters.get(col, '{}')
            match unpack, dtype:
                case True, pl.List() | pl.Array():
                    return fmt.format(*value)
                case True, pl.Struct():
                    return fmt.format(**value)
                case True, pl.Object():
                    if hasattr(value, '__dict__'):
                        return fmt.format(**vars(value))
                    return fmt.format(value)
                case _:
                    return fmt.format(value)
        except Exception as e:
            warnings.warn(
                f'Error formatting cell in column {col} with value {value}: {e}. Falling back to str formatting.', stacklevel=2
            )
            return str(value)

    map_fns = {
        c: lambda x, _c=c: _format_cell(
            x, _c, unpack_containers, df.schema[_c],
        )
        for c in cols
    }
    map_exprs = {
        c: pl.col(c).map_elements(map_fns[c], return_dtype=pl.String)
        for c in cols if c in formatters
    }

    df_formatted = df.select(pl.format(line_formatter, *[
        map_exprs.get(c, pl.col(c).cast(pl.String))
            .fill_null('')
            .str.replace_all('\\', '\\\\', literal=True)
            .str.replace_all('|', '\\|', literal=True)
            .str.replace_all('\n', '<br>', literal=True)
            .alias(c)
        for c in cols
    ]).alias('result')).to_series().str.join('\n').item() # Tested: '' if empty

    return '\n'.join([header, separator, df_formatted])


def flatten_describe(
    df: pl.DataFrame, keys: List[str],
    *describe_args, **describe_kwargs
) -> pl.DataFrame:
    '''
    Flatten the output of `df.describe()` into a one-row DataFrame with columns named as "statistic_column".

    Args:
        df (pl.DataFrame): The input DataFrame to describe.
        keys (List[str]): The list of statistics to include.
        *describe_args: Additional arguments to pass to `df.describe()`.
        **describe_kwargs: Additional keyword arguments to pass to `df.describe()`.

    Returns:
        pl.DataFrame: A flattened DataFrame containing the descriptive statistics.
    '''
    return df.describe(*describe_args, **describe_kwargs).filter(
        pl.col('statistic').is_in(keys)
    ).unpivot(
        index='statistic',
        value_name='value',
        variable_name='column'
    ).filter(
        pl.col('value').is_not_null()
    ).with_columns(
        (pl.col('statistic') + '_' + pl.col('column')).alias('column_stat'),
        pl.lit(0).alias('idx')
    ).pivot(
        on='column_stat',
        index='idx',
        values='value',
        sort_columns=True
    ).drop('idx')
