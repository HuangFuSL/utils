import warnings
from typing import Any, Dict, List, Literal, Sequence, Type

import polars as pl
from polars.datatypes.classes import NumericType


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


def add_shard_column(
    df: pl.DataFrame, num_shards: int, dest_col: str,
    src_col: str | Sequence[str] | None = None, seed: int | None = None
) -> pl.DataFrame:
    '''
    Compute and add a shard column to a Polars DataFrame for sharding data across multiple workers.

    Args:
        df (pl.DataFrame): The input DataFrame.
        num_shards (int): The number of shards to create.
        dest_col (str): The name of the destination shard column.
        src_col (str | Sequence[str] | None): The source column(s) to use for assigning the shard. If None, a random shard is assigned.
        seed (int | None): The random seed for shard assignment.

    Returns:
        pl.DataFrame: The DataFrame with the added shard column.
    '''
    # Check column existance
    if dest_col in df.columns:
        raise ValueError(
            f'Destination column {dest_col} already exists in DataFrame'
        )
    if src_col is not None:
        if isinstance(src_col, str):
            src_col = [src_col]
        if not src_col:
            raise ValueError('src_col must not be empty if provided')
        if not all(col in df.columns for col in src_col):
            missing = [col for col in src_col if col not in df.columns]
            raise ValueError(
                f'Source columns {missing} do not exist in DataFrame'
            )
    if num_shards <= 0:
        raise ValueError('num_shards must be a positive integer')

    if src_col is None:
        shard_expr = (
            pl.int_range(0, pl.len(), eager=False)
            .hash(seed_1=seed)
            .mod(num_shards)
            .cast(pl.Int64)
            .alias(dest_col)
        )
    else:
        shard_expr = (
            pl.struct(src_col)
            .hash(seed_1=seed)
            .mod(num_shards)
            .cast(pl.Int64)
            .alias(dest_col)
        )
    return df.with_columns(shard_expr)

def pad_list(
    df: pl.DataFrame,
    col: str,
    side: Literal['left', 'right'] = 'right',
    pad_value: Any = None,
    max_length: int | None = None
) -> pl.DataFrame:
    '''
    Pad list columns in a Polars DataFrame to a specified length with a given padding value.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col (str): The column to perform padding.
        side (Literal['left', 'right']): The side to pad on, either 'left' or 'right'.
        pad_value (Any): The value to use for padding.
        max_length (int | None): The maximum length to pad to. If None, pads to the length of the longest list in each column. If specified and less than the length of any list, those lists will be truncated.
    '''
    if col not in df.columns:
        raise ValueError(f'Column {col} does not exist in DataFrame')
    container_dtype = df.schema[col]
    if not isinstance(container_dtype, pl.List):
        raise ValueError(f'Column {col} must be of list type to pad')
    df = df.with_columns(pl.col(col).fill_null([]).cast(container_dtype))
    if side not in ['left', 'right']:
        raise ValueError("side must be either 'left' or 'right'")
    if max_length is not None and max_length < 0:
        raise ValueError('max_length must be a non-negative integer')
    if max_length is None:
        max_length = df.select(pl.col(col).list.len().max()).item()
    assert max_length is not None

    try:
        dtype_cls = container_dtype.inner.to_python()
        if pad_value is not None:
            if not isinstance(pad_value, dtype_cls):
                # Try casting
                pad_value_casted = dtype_cls(pad_value) # type: ignore
            else:
                pad_value_casted = pad_value
        else:
            pad_value_casted = None
        pad_value_casted: Any
    except Exception as e:
        raise ValueError(f'pad_value {pad_value} cannot be cast to the inner type of the list column: {e}')

    expr = pl.col(col)
    min_length = df.select(pl.col(col).list.len().min()).item()
    num_required = max(0, max_length - min_length)

    if side == 'right':
        expr = expr \
            .list.concat(pl.repeat(pad_value_casted, num_required)) \
            .list.head(max_length) \
            .alias(col)
    else:
        expr = pl.repeat(pad_value_casted, num_required) \
            .list.concat(expr) \
            .list.tail(max_length) \
            .alias(col)
    return df.with_columns(expr)

def expand_columns(
    df: pl.DataFrame, src_cols: str | List[str],
    name_template: str = '{col}_{index}',
    drop_src: bool = True
) -> pl.DataFrame:
    '''
    Expand a column containing list or struct types into multiple columns with names generated from the specified template.

    Args:
        df (pl.DataFrame): The input DataFrame.
        src_cols (str | List[str]): The source column(s) to expand. Each column must be of list or struct type.
        name_template (str): The template for generating new column names, where '{col}' will be replaced with the source column name and '{index}' will be replaced with the list index or struct field name.
        drop_src (bool): Whether to drop the source columns after expansion.

    Returns:
        pl.DataFrame: The DataFrame with the expanded columns.
    '''
    if isinstance(src_cols, str):
        src_cols = [src_cols]
    if not src_cols:
        raise ValueError('src_cols must not be empty')
    if len(set(src_cols)) != len(src_cols):
        raise ValueError('src_cols must not contain duplicate column names')
    if any(col not in df.columns for col in src_cols):
        missing = [col for col in src_cols if col not in df.columns]
        raise ValueError(f'Source columns {missing} do not exist in DataFrame')

    all_exprs = []
    for src_col in src_cols:
        src_dtype = df.schema[src_col]

        # Convert to struct
        expr = pl.col(src_col)
        match src_dtype:
            case pl.List():
                expr = expr.list.to_struct(
                    n_field_strategy='max_width',
                    fields=lambda i: name_template.format(col=src_col, index=i)
                )
            case pl.Array():
                expr = expr.arr.to_struct(
                    fields=lambda i: name_template.format(col=src_col, index=i)
                )
            case pl.Struct() as struct:
                fields = struct.fields
                expr = expr.struct.rename_fields([
                    name_template.format(col=src_col, index=f) for f in fields
                ])
            case _:
                raise ValueError(f'Source column {src_col} must be of list, array or struct type to expand')
        expr = expr.struct.unnest()
        all_exprs.append(expr)

    ret = df.with_columns(*all_exprs)
    if drop_src:
        ret = ret.drop(src_cols)
    return ret


def merge_columns(
    df: pl.DataFrame, src_cols: Sequence[str], dest_col: str,
    drop_src: bool = False, target_dtype: Type[NumericType] = pl.Float32
) -> pl.DataFrame:
    '''
    Merge specified columns in a Polars DataFrame into a single column with values concatenated in a list.

    Args:
        df (pl.DataFrame): The input DataFrame.
        src_cols (Sequence[str]): The columns to merge.
        dest_col (str): The name of the destination column.
        drop_src (bool): Whether to drop the source columns after merging.

    Returns:
        pl.DataFrame: The DataFrame with merged column.
    '''
    # Check columns exist
    if not src_cols:
        raise ValueError('src_cols must not be empty')
    if dest_col in df.columns:
        raise ValueError(
            f'Destination column {dest_col} already exists in DataFrame')
    if not all(col in df.columns for col in src_cols):
        missing = [col for col in src_cols if col not in df.columns]
        raise ValueError(f'Source columns {missing} do not exist in DataFrame')
    if not all(df.schema[col].is_numeric() for col in src_cols):
        raise ValueError('All source columns must be numeric')
    if not target_dtype.is_numeric():
        raise ValueError('target_dtype must be a numeric type')

    df = df.with_columns(
        pl.concat_list(pl.col(src_cols).cast(target_dtype)).alias(dest_col)
    )
    if drop_src:
        df = df.drop(src_cols)
    return df


def sparse_to_index(
    df: pl.DataFrame, sparse_cols: Sequence[str], starting_index: int = 1,
    null_index: int | None = None
) -> pl.DataFrame:
    '''
    Convert specified columns in a Polars DataFrame to incremental index based on unique values in the current column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        sparse_cols (Sequence[str]): The columns to convert.
        starting_index (int): The starting index for the conversion.
        null_index (int | None): The index to assign to null values. If None, null values will raise an error.

    Returns:
        pl.DataFrame: The DataFrame with specified columns converted to sparse representation.
    '''
    if null_index is not None and null_index >= starting_index:
        raise ValueError(
            f'null_index {null_index} must be less than starting_index {starting_index}'
        )
    if any(col not in df.columns for col in sparse_cols):
        missing = [col for col in sparse_cols if col not in df.columns]
        raise ValueError(f'Sparse columns {missing} do not exist in DataFrame')
    if any(f'{col}_index' in df.columns for col in sparse_cols):
        existing = [col for col in sparse_cols if f'{col}_index' in df.columns]
        raise ValueError(
            f'Destination columns {[f"{col}_index" for col in existing]} already exist in DataFrame'
        )

    for col in sparse_cols:
        # Null and NaN check
        if null_index is None and df.select(pl.col(col).is_null().any()).item():
            raise ValueError(f'Column {col} contains null values')

        dtype = df.schema[col]
        if dtype == pl.Float32 or dtype == pl.Float64:
            if df.select(pl.col(col).is_nan().any()).item():
                raise ValueError(f'Column {col} contains NaN values')
        # Create mapping
        mapping = df \
            .select(pl.col(col)) \
            .filter(pl.col(col).is_not_null()) \
            .unique() \
            .sort(col) \
            .with_row_index(f'{col}_index', offset=starting_index) \
            .with_columns(**{
                f'{col}_index': pl.col(f'{col}_index').cast(pl.Int64)
            })
        df = df.join(mapping, on=col, how='left')
        if null_index is not None:
            df = df.with_columns(pl.col(f'{col}_index').fill_null(null_index))

    df = df \
        .drop(sparse_cols) \
        .rename({
            f'{col}_index': col for col in sparse_cols
        })
    return df
