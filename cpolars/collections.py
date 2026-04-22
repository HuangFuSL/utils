'''
`utils.cpolars.collections` - Functions for manipulating containers (lists, arrays, structs) in Polars DataFrames.
'''


from typing import Any, List, Literal, Sequence, Type

import polars as pl
from polars.datatypes.classes import NumericType

from .helpers import AnyFrame, get_columns, get_schema


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
    columns = get_columns(df)
    if col not in columns:
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
    schema = get_schema(df)
    columns = schema.names()
    if isinstance(src_cols, str):
        src_cols = [src_cols]
    if not src_cols:
        raise ValueError('src_cols must not be empty')
    if len(set(src_cols)) != len(src_cols):
        raise ValueError('src_cols must not contain duplicate column names')
    if any(col not in columns for col in src_cols):
        missing = [col for col in src_cols if col not in columns]
        raise ValueError(f'Source columns {missing} do not exist in DataFrame')

    convert_exprs = []
    unnest_exprs = []
    for src_col in src_cols:
        src_dtype = schema[src_col]

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
                    name_template.format(col=src_col, index=f.name)
                    for f in fields
                ])
            case _:
                raise ValueError(f'Source column {src_col} must be of list, array or struct type to expand')

        convert_exprs.append(expr.alias(src_col))
        unnest_exprs.append(pl.col(src_col).struct.unnest())

    ret = df.with_columns(*convert_exprs).with_columns(*unnest_exprs)
    if drop_src:
        ret = ret.drop(src_cols)
    return ret


def merge_columns(
    df: AnyFrame, src_cols: Sequence[str], dest_col: str,
    drop_src: bool = False, target_dtype: Type[NumericType] = pl.Float32
) -> AnyFrame:
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
    schema = get_schema(df)
    columns = schema.names()
    if not src_cols:
        raise ValueError('src_cols must not be empty')
    if dest_col in columns:
        raise ValueError(
            f'Destination column {dest_col} already exists in DataFrame')
    if not all(col in columns for col in src_cols):
        missing = [col for col in src_cols if col not in columns]
        raise ValueError(f'Source columns {missing} do not exist in DataFrame')
    if not all(schema[col].is_numeric() for col in src_cols):
        raise ValueError('All source columns must be numeric')
    if not target_dtype.is_numeric():
        raise ValueError('target_dtype must be a numeric type')

    df = df.with_columns(
        pl.concat_list(pl.col(src_cols).cast(target_dtype)).cast(
            pl.List(target_dtype)
        ).alias(dest_col)
    )
    if drop_src:
        df = df.drop(src_cols)
    return df
