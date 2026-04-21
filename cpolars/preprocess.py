'''
`utils.cpolars.preprocess` - Functions for preprocessing features.
'''

import polars as pl
from typing import Sequence


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


def sparse_to_index(
    df: pl.DataFrame, sparse_cols: Sequence[str], starting_index: int = 1,
    null_index: int | None = None
) -> pl.DataFrame:
    '''
    Convert specified columns in a Polars DataFrame to incremental index based on unique values in the current column. This function is in-place.

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
