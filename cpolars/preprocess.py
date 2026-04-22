'''
`utils.cpolars.preprocess` - Functions for preprocessing features.
'''
from typing import Literal, Sequence

import polars as pl

from .helpers import AnyFrame, get_columns, get_schema


def add_shard_column(
    df: AnyFrame, num_shards: int, dest_col: str,
    src_col: str | Sequence[str] | None = None, seed: int | None = None
) -> AnyFrame:
    '''
    Compute and add a shard column to a Polars DataFrame for sharding data across multiple workers.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
        num_shards (int): The number of shards to create.
        dest_col (str): The name of the destination shard column.
        src_col (str | Sequence[str] | None): The source column(s) to use for assigning the shard. If None, a random shard is assigned.
        seed (int | None): The random seed for shard assignment.

    Returns:
        pl.DataFrame | pl.LazyFrame: The DataFrame with the added shard column.
    '''
    # Check column existance
    columns = get_columns(df)
    if dest_col in columns:
        raise ValueError(
            f'Destination column {dest_col} already exists in DataFrame'
        )
    if src_col is not None:
        if isinstance(src_col, str):
            src_col = [src_col]
        if not src_col:
            raise ValueError('src_col must not be empty if provided')
        if not all(col in columns for col in src_col):
            missing = [col for col in src_col if col not in columns]
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

def normalize(
    df: AnyFrame, cols: Sequence[str],
    method: Literal['min-max', 'z-score'] = 'z-score'
) -> AnyFrame:
    '''
    Normalize specified columns in a Polars DataFrame.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
        cols (Sequence[str]): The columns to normalize.
        method (Literal['min-max', 'z-score']): The normalization method to use.

    Returns:
        pl.DataFrame | pl.LazyFrame: The DataFrame with normalized columns.
    '''
    if method == 'min-max':
        return df.with_columns([
            ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(col)
            for col in cols
        ])
    elif method == 'z-score':
        return df.with_columns([
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
            for col in cols
        ])
    else:
        raise ValueError("Method must be either 'min-max' or 'z-score'")

def sparse_to_index(
    df: AnyFrame, sparse_cols: Sequence[str], starting_index: int = 1,
    null_index: int | None = None
) -> AnyFrame:
    '''
    Convert specified columns in a Polars DataFrame to incremental index based on unique values in the current column. This function replaces the original columns. If `df` is a LazyFrame, the check for null and NaN values is skipped for performance reasons.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
        sparse_cols (Sequence[str]): The columns to convert.
        starting_index (int): The starting index for the conversion.
        null_index (int | None): The index to assign to null values. If None, null values will raise an error.

    Returns:
        pl.DataFrame | pl.LazyFrame: The DataFrame with specified columns converted to sparse representation.
    '''
    columns = get_columns(df)
    if null_index is not None and null_index >= starting_index:
        raise ValueError(
            f'null_index {null_index} must be less than starting_index {starting_index}'
        )
    if any(col not in columns for col in sparse_cols):
        missing = [col for col in sparse_cols if col not in columns]
        raise ValueError(f'Sparse columns {missing} do not exist in DataFrame')
    if any(f'__sparse_to_index_{col}__' in columns for col in sparse_cols):
        existing = [
            col for col in sparse_cols
            if f'__sparse_to_index_{col}__' in columns
        ]
        raise ValueError(
            f'Destination columns {[f"__sparse_to_index_{col}__" for col in existing]} already exist in DataFrame'
        )

    for col in sparse_cols:
        # Null and NaN check
        # To guarantee performance, check is skipped for LazyFrames
        if isinstance(df, pl.DataFrame):
            if null_index is None:
                nulls = df.select(pl.col(col).is_null().any())
                if nulls.item():
                    raise ValueError(f'Column {col} contains null values')

            dtype = df.schema[col]
            if dtype == pl.Float32 or dtype == pl.Float64:
                nans = df.select(pl.col(col).is_nan().any())
                if nans.item():
                    raise ValueError(f'Column {col} contains NaN values')
        # Create mapping
        mapping = df \
            .select(pl.col(col)) \
            .filter(pl.col(col).is_not_null()) \
            .unique() \
            .sort(col) \
            .with_row_index(
                f'__sparse_to_index_{col}__', offset=starting_index
            ) \
            .with_columns(**{
                f'__sparse_to_index_{col}__': \
                    pl.col(f'__sparse_to_index_{col}__').cast(pl.Int64)
            })
        df = df.join(mapping, on=col, how='left') # type: ignore
        if null_index is not None:
            df = df.with_columns(
                pl.col(f'__sparse_to_index_{col}__').fill_null(null_index)
            )

    df = df \
        .drop(sparse_cols) \
        .rename({
            f'__sparse_to_index_{col}__': col for col in sparse_cols
        })
    return df
