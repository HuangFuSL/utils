'''
`utils.cpolars.preprocess` - Functions for preprocessing features.
'''
import math
from typing import Dict, List, Literal, Sequence, Tuple, overload

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
            ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).fill_nan(0).alias(col)
            for col in cols
        ])
    elif method == 'z-score':
        return df.with_columns([
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).fill_nan(0).alias(col)
            for col in cols
        ])
    else:
        raise ValueError("Method must be either 'min-max' or 'z-score'")

def cut_by(
    df: AnyFrame, cutpoints: Dict[str, List[float | int]],
    dest_cols: Sequence[str] | None = None,
    side: Literal['left', 'right'] = 'right'
) -> AnyFrame:
    '''
    Cut specified columns in a Polars DataFrame into buckets defined by cutpoints.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
        cutpoints (Dict[str, List[float | int]]): A dictionary mapping column names to their respective cutpoints.
        cols (Sequence[str]): The columns to cut, the column must be present in both ``df`` and ``cutpoints``.
        dest_cols (Sequence[str] | None): The destination column names for the cut columns. If None, original columns will be replaced.
        side (Literal['left', 'right']): The side to assign values that are exactly on the cutpoint. 'left' means the value will be assigned to the left bucket, while 'right' means the value will be assigned to the right bucket.

    Returns:
        pl.DataFrame | pl.LazyFrame: The DataFrame with cut columns.
    '''
    col_names = list(cutpoints.keys())
    if dest_cols is not None:
        if len(dest_cols) != len(cutpoints):
            raise ValueError('Length of dest_cols must match length of cols')
    else:
        dest_cols = list(col_names)
    df_schema = get_schema(df)
    for col in cutpoints:
        if col not in df_schema:
            raise ValueError(f'Column {col} does not exist in DataFrame')

    return df.with_columns(*[
        pl.lit(pl.Series(cutpoints[col], dtype=pl.Float64))
            .sort()
            .search_sorted(pl.col(col), side=side)
            .alias(dest_col)
        for col, dest_col in zip(cutpoints, dest_cols)
    ])


@overload
def width_bucketize(
    df: AnyFrame, cols: Sequence[str], n_buckets: int,
    dest_cols: Sequence[str] | None = None,
    return_cutpoints: bool = False,
    side: Literal['left', 'right'] = 'right'
) -> Tuple[AnyFrame, None]:
    ...


@overload
def width_bucketize(
    df: AnyFrame, cols: Sequence[str], n_buckets: int,
    dest_cols: Sequence[str] | None = None,
    return_cutpoints: bool = True,
    side: Literal['left', 'right'] = 'right'
) -> Tuple[AnyFrame, Dict[str, List[float]]]:
    ...


def width_bucketize(
    df: AnyFrame, cols: Sequence[str], n_buckets: int,
    dest_cols: Sequence[str] | None = None,
    return_cutpoints: bool = False,
    side: Literal['left', 'right'] = 'right'
) -> Tuple[AnyFrame, Dict[str, List[float]] | None]:
    '''
    Bucketize specified columns in a Polars DataFrame into equal-width buckets.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
        cols (Sequence[str]): The columns to bucketize.
        n_buckets (int): The number of buckets to create.
        dest_cols (Sequence[str] | None): The destination column names for the bucketized columns. If None, original columns will be replaced.
        return_cutpoints (bool): Whether to return the cutpoints used for bucketization. If `df` is LazyFrame, cutpoints requires df.collect() and may cause performance issues.
        side (Literal['left', 'right']): The side to assign values that are exactly on the cutpoint. 'left' means the value will be assigned to the left bucket, while 'right' means the value will be assigned to the right bucket.

    Returns:
        Tuple[pl.DataFrame | pl.LazyFrame, Dict[str, List[float]] | None]: The DataFrame with bucketized columns, and optionally the cutpoints used for bucketization.
    '''
    if dest_cols is not None:
        if len(dest_cols) != len(cols):
            raise ValueError('Length of dest_cols must match length of cols')
    else:
        dest_cols = cols
    if n_buckets <= 0:
        raise ValueError('n_buckets must be a positive integer')

    cut_points = {}
    if not return_cutpoints:
        # Shift cutpoints by 1 ULP to compensate for floating-point drift
        # in normalize's (x - min) / (max - min) vs literal k/n comparison.
        direction = 1.0 if side == 'left' else -1.0
        cut_points = {
            col: [math.nextafter(k / n_buckets, direction) for k in range(1, n_buckets)]
            for col in dest_cols
        }
        df = normalize(df.with_columns(*[
            pl.col(col).alias(dest_col)
            for col, dest_col in zip(cols, dest_cols)
        ]), cols=dest_cols, method='min-max')
    else:
        data_range_df = pl.collect_all([
            df.lazy().select([
                pl.col(col).min().alias('min'),
                pl.col(col).max().alias('max')
            ])
            for col in cols
        ])
        data_range = [_.to_dicts()[0] for _ in data_range_df]
        for i, col in enumerate(cols):
            min_val, max_val = data_range[i]['min'], data_range[i]['max']
            if min_val is None or max_val is None or min_val == max_val:
                cut_points[col] = []
            else:
                cut_points[col] = [
                    min_val + (max_val - min_val) * j / n_buckets
                    for j in range(1, n_buckets)
                ]
    return cut_by(
        df, cutpoints=cut_points, dest_cols=dest_cols, side=side
    ), None if not return_cutpoints else cut_points

@overload
def quantile_bucketize(
    df: AnyFrame, cols: Sequence[str], n_buckets: int,
    dest_cols: Sequence[str] | None = None,
    return_cutpoints: bool = False,
    side: Literal['left', 'right'] = 'right'
) -> Tuple[AnyFrame, None]:
    ...


@overload
def quantile_bucketize(
    df: AnyFrame, cols: Sequence[str], n_buckets: int,
    dest_cols: Sequence[str] | None = None,
    return_cutpoints: bool = True,
    side: Literal['left', 'right'] = 'right'
) -> Tuple[AnyFrame, Dict[str, List[float]]]:
    ...


def quantile_bucketize(
    df: AnyFrame, cols: Sequence[str], n_buckets: int,
    dest_cols: Sequence[str] | None = None,
    return_cutpoints: bool = False,
    side: Literal['left', 'right'] = 'right'
) -> Tuple[AnyFrame, Dict[str, List[float]] | None]:
    '''
    Bucketize specified columns in a Polars DataFrame into quantile-based buckets.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
        cols (Sequence[str]): The columns to bucketize.
        n_buckets (int): The number of buckets to create.
        dest_cols (Sequence[str] | None): The destination column names for the bucketized columns. If None, original columns will be replaced.
        return_cutpoints (bool): Whether to return the cutpoints used for bucketization. If `df` is LazyFrame, cutpoints requires df.collect() and may cause performance issues.
        side (Literal['left', 'right']): The side to assign values that are exactly on the cutpoint.

    Returns:
        Tuple[pl.DataFrame | pl.LazyFrame, Dict[str, List[float]] | None]: The DataFrame with bucketized columns, and optionally the cutpoints used for bucketization.
    '''
    if dest_cols is not None:
        if len(dest_cols) != len(cols):
            raise ValueError('Length of dest_cols must match length of cols')
    else:
        dest_cols = cols
    if n_buckets <= 0:
        raise ValueError('n_buckets must be a positive integer')

    quantile_tie = 'lower' if side == 'left' else 'higher'
    qs = [(i + 1) / n_buckets for i in range(n_buckets - 1)]

    if return_cutpoints:
        cut_points = df.lazy().select([
            pl.col(col).quantile(qs, interpolation=quantile_tie).alias(col)
            for col in cols
        ]).collect().to_dicts()[0]
        return cut_by(
            df, cutpoints=cut_points, dest_cols=dest_cols, side=side
        ), cut_points
    else:
        return df.with_columns([
            pl.col(col).quantile(qs, interpolation=quantile_tie)
            .explode()
            .search_sorted(pl.col(col), side=side)
            .alias(dest_col)
            for col, dest_col in zip(cols, dest_cols)
        ]), None

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
