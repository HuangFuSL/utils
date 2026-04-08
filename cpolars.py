from typing import List, Sequence, Type

from polars.datatypes.classes import NumericType
import polars as pl

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
