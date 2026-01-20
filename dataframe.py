'''
`utils.dataframe` - Utilities for working with Polars DataFrames.
'''

import json
import os
import tempfile
from typing import Dict, List, Sequence, Tuple, Type

import polars as pl
import torch
from polars._typing import ParquetCompression
from polars.datatypes.classes import NumericType

# On certain systems, datasets must be loaded **after** torch
import datasets

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
    df: pl.DataFrame, sparse_cols: Sequence[str], starting_index: int = 1
) -> pl.DataFrame:
    '''
    Convert specified columns in a Polars DataFrame to incremental index based on unique values in the current column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        sparse_cols (Sequence[str]): The columns to convert.
        starting_index (int): The starting index for the conversion.

    Returns:
        pl.DataFrame: The DataFrame with specified columns converted to sparse representation.
    '''
    for col in sparse_cols:
        # Null and NaN check
        if df.select(pl.col(col).is_null().any()).item():
            raise ValueError(f'Column {col} contains null values')
        dtype = df.schema[col]
        if dtype == pl.Float32 or dtype == pl.Float64:
            if df.select(pl.col(col).is_nan().any()).item():
                raise ValueError(f'Column {col} contains NaN values')
        # Create mapping
        mapping = df \
            .select(pl.col(col)) \
            .unique() \
            .sort(col) \
            .with_row_index(f'{col}_index', offset=starting_index) \
            .with_columns(**{
                f'{col}_index': pl.col(f'{col}_index').cast(pl.Int64)
            })
        df = df.join(mapping, on=col, how='left')
    df = df \
        .drop(sparse_cols) \
        .rename({
            f'{col}_index': col for col in sparse_cols
        })
    return df


def save_dataset(
    df: pl.DataFrame, dest_dir: str, *,
    name_template: str = '{i}.parquet',
    split: str = 'train',
    compression: ParquetCompression = 'zstd',
    data_cols: Sequence[str] = ('features', 'label'),
    partition_cols: Sequence[str] | None = None,
) -> List[str]:
    '''
    Write a Polars DataFrame to disk as a partitioned Parquet dataset.

    Args:
        df (pl.DataFrame): The DataFrame to write.
        dest_dir (str): The destination directory to write the dataset to.
        name_template (str): The template for naming Parquet files. Must include '{i}' for the file index.
        split (str): The dataset split name (e.g., 'train', 'val', 'test').
        compression (ParquetCompression): The compression algorithm to use for Parquet files.
        data_cols (Sequence[str]): The columns to include as data in the dataset, by default ('features', 'label').
        partition_cols (Sequence[str] | None): The columns to partition the dataset by. If None, no partitioning is done.
    Returns:
        List[str]: A list of file paths to the written Parquet files.
    '''
    # Make dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    split_dir = os.path.join(dest_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir, exist_ok=True)

    # Load or init config
    if not os.path.exists(os.path.join(dest_dir, 'config.json')):
        config = {}
    else:
        with open(os.path.join(dest_dir, 'config.json'), 'r') as f:
            config = json.load(f)
    if split in config or os.listdir(split_dir):
        raise ValueError(f'Dataset split {split} already exists in {dest_dir}')

    # Sanity checks
    if not data_cols:
        raise ValueError('data_cols must not be empty')
    if len(data_cols) != len(set(data_cols)):
        raise ValueError('data_cols must be unique')
    if partition_cols is not None and len(partition_cols) != len(set(partition_cols)):
        raise ValueError('partition_cols must be unique')
    if set(data_cols) & set(partition_cols or []):
        raise ValueError('data_cols and partition_cols must be disjoint')
    if (set(data_cols) | set(partition_cols or [])) - set(df.columns):
        raise ValueError(
            'Some data_cols or partition_cols are not in the DataFrame columns')
    if '{i}' not in name_template:
        raise ValueError("name_template must include '{i}' for the file index")

    # Write dataset
    if partition_cols is None:
        partition_cols = []
    df = df.select([pl.col(col) for col in [*data_cols, *partition_cols]])
    if partition_cols:
        df.write_parquet(
            split_dir,
            compression=compression,
            use_pyarrow=True,
            pyarrow_options={
                'partition_cols': list(partition_cols),
                'basename_template': name_template,
            },
        )
    else:
        df.write_parquet(
            split_dir,
            compression=compression,
            use_pyarrow=True,
            pyarrow_options={'basename_template': name_template},
        )

    # Gather all the written files
    file_list = []
    for dirpath, dirnames, files in os.walk(split_dir):
        file_list.extend([
            os.path.relpath(os.path.join(dirpath, f), dest_dir)
            for f in files if f.endswith('.parquet')
        ])
    file_list.sort()

    # Write config
    config[split] = file_list
    temp = tempfile.NamedTemporaryFile('w', delete=False, dir=dest_dir)
    with temp:
        json.dump(config, temp)
    os.replace(temp.name, os.path.join(dest_dir, 'config.json'))
    return file_list


def load_dataset(
    src_dir: str, split: str = 'train', streaming: bool = False,
    data_cols: Sequence[str] | None = None
):
    '''
    Load a Polars DataFrame to PyTorch Dataset from a partitioned Parquet dataset on disk.

    Args:
        src_dir (str): The source directory to load the dataset from.
        split (str): The dataset split name (e.g., 'train', 'val', 'test').
        streaming (bool): Whether to load the dataset in streaming mode.

    Returns:
        datasets.Dataset: The loaded dataset.
    '''
    # Sanity checks
    if not os.path.exists(os.path.join(src_dir, 'config.json')):
        raise ValueError(f'No config.json found in {src_dir}')
    with open(os.path.join(src_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    if split not in config:
        raise ValueError(f'Dataset split {split} not found in {src_dir}')

    # Transform file paths to full paths
    config = {
        split: [os.path.join(src_dir, f) for f in config[split]]
        for split in config
    }
    dataset = datasets.load_dataset(
        'parquet', data_files=config, split=split, streaming=streaming
    )
    if data_cols is not None:
        # type: ignore
        return dataset.with_format('torch', columns=list(data_cols))
    return dataset.with_format('torch')


SCHEMA_TYPE = \
    torch.dtype | \
    Tuple[torch.dtype, bool]


def get_collate_fn(
    data_cols: Sequence[str] = ('features', 'label'),
    schema: Dict[str, SCHEMA_TYPE] | None = None,
    return_dict: bool = False
):
    '''
    Get a collate function for PyTorch DataLoader that batches data from specified columns. The dataset must yield dictionaries of PyTorch tensors.

    Args:
        data_cols (Sequence[str]): The columns to include in the batch, by default ('features', 'label').
        schema (Dict[str, Tuple[torch.dtype, bool]]): The schema defining the data types, and whether the first dimension is variable-length.
        return_dict (bool): Whether to return a dictionary of batched tensors. If False, returns a tuple.

    Returns:
        Callable: A collate function for PyTorch DataLoader.
    '''
    if schema is None:
        schema = {}
    collate_fn_dict = {}
    default = (torch.float32, False)
    for field in data_cols:
        if field in schema:
            dtype_info = schema[field]
            if isinstance(dtype_info, tuple):
                dtype, var_len = dtype_info
            else:
                dtype, var_len = dtype_info, False
        else:
            dtype, var_len = default
        if not var_len:
            collate_fn_dict[field] = (
                lambda x, dtype=dtype:
                torch.stack(x).to(dtype)
            )
        else:
            collate_fn_dict[field] = (
                lambda x, dtype=dtype:
                    torch.nn.utils.rnn.pack_sequence(
                        x, enforce_sorted=False
                    ).to(dtype)
            )

    def collate_fn(batch):
        fields = {
            field: []
            for field in data_cols
        }
        for item in batch:
            for field in data_cols:
                fields[field].append(item[field])
        if return_dict:
            batch_out = {
                field: collate_fn_dict[field](fields[field])
                for field in data_cols
            }
        else:
            batch_out = tuple(
                collate_fn_dict[field](fields[field])
                for field in data_cols
            )
        return batch_out
    return collate_fn
