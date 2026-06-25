'''
`utils.cpolars.dataset` - Functions for saving and loading partitioned Parquet datasets on disk, and interfacing with Hugging Face Datasets and PyTorch DataLoaders.
'''

import json
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Literal, Sequence, overload
from urllib.parse import unquote
import warnings

# On certain systems, datasets must be loaded **after** torch
import datasets
import numpy as np
import polars as pl
import torch
from polars._typing import ParquetCompression


AnyDType = pl.DataType | torch.dtype | np.dtype


def _normalize_dtype_name(dtype: AnyDType) -> str:
    match dtype:
        case torch.dtype():
            ret = str(dtype).removeprefix('torch.')
        case np.dtype():
            ret = dtype.name
        case pl.DataType():
            base = dtype.base_type
            ret = base.__name__.lower()
        case _:
            raise TypeError(f'Unsupported dtype type: {type(dtype)}')
    # Post-process
    match ret:
        case 'boolean':
            return 'bool'
        case _ if all(_ not in ret for _ in ['int', 'float', 'bool']):
            raise ValueError(f'Unsupported dtype: {ret}')
    return ret

@overload
def convert_dtype(dtype: AnyDType, to: Literal['torch']) -> torch.dtype: ...

@overload
def convert_dtype(dtype: AnyDType, to: Literal['polars']) -> pl.DataType: ...

@overload
def convert_dtype(dtype: AnyDType, to: Literal['numpy']) -> np.dtype: ...

def convert_dtype(
    dtype: AnyDType,
    to: Literal['torch', 'polars', 'numpy']
) -> AnyDType:
    type_name = _normalize_dtype_name(dtype)
    match to:
        case 'torch':
            return getattr(torch, type_name)
        case 'polars':
            match type_name:
                case 'bool':
                    pl_name = 'Boolean'
                case _ if type_name.startswith('u'):
                    pl_name = 'U' + type_name[1:].capitalize()
                case _:
                    pl_name = type_name.capitalize()
            return getattr(pl, pl_name)
        case 'numpy':
            return np.dtype(type_name)
        case _:
            raise ValueError(f'Unsupported target type: {to}')


def read_tensor(path: str, col_name: str, batch_dim: int = 0) -> pl.Series:
    '''
    Read a PyTorch tensor from a file and return it as a Polars Series.

    Args:
        path (str): The file path to read the tensor from.
        col_name (str): The name of the column to create in the returned Series.
        batch_dim (int): The dimension to treat as the batch dimension when converting to a Series.


    Returns:
        pl.Series: A Polars Series containing the tensor data.
    '''
    tensor = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'The loaded object from {path} is not a PyTorch tensor')
    if batch_dim < 0 or batch_dim >= tensor.ndim:
        raise ValueError(f'batch_dim must be between 0 and {tensor.ndim - 1}, got {batch_dim}')
    if tensor.shape[batch_dim] == 0:
        raise ValueError(f'The batch dimension (dim {batch_dim}) of the tensor must have size greater than 0')
    if tensor.is_floating_point() and tensor.dtype.itemsize < 4:
        # Polars only supports float32 and float64
        tensor = tensor.to(torch.float32)

    inner_shape = list(tensor.shape)
    inner_shape.pop(batch_dim)
    if batch_dim != 0:
        tensor = torch.movedim(tensor, batch_dim, 0)
    dest_dtype = convert_dtype(tensor.dtype, 'polars')

    return pl.Series(
        col_name, tensor.numpy(), dtype=pl.Array(dest_dtype, tuple(inner_shape))
    )

def to_python_type(dtype: pl.DataType) -> str:
    while dtype.is_nested():
        dtype = dtype.inner # type: ignore
    return dtype.to_python().__name__.lower()

def save_dataset(
    df: pl.DataFrame, dest_dir: str, *,
    name_template: str = '{i}.parquet',
    split: str = 'full',
    compression: ParquetCompression = 'zstd',
    data_cols: Sequence[str] | None = None,
    pack_sequence: Sequence[str] | None = None,
    partition_cols: Sequence[str] | None = None,
    sub_splits: pl.Expr | Dict[str, pl.Expr] | None = None,
    **config_kwargs: Any
) -> None:
    '''
    Write a Polars DataFrame to disk as a partitioned Parquet dataset.

    Args:
        df (pl.DataFrame): The DataFrame to write.
        dest_dir (str): The destination directory to write the dataset to.
        name_template (str): The template for naming Parquet files. Must include '{i}' for the file index.
        split (str): The dataset split name (e.g., 'train', 'val', 'test').
        compression (ParquetCompression): The compression algorithm to use for Parquet files.
        data_cols (Sequence[str] | None): The columns to include as data in the dataset. If None, defaults to all columns except partition_cols.
        pack_sequence (Sequence[str] | None): The columns to pack as variable-length sequences. Must be a subset of data_cols. If None, no packing is done.
        partition_cols (Sequence[str] | None): The columns to partition the dataset by. If None, no partitioning is done.
        sub_splits (pl.Expr | Dict[str, pl.Expr] | None): Logical splits defined as Polars expressions over ``partition_cols``. If a dictionary is provided, the keys are used as split prefixes.
        config_kwargs (Any): Additional keyword arguments to include in the config.json file.
    '''
    # Make dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    split_dir = os.path.join(dest_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir, exist_ok=True)

    # Load or init config
    if not os.path.exists(os.path.join(dest_dir, 'config.json')):
        config = {**config_kwargs}
    else:
        with open(os.path.join(dest_dir, 'config.json'), 'r') as f:
            config = json.load(f) | config_kwargs
    if split in config.get('splits', []) or os.listdir(split_dir):
        raise ValueError(f'Dataset split {split} already exists in {dest_dir}')

    # Sanity checks
    if not len(df):
        raise ValueError('Input DataFrame is empty')
    schema = df.collect_schema()
    if data_cols is None:
        data_cols = [
            c for c in df.columns
            if c not in (partition_cols or [])
            and schema[c].base_type() is not pl.Struct
            and to_python_type(schema[c]) in {'int', 'float', 'bool'}
        ]
    for col in data_cols:
        if schema[col].base_type() is pl.Struct:
            raise ValueError(f'pl.Struct dtype in {col!r} is not supported.')
        if to_python_type(schema[col]) not in {'int', 'float', 'bool'}:
            raise ValueError(f'data_col {col!r} must be of type int, float, or bool.')
    sub_split_df = None
    if partition_cols is not None:
        if len(partition_cols) != len(set(partition_cols)):
            raise ValueError('partition_cols must be unique')
        if set(data_cols) & set(partition_cols):
            raise ValueError('data_cols and partition_cols must be disjoint')
        if (set(data_cols) | set(partition_cols)) - set(df.columns):
            raise ValueError(
                'Some data_cols or partition_cols are not in the DataFrame columns'
            )
        if sub_splits is not None:
            if not isinstance(sub_splits, dict):
                sub_splits = {'': sub_splits}
            sub_split_df = df.select(
                [pl.col(_) for _ in partition_cols]
            ).unique().with_columns(
                *[
                    pl.format(f'{col}={{{col}}}').alias(f'__path_{col}')
                    for col in partition_cols
                ],
                *[
                    pl.concat_str(pl.lit(prefix), v).alias(f'__group_{k}')
                    for k, (prefix, v) in enumerate(sub_splits.items())
                ],
            ).with_columns(
                pl.concat_str(
                    pl.lit(split), *[
                        f'__path_{k}' for k in partition_cols
                    ],
                    separator='/'
                ).alias('__sub_split_path')
            )
    else:
        if sub_splits is not None:
            raise ValueError(
                'sub_splits cannot be specified when partition_cols is None'
            )
    if '{i}' not in name_template:
        raise ValueError("name_template must include '{i}' for the file index")

    # Write dataset
    if partition_cols is None:
        partition_cols = []
    df = df.select([pl.col(col) for col in [*data_cols, *partition_cols]])
    all_files = []
    root_dir = pathlib.Path(dest_dir)
    if partition_cols:
        df.write_parquet(
            split_dir,
            compression=compression,
            use_pyarrow=True,
            pyarrow_options={
                'partition_cols': list(partition_cols),
                'basename_template': name_template,
                'file_visitor': lambda x: all_files.append(
                    str(pathlib.Path(x.path).relative_to(root_dir))
                )
            },
        )
    else:
        dest_file = pathlib.Path(split_dir) / name_template.format(i=0)
        df.write_parquet(
            dest_file,
            compression=compression,
            use_pyarrow=True,
        )
        all_files.append(str(dest_file.relative_to(root_dir)))
    all_files_df = pl.DataFrame({
        'file': sorted(all_files),
        '__sub_split_path': [
            unquote('/'.join(f.strip('/').split('/')[:-1]))
            for f in all_files
        ]
    })

    # Write config
    # Schema
    schema = df.collect_schema()
    config['schema'] = {
        col: (to_python_type(schema[col]), col in (pack_sequence or []))
        for col in data_cols
    }
    # Logical splits
    config.setdefault('splits', {})
    if sub_split_df is not None:
        sub_split_df = sub_split_df.join(
            all_files_df, on='__sub_split_path', how='inner', validate='1:m'
        )
        for k in range(len(sub_splits or [])):
            col_name = f'__group_{k}'
            col = pl.col(col_name)
            sub_split_dict = sub_split_df.group_by(col) \
                .agg(pl.col('file').explode()) \
                .filter(col.is_not_null()) \
                .to_dicts()
            for group in sub_split_dict:
                if group[col_name] in config:
                    warnings.warn(f'Overriding existing split {group[col_name]} in config')
                config['splits'][group[col_name]] = group['file']

    config['splits'][split] = all_files
    temp = tempfile.NamedTemporaryFile('w', delete=False, dir=dest_dir)
    with temp:
        json.dump(config, temp)
    os.replace(temp.name, os.path.join(dest_dir, 'config.json'))

class HfDataset(torch.utils.data.Dataset):
    '''
    PyTorch :class:`~torch.utils.data.Dataset` backed by a partitioned Parquet
    dataset on disk. Implements ``__len__`` and ``__getitem__``, delegating to
    the underlying HuggingFace Dataset. Suitable as a base class or for direct
    use with :func:`~torch.utils.data.DataLoader`.

    Subclass and override :meth:`__getitem__` for per-sample transforms
    (e.g. random cropping, augmentation) that must execute fresh on each access.

    Usage:

        class MyDataset(HfDataset):
            def __getitem__(self, idx):
                sample = super().__getitem__(idx)
                sample['features'] = my_augment(sample['features'])
                return sample

    Args:
        root_dir (str): The root directory of the dataset.
        split (str): The dataset split name (e.g., 'train', 'val', 'test').
        columns (Sequence[str] | None): Columns to include. None for all.
        float_dtype (torch.dtype): Torch dtype for float columns.
        int_dtype (torch.dtype): Torch dtype for integer columns.
    '''
    def __init__(
        self, root_dir: str, split: str = 'train', *,
        columns: Sequence[str] | None = None,
        float_dtype: torch.dtype = torch.float32,
        int_dtype: torch.dtype = torch.int64,
    ):
        self.schema = None
        self.root_dir = root_dir
        self.split = split
        self._dataset = self.load_dataset(streaming=False, columns=columns)
        self.dtype_map = {
            'bool': torch.bool,
            'int': int_dtype,
            'float': float_dtype
        }

    @property
    def columns(self) -> List[str]:
        '''
        Get the list of columns in the dataset.

        Returns:
            List[str]: The list of column names.
        '''
        return self._dataset.column_names # type: ignore

    def __len__(self) -> int:
        return len(self._dataset) # type: ignore

    def __getitem__(self, idx):
        return self._dataset[int(idx)]

    def get_dataloader(
        self, batch_size: int, shuffle: bool | None = None,
        num_workers: int = 0,
        pin_memory: bool = False, prefetch_factor: int | None = None,
        return_dict: bool = True,
        *,
        sampler: torch.utils.data.Sampler | None = None,
        weight_column: str | None = None,
    ):
        '''
        Get a PyTorch DataLoader for the dataset.

        Args:
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool | None): Whether to shuffle. Must be ``None`` when ``sampler`` or ``weight_column`` is provided.
            num_workers (int): The number of worker processes for data loading.
            pin_memory (bool): Whether to use pinned memory for data loading.
            prefetch_factor (int): The number of samples to prefetch per worker.
            return_dict (bool): Whether to return a dictionary of batched tensors. If False, returns a tuple.
            sampler (torch.utils.data.Sampler | None): Custom sampler.
                Mutually exclusive with ``shuffle`` and ``weight_column``.
            weight_column (str | None): Column name for per-sample weights.
                Creates a :class:`~torch.utils.data.WeightedRandomSampler` with
                ``replacement=True``. Mutually exclusive with ``shuffle`` and
                ``sampler``.

        Returns:
            torch.utils.data.DataLoader: The PyTorch DataLoader for the dataset.
        '''
        persistent_workers = num_workers > 0

        if sampler is None and weight_column is not None:
            weights = torch.as_tensor(
                self._dataset[weight_column], dtype=torch.float32
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, len(weights), replacement=True
            )

        if sampler is not None:
            shuffle = None

        return torch.utils.data.DataLoader(
            self, # type: ignore
            shuffle=shuffle,
            sampler=sampler,
            batch_size=batch_size, collate_fn=self.get_collate_fn(return_dict),
            num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )

    def load_dataset(
        self, streaming: bool = False, columns: Sequence[str] | None = None
    ):
        '''
        Load a Polars DataFrame to PyTorch Dataset from a partitioned Parquet dataset on disk.

        Args:
            src_dir (str): The source directory to load the dataset from.
            streaming (bool): Whether to load the dataset in streaming mode.
            columns (Sequence[str] | None): The columns to include in the dataset. If None, all columns are included.

        Returns:
            datasets.Dataset: The loaded dataset.
        '''
        # Sanity checks
        if not os.path.exists(os.path.join(self.root_dir, 'config.json')):
            raise ValueError(f'No config.json found in {self.root_dir}')
        with open(os.path.join(self.root_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        if self.split not in config.get('splits', []):
            raise ValueError(f'Dataset split {self.split} not found in {self.root_dir}')
        self.schema = config.get('schema', {})
        if columns is not None:
            self.schema = {k: self.schema[k] for k in columns}

        # Transform file paths to full paths
        config = {
            split: [os.path.join(self.root_dir, f) for f in config['splits'][split]]
            for split in config['splits']
        }
        dataset = datasets.load_dataset(
            'parquet', data_files=config, split=self.split, streaming=streaming
        )
        if self.schema is not None:
            # type: ignore
            return dataset.with_format(
                'torch', columns=list(self.schema.keys())
            )
        return dataset.with_format('torch')

    def get_collate_fn(self, return_dict: bool = False):
        '''
        Get a collate function for PyTorch DataLoader that batches data from specified columns. The dataset must yield dictionaries of PyTorch tensors.

        Args:
            data_cols (Sequence[str]): The columns to include in the batch, by default ('features', 'label').
            return_dict (bool): Whether to return a dictionary of batched tensors. If False, returns a tuple.

        Returns:
            Callable: A collate function for PyTorch DataLoader.
        '''
        if self.schema is None:
            schema = {}
            data_cols = self.columns
        else:
            schema = self.schema
            data_cols = list(self.schema.keys())

        collate_fn_dict = {}
        for field in data_cols:
            dtype, var_len = schema[field]
            torch_dtype = self.dtype_map[dtype]
            if not var_len:
                collate_fn_dict[field] = (
                    lambda x, dtype=torch_dtype: torch.stack(x).to(dtype)
                )
            else:
                collate_fn_dict[field] = (
                    lambda x, dtype=torch_dtype:
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
