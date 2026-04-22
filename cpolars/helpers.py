from typing import Sequence, TypeVar

import polars as pl

AnyFrame = TypeVar('AnyFrame', pl.DataFrame, pl.LazyFrame)

def get_columns(df: AnyFrame) -> Sequence[str]:
    if isinstance(df, pl.LazyFrame):
        columns = df.collect_schema().names()
    else:
        columns = df.columns
    return columns

def get_schema(df: AnyFrame) -> pl.Schema:
    if isinstance(df, pl.LazyFrame):
        schema = df.collect_schema()
    else:
        schema = df.schema
    return schema