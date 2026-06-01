from typing import Sequence, TypeVar

import polars as pl

AnyFrame = TypeVar('AnyFrame', pl.DataFrame, pl.LazyFrame)
T = TypeVar('T')

Expr = pl.Expr | str

def ensure_expr(expr: Expr) -> pl.Expr:
    if isinstance(expr, str):
        return pl.col(expr)
    return expr

def ensure_exprs(exprs: Expr | Sequence[Expr]) -> Sequence[pl.Expr]:
    if isinstance(exprs, Sequence) and not isinstance(exprs, str):
        return [ensure_expr(expr) for expr in exprs]
    else:
        return [ensure_expr(exprs)]

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

def inspect_names(df: AnyFrame, exprs: Expr | Sequence[Expr]) -> Sequence[str]:
    '''
    Get the output column names of the given expressions.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame.
        exprs (Sequence[Expr]): The expressions to inspect.

    Returns:
        Sequence[str]: The output column names of the given expressions.
    '''
    return df.lazy().select(ensure_exprs(exprs)).collect_schema().names()
