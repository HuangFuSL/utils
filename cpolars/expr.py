'''
`utils.cpolars.expr` - Functions for creating and manipulating Polars expressions.
'''

from typing import Any, List

import polars as pl


def chained_when(
    whens: List[pl.Expr], thens: List[pl.Expr | Any],
    else_: pl.Expr | Any
) -> pl.Expr:
    '''
    Perform chained when-then-otherwise logic on a Polars DataFrame, returning a new column based on the specified conditions and corresponding values.

    Notice:
    - Literal values passed in `thens` and `else_` will be interpreted as constant values, rather than column names. Use `pl.col()` to specify column references if needed.
    - Expressions in `thens` and `else_` must be of compatible types and applicable to the whole column, as they will be evaluated for all rows where their corresponding conditions are met.

    Args:
        whens (List[pl.Expr]): List of conditions to evaluate.
        thens (List[pl.Expr | Any]): List of values to return for each corresponding condition in `whens`.
        else_ (pl.Expr | Any): Value to return if none of the conditions in `whens` are met.
    Returns:
        pl.Expr: A Polars expression representing the result of the chained when-then-otherwise logic.
    '''
    if len(whens) != len(thens):
        raise ValueError('Length of whens and thens must be the same')
    ensure_expr = lambda x: x if isinstance(x, pl.Expr) else pl.lit(x)
    expr = ensure_expr(else_)
    for when, then in zip(reversed(whens), reversed(thens)):
        expr = pl.when(when).then(ensure_expr(then)).otherwise(expr)
    return expr