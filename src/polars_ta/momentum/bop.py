"""Balance of Power."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns


def _bop_expr(
    open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr
) -> pl.Expr:
    span = high - low
    return (
        pl.when(span.is_null() | close.is_null() | open_.is_null())
        .then(None)
        .when(span > 0.0)
        .then((close - open_) / span)
        .otherwise(0.0)
    )


@overload
def bop(
    open_: str | pl.Expr,
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
) -> pl.Expr: ...


@overload
def bop(
    open_: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series
) -> pl.Series: ...


def bop(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr | pl.Series:
    """Balance of Power: the share of each bar's range the close gained.

    A single-bar measure with no lookback at all: ``+1`` means the bar opened
    at its low and closed at its high, ``-1`` the reverse.

    Args:
        open_: Column name, expression, or series of opening prices.
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.

    Returns:
        A value in ``[-1, 1]``: a ``pl.Series`` when every input is a series,
        otherwise a ``pl.Expr``. There is no warm-up, and a bar with no range
        reports ``0.0``.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((open_, high, low, close), _bop_expr)
