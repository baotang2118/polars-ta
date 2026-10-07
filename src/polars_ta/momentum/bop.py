"""Balance of Power."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs


def _bop_expr(open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    span = high - low
    return (
        pl.when(span.is_null() | close.is_null() | open_.is_null())
        .then(None)
        .when(span > 0.0)
        .then((close - open_) / span)
        .otherwise(0.0)
    )


def bop(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Balance of Power: the share of each bar's range the close gained.

    A single-bar measure with no lookback at all: ``+1`` means the bar opened
    at its low and closed at its high, ``-1`` the reverse.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding a value in ``[-1, 1]``. There is no warm-up,
        and a bar with no range reports ``0.0``.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    return _bop_expr(*to_exprs(open_, high, low, close))
