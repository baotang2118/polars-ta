"""True Range and Average True Range."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window
from polars_ta.overlay.ma import ema


def _true_range_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    previous_close = close.shift(1)
    # max_horizontal ignores nulls, so guard the inputs rather than relying on it.
    known = high.is_not_null() & low.is_not_null() & previous_close.is_not_null()
    return (
        pl.when(known)
        .then(
            pl.max_horizontal(
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            )
        )
        .otherwise(None)
    )


def _atr_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    return ema(_true_range_expr(high, low, close), window, alpha=1.0 / window)


@overload
def true_range(
    high: str | pl.Expr, low: str | pl.Expr, close: str | pl.Expr
) -> pl.Expr: ...


@overload
def true_range(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series: ...


def true_range(
    high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr | pl.Series:
    """True Range: the widest of today's span and either gap from yesterday's close.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.

    Returns:
        A non-negative range: a ``pl.Series`` when every input is a series,
        otherwise a ``pl.Expr``. The first row is null, since it has no
        previous close.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((high, low, close), _true_range_expr)


@overload
def atr(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    window: int = 14,
) -> pl.Expr: ...


@overload
def atr(
    high: pl.Series, low: pl.Series, close: pl.Series, window: int = 14
) -> pl.Series: ...


def atr(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr | pl.Series:
    """Average True Range: Wilder-smoothed true range, a pure volatility measure.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        window: Number of periods to smooth over.

    Returns:
        A non-negative volatility measure in price units: a ``pl.Series`` when
        every input is a series, otherwise a ``pl.Expr``. The first ``window``
        rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _atr_expr(h, low_, c, window),
    )
