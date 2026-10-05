"""Midpoint overlays: the centre of a rolling range."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import (
    IntoColumn,
    apply_to_column,
    apply_to_columns,
    validate_window,
)


def _midpoint_expr(values: pl.Expr, window: int) -> pl.Expr:
    highest = values.rolling_max(window_size=window, min_samples=window)
    lowest = values.rolling_min(window_size=window, min_samples=window)
    return (highest + lowest) / 2.0


def _midprice_expr(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    highest = high.rolling_max(window_size=window, min_samples=window)
    lowest = low.rolling_min(window_size=window, min_samples=window)
    return (highest + lowest) / 2.0


@overload
def midpoint(column: str | pl.Expr, window: int = 14) -> pl.Expr: ...


@overload
def midpoint(column: pl.Series, window: int = 14) -> pl.Series: ...


def midpoint(column: IntoColumn, window: int = 14) -> pl.Expr | pl.Series:
    """Midpoint: the average of the highest and lowest value over ``window``.

    Unlike a moving average this ignores everything between the two extremes,
    so it only moves when a new extreme enters or an old one leaves.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods in the lookback range.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window - 1`` rows are null, as is any row whose window
        contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(column, lambda values: _midpoint_expr(values, window))


@overload
def midprice(high: str | pl.Expr, low: str | pl.Expr, window: int = 14) -> pl.Expr: ...


@overload
def midprice(high: pl.Series, low: pl.Series, window: int = 14) -> pl.Series: ...


def midprice(
    high: IntoColumn, low: IntoColumn, window: int = 14
) -> pl.Expr | pl.Series:
    """Midpoint price: the average of the highest high and the lowest low.

    The two-column counterpart of :func:`midpoint`, taking the extremes from
    the bar highs and lows rather than from a single series.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        window: Number of periods in the lookback range.

    Returns:
        A ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``window - 1`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low), lambda h, low_: _midprice_expr(h, low_, window)
    )
