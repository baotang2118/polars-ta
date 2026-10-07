"""Midpoint overlays: the centre of a rolling range."""

from __future__ import annotations

import polars as pl

from polars_ta._common import (
    IntoColumn,
    to_expr,
    to_exprs,
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


def midpoint(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Midpoint: the average of the highest and lowest value over ``window``.

    Unlike a moving average this ignores everything between the two extremes,
    so it only moves when a new extreme enters or an old one leaves.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the lookback range.

    Returns:
        A ``pl.Expr``. The first ``window - 1`` rows are null, as is any row
        whose window contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _midpoint_expr(to_expr(column), window)


def midprice(high: IntoColumn, low: IntoColumn, window: int = 14) -> pl.Expr:
    """Midpoint price: the average of the highest high and the lowest low.

    The two-column counterpart of :func:`midpoint`, taking the extremes from
    the bar highs and lows rather than from a single series.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        window: Number of periods in the lookback range.

    Returns:
        A ``pl.Expr``. The first ``window - 1`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _midprice_expr(*to_exprs(high, low), window)
