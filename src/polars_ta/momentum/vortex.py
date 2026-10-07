"""Vortex Indicator."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window
from polars_ta.volatility.atr import _true_range_expr

VORTEX_FIELDS = ("plus", "minus")


def _vortex_ratio(movement: pl.Expr, range_sum: pl.Expr, window: int) -> pl.Expr:
    moved = movement.rolling_sum(window_size=window, min_samples=window)
    return (
        pl.when(moved.is_null() | range_sum.is_null())
        .then(None)
        .when(range_sum != 0.0)
        .then(moved / range_sum)
        .otherwise(0.0)
    )


def _vortex_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    range_sum = _true_range_expr(high, low, close).rolling_sum(
        window_size=window, min_samples=window
    )
    return pl.struct(
        plus=_vortex_ratio((high - low.shift(1)).abs(), range_sum, window),
        minus=_vortex_ratio((low - high.shift(1)).abs(), range_sum, window),
    )


@overload
def vortex(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    window: int = 14,
) -> pl.Expr: ...


@overload
def vortex(
    high: pl.Series, low: pl.Series, close: pl.Series, window: int = 14
) -> pl.Series: ...


def vortex(
    high: IntoColumn, low: IntoColumn, close: IntoColumn, window: int = 14
) -> pl.Expr | pl.Series:
    """Vortex Indicator: upward and downward movement, each against true range.

    ``plus`` measures the distance from the previous low up to today's high and
    ``minus`` the distance from the previous high down to today's low, so the
    lines cross when one direction starts covering more ground than the other.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        window: Number of periods summed.

    Returns:
        A struct with non-negative fields ``plus`` and ``minus``: a
        ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null, and a window with no range reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low, close), lambda h, low_, c: _vortex_expr(h, low_, c, window)
    )
