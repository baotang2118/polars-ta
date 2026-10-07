"""Vortex Indicator."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window
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


def vortex(
    high: IntoColumn, low: IntoColumn, close: IntoColumn, window: int = 14
) -> pl.Expr:
    """Vortex Indicator: upward and downward movement, each against true range.

    ``plus`` measures the distance from the previous low up to today's high and
    ``minus`` the distance from the previous high down to today's low, so the
    lines cross when one direction starts covering more ground than the other.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods summed.

    Returns:
        A ``pl.Expr`` yielding a struct with non-negative fields ``plus`` and
        ``minus``. The first ``window`` rows are null, and a window with no
        range reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _vortex_expr(*to_exprs(high, low, close), window)
