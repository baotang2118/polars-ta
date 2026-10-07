"""Mass Index."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window
from polars_ta.overlay.ma import ema


def _mass_expr(
    high: pl.Expr, low: pl.Expr, fast_period: int, slow_period: int
) -> pl.Expr:
    amplitude = high - low
    single = ema(amplitude, fast_period, mode="recursive")
    double = ema(single, fast_period, mode="recursive")
    ratio = (
        pl.when(single.is_null() | double.is_null())
        .then(None)
        .when(double != 0.0)
        .then(single / double)
        .otherwise(0.0)
    )
    return ratio.rolling_sum(window_size=slow_period, min_samples=slow_period)


@overload
def mass(
    high: str | pl.Expr,
    low: str | pl.Expr,
    fast_period: int = 9,
    slow_period: int = 25,
) -> pl.Expr: ...


@overload
def mass(
    high: pl.Series, low: pl.Series, fast_period: int = 9, slow_period: int = 25
) -> pl.Series: ...


def mass(
    high: IntoColumn,
    low: IntoColumn,
    fast_period: int = 9,
    slow_period: int = 25,
) -> pl.Expr | pl.Series:
    """Mass Index: a sum of how fast the trading range is widening.

    The ratio of a smoothed range to a doubly smoothed one rises above one when
    bars start spanning more than they recently did. Summing it looks for a
    range bulge, which often precedes a reversal, without caring which way
    price is going.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        fast_period: Number of periods in both exponential averages.
        slow_period: Number of periods the ratio is summed over.

    Returns:
        A value near ``slow_period``: a ``pl.Series`` when every input is a
        series, otherwise a ``pl.Expr``. The first
        ``2 * (fast_period - 1) + slow_period - 1`` rows are null. Both
        averages use the recursive seeding convention, as the common reference
        implementations do.

    Raises:
        ValueError: If a period is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    return apply_to_columns(
        (high, low), lambda h, low_: _mass_expr(h, low_, fast_period, slow_period)
    )
