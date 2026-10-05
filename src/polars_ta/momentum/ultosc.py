"""Ultimate Oscillator."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window

_WEIGHTS = (4.0, 2.0, 1.0)
"""Weight given to the short, medium, and long averages; they sum to seven."""


def _ultosc_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    periods: tuple[int, int, int],
) -> pl.Expr:
    previous_close = close.shift(1)
    known = high.is_not_null() & low.is_not_null() & previous_close.is_not_null()
    # Buying pressure is measured against the lower of today's low and the
    # previous close, so a gap counts as part of the bar's range.
    floor = pl.when(known).then(pl.min_horizontal(low, previous_close)).otherwise(None)
    ceiling = (
        pl.when(known).then(pl.max_horizontal(high, previous_close)).otherwise(None)
    )
    buying = pl.when(known & close.is_not_null()).then(close - floor).otherwise(None)
    ranges = ceiling - floor

    total = None
    for weight, window in zip(_WEIGHTS, periods):
        pressure = buying.rolling_sum(window_size=window, min_samples=window)
        span = ranges.rolling_sum(window_size=window, min_samples=window)
        term = (
            pl.when(span.is_null() | pressure.is_null())
            .then(None)
            .when(span > 0.0)
            .then(weight * pressure / span)
            .otherwise(0.0)
        )
        total = term if total is None else total + term
    return 100.0 * total / sum(_WEIGHTS)


@overload
def ultosc(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    short_period: int = 7,
    medium_period: int = 14,
    long_period: int = 28,
) -> pl.Expr: ...


@overload
def ultosc(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    short_period: int = 7,
    medium_period: int = 14,
    long_period: int = 28,
) -> pl.Series: ...


def ultosc(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    short_period: int = 7,
    medium_period: int = 14,
    long_period: int = 28,
) -> pl.Expr | pl.Series:
    """Ultimate Oscillator: buying pressure blended across three time frames.

    Williams combined three lookbacks precisely to avoid the false divergences
    that a single-period oscillator produces, weighting them ``4 : 2 : 1`` so
    the shortest still dominates.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        short_period: Shortest lookback, given weight four.
        medium_period: Middle lookback, given weight two.
        long_period: Longest lookback, given weight one.

    Returns:
        A value in ``[0, 100]``: a ``pl.Series`` when every input is a series,
        otherwise a ``pl.Expr``. The first ``max(periods)`` rows are null, and
        a term whose range summed to zero contributes nothing.

    Raises:
        ValueError: If any period is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    periods = (short_period, medium_period, long_period)
    for window in periods:
        validate_window(window)
    return apply_to_columns(
        (high, low, close), lambda h, low_, c: _ultosc_expr(h, low_, c, periods)
    )
