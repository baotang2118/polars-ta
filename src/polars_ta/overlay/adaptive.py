"""Adaptive moving averages, whose smoothing factor varies bar by bar."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window

_RETURN_DTYPE = pl.Float64


def _kama_scan(inputs: pl.Series) -> pl.Series:
    """Run the KAMA recursion, whose smoothing factor changes every bar."""
    values = inputs.struct.field("value").to_list()
    factors = inputs.struct.field("factor").to_list()
    seeds = inputs.struct.field("seed").to_list()

    result: list[float | None] = [None] * len(values)
    previous: float | None = None
    for index in range(len(values)):
        if factors[index] is None or values[index] is None:
            previous = None
            continue
        if previous is None:
            previous = seeds[index]
            if previous is None:
                continue
        previous += factors[index] * (values[index] - previous)
        result[index] = previous
    return pl.Series(values=result, dtype=_RETURN_DTYPE)


def _kama_expr(
    values: pl.Expr, window: int, fast_period: int, slow_period: int
) -> pl.Expr:
    volatility = values.diff().abs().rolling_sum(
        window_size=window, min_samples=window
    )
    direction = values - values.shift(window)
    # TA-Lib treats a move at least as large as the path that produced it as
    # perfectly efficient, which also covers the zero-volatility case.
    efficiency = (
        pl.when(volatility.is_null() | direction.is_null())
        .then(None)
        .when((volatility <= direction) | (volatility == 0.0))
        .then(1.0)
        .otherwise((direction / volatility).abs())
    )
    fastest = 2.0 / (fast_period + 1.0)
    slowest = 2.0 / (slow_period + 1.0)
    smoothing = (efficiency * (fastest - slowest) + slowest) ** 2
    return pl.struct(
        value=values, factor=smoothing, seed=values.shift(1)
    ).map_batches(_kama_scan, return_dtype=_RETURN_DTYPE)


@overload
def kama(
    column: str | pl.Expr,
    window: int = 30,
    *,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pl.Expr: ...


@overload
def kama(
    column: pl.Series,
    window: int = 30,
    *,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pl.Series: ...


def kama(
    column: IntoColumn,
    window: int = 30,
    *,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pl.Expr | pl.Series:
    """Kaufman's Adaptive Moving Average, which speeds up in a trending market.

    Each bar's efficiency ratio compares the net move over ``window`` periods
    against the total distance travelled. A straight-line move scores ``1`` and
    smooths with the ``fast_period`` factor; a directionless one scores ``0``
    and smooths with the ``slow_period`` factor.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods in the efficiency-ratio lookback.
        fast_period: Period behind the fastest permitted smoothing factor.
        slow_period: Period behind the slowest permitted smoothing factor.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null; the recursion is seeded with the
        value immediately before the first output.

    Raises:
        ValueError: If any period is not an integer of at least 1.
    """
    validate_window(window)
    validate_window(fast_period)
    validate_window(slow_period)
    return apply_to_column(
        column, lambda values: _kama_expr(values, window, fast_period, slow_period)
    )
