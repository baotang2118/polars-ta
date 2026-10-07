"""Schaff Trend Cycle."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window
from polars_ta.overlay.ma import ema


def _rescale_to_range(values: pl.Expr, window: int) -> pl.Expr:
    """Where the value sits inside its own recent range, as a percentage."""
    lowest = values.rolling_min(window_size=window, min_samples=window)
    highest = values.rolling_max(window_size=window, min_samples=window)
    span = highest - lowest
    return (
        pl.when(values.is_null() | span.is_null())
        .then(None)
        .when(span != 0.0)
        .then(100.0 * (values - lowest) / span)
        .otherwise(0.0)
    )


def _stc_expr(
    values: pl.Expr,
    fast_period: int,
    slow_period: int,
    cycle: int,
    smooth_k: int,
    smooth_d: int,
) -> pl.Expr:
    macd_line = ema(values, fast_period, mode="recursive") - ema(
        values, slow_period, mode="recursive"
    )
    first = ema(_rescale_to_range(macd_line, cycle), smooth_k, mode="recursive")
    return ema(_rescale_to_range(first, cycle), smooth_d, mode="recursive")


def stc(
    column: IntoColumn,
    fast_period: int = 23,
    slow_period: int = 50,
    cycle: int = 10,
    *,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pl.Expr:
    """Schaff Trend Cycle: a MACD line put through two stochastic passes.

    The MACD line is unbounded and slow to turn. Measuring where it sits inside
    its own recent range, twice over, bounds it to ``[0, 100]`` and makes the
    turns sharp enough to read as overbought and oversold.

    Args:
        column: Column name or expression holding the input values.
        fast_period: Period of the faster average inside the MACD line.
        slow_period: Period of the slower average inside the MACD line.
        cycle: Number of periods in both range look-backs.
        smooth_k: Period of the first smoothing pass.
        smooth_d: Period of the second smoothing pass.

    Returns:
        A ``pl.Expr`` yielding a value in ``[0, 100]``. Every average uses the
        recursive seeding convention, as the common reference implementations
        do, and a flat range reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If a period is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_window(cycle)
    validate_window(smooth_k)
    validate_window(smooth_d)
    return _stc_expr(
        to_expr(column), fast_period, slow_period, cycle, smooth_k, smooth_d
    )
