"""True Strength Index."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window
from polars_ta.overlay.ma import ema


def _double_ema(values: pl.Expr, fast_period: int, slow_period: int) -> pl.Expr:
    return ema(
        ema(values, slow_period, mode="recursive"), fast_period, mode="recursive"
    )


def _tsi_expr(values: pl.Expr, fast_period: int, slow_period: int) -> pl.Expr:
    change = values.diff()
    smoothed = _double_ema(change, fast_period, slow_period)
    magnitude = _double_ema(change.abs(), fast_period, slow_period)
    return (
        pl.when(smoothed.is_null() | magnitude.is_null())
        .then(None)
        .when(magnitude != 0.0)
        .then(100.0 * smoothed / magnitude)
        .otherwise(0.0)
    )


@overload
def tsi(
    column: str | pl.Expr, fast_period: int = 13, slow_period: int = 25
) -> pl.Expr: ...


@overload
def tsi(
    column: pl.Series, fast_period: int = 13, slow_period: int = 25
) -> pl.Series: ...


def tsi(
    column: IntoColumn, fast_period: int = 13, slow_period: int = 25
) -> pl.Expr | pl.Series:
    """True Strength Index: doubly smoothed momentum against its own magnitude.

    Smoothing the raw change twice strips the noise that makes momentum hard to
    read, and dividing by the same smoothing of the absolute change rescales
    the result to ``[-100, 100]`` regardless of the instrument's volatility.

    Args:
        column: Column name, expression, or series holding the input values.
        fast_period: Period of the second smoothing pass.
        slow_period: Period of the first smoothing pass.

    Returns:
        A value in ``[-100, 100]``: a ``pl.Series`` when ``column`` is a
        series, otherwise a ``pl.Expr``. The first
        ``slow_period + fast_period - 1`` rows are null. Both averages use the
        recursive seeding convention, as the common reference implementations
        do, and a window with no movement reports ``0.0``.

    Raises:
        ValueError: If a period is not an integer of at least 1.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    return apply_to_column(
        column, lambda values: _tsi_expr(values, fast_period, slow_period)
    )
