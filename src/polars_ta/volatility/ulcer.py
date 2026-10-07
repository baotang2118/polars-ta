"""Ulcer Index: downside volatility measured from the running peak."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window


def _ulcer_expr(values: pl.Expr, window: int) -> pl.Expr:
    peak = values.rolling_max(window_size=window, min_samples=window)
    drawdown = (
        pl.when(peak.is_null() | values.is_null())
        .then(None)
        .when(peak != 0.0)
        .then(100.0 * (values - peak) / peak)
        .otherwise(0.0)
    )
    squared = drawdown * drawdown
    return squared.rolling_mean(window_size=window, min_samples=window).sqrt()


@overload
def ulcer(column: str | pl.Expr, window: int = 14) -> pl.Expr: ...


@overload
def ulcer(column: pl.Series, window: int = 14) -> pl.Series: ...


def ulcer(column: IntoColumn, window: int = 14) -> pl.Expr | pl.Series:
    """Ulcer Index: the root mean square drawdown from the window's peak.

    Unlike standard deviation, which punishes upside and downside alike, this
    only accumulates while price sits below its recent high, so it measures
    how deep and how long the pain was rather than how much price moved.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods used for both the peak and the average.

    Returns:
        A non-negative percentage: a ``pl.Series`` when ``column`` is a series,
        otherwise a ``pl.Expr``. The first ``2 * (window - 1)`` rows are null,
        since the drawdown series itself needs a full window first.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(column, lambda values: _ulcer_expr(values, window))
