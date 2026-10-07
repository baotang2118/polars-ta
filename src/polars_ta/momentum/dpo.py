"""Detrended Price Oscillator."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window
from polars_ta.overlay.ma import _sma_expr


def _dpo_expr(values: pl.Expr, window: int) -> pl.Expr:
    # The average is centred by reaching back half a window plus one bar.
    return values.shift(window // 2 + 1) - _sma_expr(values, window)


@overload
def dpo(column: str | pl.Expr, window: int = 20) -> pl.Expr: ...


@overload
def dpo(column: pl.Series, window: int = 20) -> pl.Series: ...


def dpo(column: IntoColumn, window: int = 20) -> pl.Expr | pl.Series:
    """Detrended Price Oscillator: price against a centred moving average.

    Shifting the comparison back half a window removes the trend rather than
    lagging behind it, which leaves the shorter cycles that the trend was
    hiding. The centring reads a past bar, so the line is not meant for
    real-time signals.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods in the average.

    Returns:
        A difference in price units: a ``pl.Series`` when ``column`` is a
        series, otherwise a ``pl.Expr``. The first
        ``max(window - 1, window // 2 + 1)`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(column, lambda values: _dpo_expr(values, window))
