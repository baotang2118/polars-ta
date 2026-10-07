"""Detrended Price Oscillator."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window
from polars_ta.overlay.ma import _sma_expr


def _dpo_expr(values: pl.Expr, window: int) -> pl.Expr:
    # The average is centred by reaching back half a window plus one bar.
    return values.shift(window // 2 + 1) - _sma_expr(values, window)


def dpo(column: IntoColumn, window: int = 20) -> pl.Expr:
    """Detrended Price Oscillator: price against a centred moving average.

    Shifting the comparison back half a window removes the trend rather than
    lagging behind it, which leaves the shorter cycles that the trend was
    hiding. The centring reads a past bar, so the line is not meant for
    real-time signals.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the average.

    Returns:
        A ``pl.Expr`` yielding a difference in price units. The first
        ``max(window - 1, window // 2 + 1)`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _dpo_expr(to_expr(column), window)
