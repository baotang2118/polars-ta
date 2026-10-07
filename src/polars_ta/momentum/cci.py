"""Commodity Channel Index."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window

CCI_SCALE = 0.015
"""Lambert's constant, chosen to keep roughly 70-80% of readings in [-100, 100]."""


def _cci_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    typical = (high + low + close) / 3.0
    average = typical.rolling_mean(window_size=window, min_samples=window)
    # Deviations are measured against the window's own mean, so this cannot be
    # a rolling aggregate; it expands into one shifted term per period.
    total = (typical - average).abs()
    for offset in range(1, window):
        total = total + (typical.shift(offset) - average).abs()
    deviation = total / window
    return (
        pl.when(deviation.is_null())
        .then(None)
        .when(deviation > 0.0)
        .then((typical - average) / (CCI_SCALE * deviation))
        .otherwise(0.0)
    )


def cci(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Commodity Channel Index: typical price distance from its own mean.

    The distance is scaled by the mean absolute deviation over the same window
    and by Lambert's constant ``0.015``, matching TA-Lib's ``CCI``.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods for the mean and the deviation.

    Returns:
        A ``pl.Expr`` yielding an unbounded oscillator centred on zero. The
        first ``window - 1`` rows are null, as is any row whose window contains
        a null input. A window with zero deviation reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    high_expr, low_expr, close_expr = to_exprs(high, low, close)
    return _cci_expr(high_expr, low_expr, close_expr, window)
