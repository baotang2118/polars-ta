"""Commodity Channel Index."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window

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


@overload
def cci(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    window: int = 14,
) -> pl.Expr: ...


@overload
def cci(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    window: int = 14,
) -> pl.Series: ...


def cci(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr | pl.Series:
    """Commodity Channel Index: typical price distance from its own mean.

    The distance is scaled by the mean absolute deviation over the same window
    and by Lambert's constant ``0.015``, matching TA-Lib's ``CCI``.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        window: Number of periods for the mean and the deviation.

    Returns:
        An unbounded oscillator centred on zero: a ``pl.Series`` when every
        input is a series, otherwise a ``pl.Expr``. The first ``window - 1``
        rows are null, as is any row whose window contains a null input. A
        window with zero deviation reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _cci_expr(h, low_, c, window),
    )
