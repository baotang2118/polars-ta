"""Volume Weighted Average Price."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window
from polars_ta.overlay.transform import _typprice_expr


def _vwap_expr(
    high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr, window: int
) -> pl.Expr:
    typical = _typprice_expr(high, low, close)
    weighted = (typical * volume).rolling_sum(window_size=window, min_samples=window)
    traded = volume.rolling_sum(window_size=window, min_samples=window)
    # A window that traded nothing has no volume-weighted price, so it stays null.
    return (
        pl.when(weighted.is_null() | traded.is_null() | (traded == 0.0))
        .then(None)
        .otherwise(weighted / traded)
    )


@overload
def vwap(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    volume: str | pl.Expr,
    window: int = 14,
) -> pl.Expr: ...


@overload
def vwap(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    volume: pl.Series,
    window: int = 14,
) -> pl.Series: ...


def vwap(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    volume: IntoColumn,
    window: int = 14,
) -> pl.Expr | pl.Series:
    """Volume Weighted Average Price over a rolling window.

    Each bar's typical price is weighted by its volume, so the line sits where
    most of the trade actually happened rather than where the midpoints were.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        volume: Column name, expression, or series of traded volume.
        window: Number of periods averaged.

    Returns:
        A price: a ``pl.Series`` when every input is a series, otherwise a
        ``pl.Expr``. The first ``window - 1`` rows are null, as is any window
        that traded no volume.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low, close, volume),
        lambda h, low_, c, v: _vwap_expr(h, low_, c, v, window),
    )
