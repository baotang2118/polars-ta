"""Volume Weighted Average Price."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window
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


def vwap(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    volume: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Volume Weighted Average Price over a rolling window.

    Each bar's typical price is weighted by its volume, so the line sits where
    most of the trade actually happened rather than where the midpoints were.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        volume: Column name or expression of traded volume.
        window: Number of periods averaged.

    Returns:
        A ``pl.Expr`` yielding a price. The first ``window - 1`` rows are null,
        as is any window that traded no volume.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _vwap_expr(*to_exprs(high, low, close, volume), window)
