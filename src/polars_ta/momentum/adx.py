"""Average Directional Index and the Directional Indicators."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window
from polars_ta.overlay.ma import ema
from polars_ta.volatility.atr import _true_range_expr

ADX_FIELDS = ("adx", "plus_di", "minus_di")


def _adx_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    # A bar counts toward only the larger of the two moves, and only if positive.
    plus_dm = (
        pl.when(up_move.is_null())
        .then(None)
        .when((up_move > down_move) & (up_move > 0.0))
        .then(up_move)
        .otherwise(0.0)
    )
    minus_dm = (
        pl.when(down_move.is_null())
        .then(None)
        .when((down_move > up_move) & (down_move > 0.0))
        .then(down_move)
        .otherwise(0.0)
    )

    smoothing = 1.0 / window
    smoothed_range = ema(_true_range_expr(high, low, close), window, alpha=smoothing)
    smoothed_plus = ema(plus_dm, window, alpha=smoothing)
    smoothed_minus = ema(minus_dm, window, alpha=smoothing)
    # A null condition would otherwise fall through and emit 0.0, erasing warm-up.
    known = smoothed_range.is_not_null() & smoothed_plus.is_not_null()
    plus_di = (
        pl.when(~known)
        .then(None)
        .when(smoothed_range > 0.0)
        .then(100.0 * smoothed_plus / smoothed_range)
        .otherwise(0.0)
    )
    minus_di = (
        pl.when(~known)
        .then(None)
        .when(smoothed_range > 0.0)
        .then(100.0 * smoothed_minus / smoothed_range)
        .otherwise(0.0)
    )

    total = plus_di + minus_di
    directional_index = (
        pl.when(total.is_null())
        .then(None)
        .when(total > 0.0)
        .then(100.0 * (plus_di - minus_di).abs() / total)
        .otherwise(0.0)
    )
    return pl.struct(
        adx=ema(directional_index, window, alpha=smoothing),
        plus_di=plus_di,
        minus_di=minus_di,
    )


@overload
def adx(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    window: int = 14,
) -> pl.Expr: ...


@overload
def adx(
    high: pl.Series, low: pl.Series, close: pl.Series, window: int = 14
) -> pl.Series: ...


def adx(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr | pl.Series:
    """Average Directional Index with its two Directional Indicators.

    ``plus_di`` and ``minus_di`` measure the strength of upward and downward
    movement; ``adx`` smooths the normalized gap between them into a single
    trend-strength reading that ignores direction.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        window: Number of periods for all three Wilder-smoothing passes.

    Returns:
        A struct with fields ``adx``, ``plus_di``, and ``minus_di``, each in
        ``[0, 100]``: a ``pl.Series`` when every input is a series, otherwise a
        ``pl.Expr``. The indicators start after ``window`` rows and ``adx``
        after ``2 * window - 1``, matching the separate TA-Lib lookbacks.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _adx_expr(h, low_, c, window),
    )
