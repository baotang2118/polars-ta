"""Channel overlays: envelopes drawn around a centre line."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import (
    IntoColumn,
    apply_to_columns,
    validate_positive,
    validate_window,
)
from polars_ta.overlay.ma import EmaMode, _ema_expr, _resolve_alpha
from polars_ta.volatility.atr import _atr_expr

DONCHIAN_FIELDS = ("lower", "middle", "upper")
KELTNER_FIELDS = ("lower", "middle", "upper")


def _donchian_expr(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    upper = high.rolling_max(window_size=window, min_samples=window)
    lower = low.rolling_min(window_size=window, min_samples=window)
    return pl.struct(lower=lower, middle=(upper + lower) / 2.0, upper=upper)


def _keltner_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
    alpha: float,
    mode: EmaMode,
    atr_window: int,
    multiplier: float,
) -> pl.Expr:
    middle = _ema_expr(close, window, alpha, mode)
    offset = multiplier * _atr_expr(high, low, close, atr_window)
    return pl.struct(lower=middle - offset, middle=middle, upper=middle + offset)


@overload
def donchian(high: str | pl.Expr, low: str | pl.Expr, window: int = 20) -> pl.Expr: ...


@overload
def donchian(high: pl.Series, low: pl.Series, window: int = 20) -> pl.Series: ...


def donchian(
    high: IntoColumn, low: IntoColumn, window: int = 20
) -> pl.Expr | pl.Series:
    """Donchian Channels: the highest high and lowest low of the last ``window`` bars.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        window: Number of bars spanned by the channel.

    Returns:
        A struct with fields ``lower``, ``middle``, and ``upper``: a
        ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``window - 1`` rows are null, as is any row whose window
        contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low), lambda h, low_: _donchian_expr(h, low_, window)
    )


@overload
def keltner(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    window: int = 20,
    *,
    atr_window: int = 10,
    multiplier: float = 2.0,
    mode: EmaMode = "talib",
) -> pl.Expr: ...


@overload
def keltner(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    window: int = 20,
    *,
    atr_window: int = 10,
    multiplier: float = 2.0,
    mode: EmaMode = "talib",
) -> pl.Series: ...


def keltner(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 20,
    *,
    atr_window: int = 10,
    multiplier: float = 2.0,
    mode: EmaMode = "talib",
) -> pl.Expr | pl.Series:
    """Keltner Channels: an exponential average with an ATR envelope.

    Where Bollinger Bands scale their envelope by the standard deviation of the
    close, Keltner scales it by the Average True Range, so gaps and the full
    bar range widen the channel rather than only closing prices. That makes the
    channel steadier, and a close outside it a stronger breakout signal.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        window: Number of periods in the exponential average centre line.
        atr_window: Number of periods in the ATR setting the channel width.
        multiplier: Channel half-width in ATRs.
        mode: EMA seeding convention; see :func:`polars_ta.overlay.ma.ema`.

    Returns:
        A struct with fields ``lower``, ``middle``, and ``upper``: a
        ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        ``middle`` depends only on the close and starts after ``window - 1``
        rows; the two edges also need the ATR and so start after
        ``max(window - 1, atr_window)`` rows.

    Raises:
        ValueError: If a window, ``multiplier``, or ``mode`` is invalid.
        TypeError: If series inputs are mixed with names or expressions.
    """
    smoothing = _resolve_alpha(window, None, mode)
    validate_window(atr_window)
    validate_positive("multiplier", multiplier)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _keltner_expr(
            h, low_, c, window, smoothing, mode, atr_window, float(multiplier)
        ),
    )
