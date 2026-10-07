"""Stochastic oscillator and its relatives."""

from __future__ import annotations

import polars as pl

from polars_ta._common import (
    IntoColumn,
    to_expr,
    to_exprs,
    validate_window,
)
from polars_ta.momentum.rsi import _rsi_expr

STOCH_FIELDS = ("k", "d")
STOCHF_FIELDS = ("fast_k", "fast_d")


def _fast_k_expr(
    high: pl.Expr, low: pl.Expr, close: pl.Expr, fastk_period: int
) -> pl.Expr:
    lowest = low.rolling_min(window_size=fastk_period, min_samples=fastk_period)
    highest = high.rolling_max(window_size=fastk_period, min_samples=fastk_period)
    span = highest - lowest
    return (
        pl.when(span.is_null() | close.is_null())
        .then(None)
        .when(span > 0.0)
        .then(100.0 * (close - lowest) / span)
        .otherwise(0.0)
    )


def _stoch_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    fastk_period: int,
    slowk_period: int,
    slowd_period: int,
) -> pl.Expr:
    fast_k = _fast_k_expr(high, low, close, fastk_period)
    slow_k = fast_k.rolling_mean(window_size=slowk_period, min_samples=slowk_period)
    slow_d = slow_k.rolling_mean(window_size=slowd_period, min_samples=slowd_period)
    # TA-Lib emits both lines from the same bar, so hold %K back until %D exists.
    return pl.struct(k=pl.when(slow_d.is_null()).then(None).otherwise(slow_k), d=slow_d)


def _stochf_expr(fast_k: pl.Expr, fastd_period: int) -> pl.Expr:
    fast_d = fast_k.rolling_mean(window_size=fastd_period, min_samples=fastd_period)
    return pl.struct(
        fast_k=pl.when(fast_d.is_null()).then(None).otherwise(fast_k), fast_d=fast_d
    )


def _willr_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    lowest = low.rolling_min(window_size=window, min_samples=window)
    highest = high.rolling_max(window_size=window, min_samples=window)
    span = highest - lowest
    return (
        pl.when(span.is_null() | close.is_null())
        .then(None)
        .when(span > 0.0)
        .then(-100.0 * (highest - close) / span)
        .otherwise(0.0)
    )


def stoch(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pl.Expr:
    """Stochastic oscillator: where the close sits within its recent range.

    Returns the slow lines, as TA-Lib's ``STOCH`` does. Raw fast %K is smoothed
    once into %K and again into the %D signal line, both with simple moving
    averages.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        fastk_period: Lookback for the high/low range of raw fast %K.
        slowk_period: Smoothing applied to fast %K to give %K.
        slowd_period: Smoothing applied to %K to give the %D signal line.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``k`` and ``d``, each in
        ``[0, 100]``. The first
        ``(fastk_period - 1) + (slowk_period - 1) + (slowd_period - 1)``
        rows are null. A bar whose range is flat contributes a raw %K of ``0``.

    Raises:
        ValueError: If any period is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fastk_period)
    validate_window(slowk_period)
    validate_window(slowd_period)
    high_expr, low_expr, close_expr = to_exprs(high, low, close)
    return _stoch_expr(
        high_expr, low_expr, close_expr, fastk_period, slowk_period, slowd_period
    )


def stochf(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Expr:
    """Fast stochastic: raw %K with a single smoothing pass for %D.

    :func:`stoch` smooths raw %K once more before reporting it; this is the
    unsmoothed pair, which reacts sooner and is correspondingly noisier.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        fastk_period: Lookback for the high/low range of raw fast %K.
        fastd_period: Smoothing applied to fast %K to give fast %D.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``fast_k`` and ``fast_d``,
        each in ``[0, 100]``. The first
        ``(fastk_period - 1) + (fastd_period - 1)`` rows are null.

    Raises:
        ValueError: If any period is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fastk_period)
    validate_window(fastd_period)
    high_expr, low_expr, close_expr = to_exprs(high, low, close)
    return _stochf_expr(
        _fast_k_expr(high_expr, low_expr, close_expr, fastk_period), fastd_period
    )


def stochrsi(
    column: IntoColumn,
    window: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Expr:
    """Stochastic RSI: the fast stochastic applied to the RSI instead of price.

    RSI rarely reaches its own extremes, so reading where it sits inside its
    recent range produces a far more sensitive overbought/oversold signal than
    the RSI level alone.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods for the underlying RSI.
        fastk_period: Lookback for the RSI range of raw fast %K.
        fastd_period: Smoothing applied to fast %K to give fast %D.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``fast_k`` and ``fast_d``,
        each in ``[0, 100]``. The first
        ``window + (fastk_period - 1) + (fastd_period - 1)`` rows are null.

    Raises:
        ValueError: If any period is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    validate_window(fastk_period)
    validate_window(fastd_period)
    strength = _rsi_expr(to_expr(column), window)
    return _stochf_expr(
        _fast_k_expr(strength, strength, strength, fastk_period), fastd_period
    )


def willr(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Williams %R: how far the close sits below the recent high.

    The mirror image of raw fast %K, reported on a ``[-100, 0]`` scale where
    ``0`` means the close finished at the top of its range.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods in the high/low range.

    Returns:
        A ``pl.Expr`` yielding a value in ``[-100, 0]``. The first
        ``window - 1`` rows are null, and a flat range reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    high_expr, low_expr, close_expr = to_exprs(high, low, close)
    return _willr_expr(high_expr, low_expr, close_expr, window)
