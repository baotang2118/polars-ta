"""Stochastic oscillator and its relatives."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import (
    IntoColumn,
    apply_to_column,
    apply_to_columns,
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


def _willr_expr(
    high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int
) -> pl.Expr:
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



@overload
def stoch(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pl.Expr: ...


@overload
def stoch(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pl.Series: ...


def stoch(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pl.Expr | pl.Series:
    """Stochastic oscillator: where the close sits within its recent range.

    Returns the slow lines, as TA-Lib's ``STOCH`` does. Raw fast %K is smoothed
    once into %K and again into the %D signal line, both with simple moving
    averages.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        fastk_period: Lookback for the high/low range of raw fast %K.
        slowk_period: Smoothing applied to fast %K to give %K.
        slowd_period: Smoothing applied to %K to give the %D signal line.

    Returns:
        A struct with fields ``k`` and ``d``, each in ``[0, 100]``: a
        ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``(fastk_period - 1) + (slowk_period - 1) + (slowd_period - 1)``
        rows are null. A bar whose range is flat contributes a raw %K of ``0``.

    Raises:
        ValueError: If any period is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(fastk_period)
    validate_window(slowk_period)
    validate_window(slowd_period)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _stoch_expr(
            h, low_, c, fastk_period, slowk_period, slowd_period
        ),
    )


@overload
def stochf(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Expr: ...


@overload
def stochf(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Series: ...


def stochf(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Expr | pl.Series:
    """Fast stochastic: raw %K with a single smoothing pass for %D.

    :func:`stoch` smooths raw %K once more before reporting it; this is the
    unsmoothed pair, which reacts sooner and is correspondingly noisier.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        fastk_period: Lookback for the high/low range of raw fast %K.
        fastd_period: Smoothing applied to fast %K to give fast %D.

    Returns:
        A struct with fields ``fast_k`` and ``fast_d``, each in ``[0, 100]``: a
        ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``(fastk_period - 1) + (fastd_period - 1)`` rows are null.

    Raises:
        ValueError: If any period is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(fastk_period)
    validate_window(fastd_period)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _stochf_expr(
            _fast_k_expr(h, low_, c, fastk_period), fastd_period
        ),
    )


@overload
def stochrsi(
    column: str | pl.Expr,
    window: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Expr: ...


@overload
def stochrsi(
    column: pl.Series,
    window: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Series: ...


def stochrsi(
    column: IntoColumn,
    window: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
) -> pl.Expr | pl.Series:
    """Stochastic RSI: the fast stochastic applied to the RSI instead of price.

    RSI rarely reaches its own extremes, so reading where it sits inside its
    recent range produces a far more sensitive overbought/oversold signal than
    the RSI level alone.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods for the underlying RSI.
        fastk_period: Lookback for the RSI range of raw fast %K.
        fastd_period: Smoothing applied to fast %K to give fast %D.

    Returns:
        A struct with fields ``fast_k`` and ``fast_d``, each in ``[0, 100]``: a
        ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``. The
        first ``window + (fastk_period - 1) + (fastd_period - 1)`` rows are null.

    Raises:
        ValueError: If any period is not an integer of at least 1.
    """
    validate_window(window)
    validate_window(fastk_period)
    validate_window(fastd_period)

    def build(values: pl.Expr) -> pl.Expr:
        strength = _rsi_expr(values, window)
        return _stochf_expr(
            _fast_k_expr(strength, strength, strength, fastk_period), fastd_period
        )

    return apply_to_column(column, build)


@overload
def willr(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    window: int = 14,
) -> pl.Expr: ...


@overload
def willr(
    high: pl.Series, low: pl.Series, close: pl.Series, window: int = 14
) -> pl.Series: ...


def willr(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr | pl.Series:
    """Williams %R: how far the close sits below the recent high.

    The mirror image of raw fast %K, reported on a ``[-100, 0]`` scale where
    ``0`` means the close finished at the top of its range.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        window: Number of periods in the high/low range.

    Returns:
        A value in ``[-100, 0]``: a ``pl.Series`` when every input is a series,
        otherwise a ``pl.Expr``. The first ``window - 1`` rows are null, and a
        flat range reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low, close), lambda h, low_, c: _willr_expr(h, low_, c, window)
    )
