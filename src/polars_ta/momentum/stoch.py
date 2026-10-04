"""Stochastic oscillator."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window

STOCH_FIELDS = ("k", "d")


def _stoch_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    fastk_period: int,
    slowk_period: int,
    slowd_period: int,
) -> pl.Expr:
    lowest = low.rolling_min(window_size=fastk_period, min_samples=fastk_period)
    highest = high.rolling_max(window_size=fastk_period, min_samples=fastk_period)
    span = highest - lowest
    fast_k = (
        pl.when(span.is_null() | close.is_null())
        .then(None)
        .when(span > 0.0)
        .then(100.0 * (close - lowest) / span)
        .otherwise(0.0)
    )
    slow_k = fast_k.rolling_mean(window_size=slowk_period, min_samples=slowk_period)
    slow_d = slow_k.rolling_mean(window_size=slowd_period, min_samples=slowd_period)
    # TA-Lib emits both lines from the same bar, so hold %K back until %D exists.
    return pl.struct(k=pl.when(slow_d.is_null()).then(None).otherwise(slow_k), d=slow_d)


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
