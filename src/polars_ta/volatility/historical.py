"""Volatility read from range expansion and from the dispersion of returns."""

from __future__ import annotations

import math

import polars as pl

from polars_ta._common import (
    IntoColumn,
    to_expr,
    to_exprs,
    validate_positive,
    validate_window,
)
from polars_ta.overlay.ma import _ema_expr

TRADING_DAYS_PER_YEAR = 252.0
"""Sessions in a year, the usual annualisation factor for daily bars."""


def _cvi_expr(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    smoothed = _ema_expr(high - low, window, 2.0 / (window + 1.0), "talib")
    reference = smoothed.shift(window)
    return (
        pl.when(smoothed.is_null() | reference.is_null())
        .then(None)
        .when(reference != 0.0)
        .then((smoothed - reference) / reference * 100.0)
        .otherwise(0.0)
    )


def _annualized_volatility_expr(
    values: pl.Expr, window: int, periods_per_year: float
) -> pl.Expr:
    previous = values.shift(1)
    returns = (
        pl.when(previous.is_null() | values.is_null())
        .then(None)
        .when(previous != 0.0)
        .then(values / previous - 1.0)
        .otherwise(0.0)
    )
    deviation = returns.rolling_std(window_size=window, min_samples=window, ddof=0)
    return deviation * math.sqrt(periods_per_year)


def cvi(high: IntoColumn, low: IntoColumn, window: int = 10) -> pl.Expr:
    """Chaikin's Volatility: how much the smoothed bar range has widened.

    Chaikin read a widening range as a market approaching a top and a
    narrowing one as a base forming, so the reading is a rate of change of the
    high-low spread rather than of price itself.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        window: Number of periods in the exponential average, and the lookback
            the average is compared against.

    Returns:
        A ``pl.Expr`` yielding a percentage. The first ``2 * window - 1`` rows
        are null, and a zero reference average reports ``0.0`` rather than
        dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    high_expr, low_expr = to_exprs(high, low)
    return _cvi_expr(high_expr, low_expr, window)


def annualized_volatility(
    column: IntoColumn,
    window: int = 20,
    *,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pl.Expr:
    """Annualised historical volatility: the dispersion of recent returns, scaled up.

    The standard deviation of the last ``window`` simple returns is multiplied
    by ``sqrt(periods_per_year)``, which restates a per-bar figure on the
    annual scale option pricing quotes. Set ``periods_per_year`` to match the
    bar size: ``252`` for daily sessions, ``52`` for weekly, ``12`` for monthly.

    Args:
        column: Column name or expression holding the input values.
        window: Number of returns in the standard deviation.
        periods_per_year: Bars in a year, used as the annualisation factor.

    Returns:
        A ``pl.Expr`` yielding a fraction, so ``0.25`` means 25% a year. The
        first ``window`` rows are null: one for the return and ``window - 1``
        for the deviation. The deviation is the population figure, and a zero
        reference price reports a ``0.0`` return rather than dividing.

    Raises:
        ValueError: If ``window`` is invalid or ``periods_per_year`` is not a
            positive finite number.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    validate_positive("periods_per_year", periods_per_year)
    return _annualized_volatility_expr(to_expr(column), window, float(periods_per_year))
