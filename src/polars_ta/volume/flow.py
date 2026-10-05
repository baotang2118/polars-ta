"""Accumulation/Distribution and On Balance Volume."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window
from polars_ta.overlay.ma import ema


def _ad_expr(
    high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr
) -> pl.Expr:
    span = high - low
    # A bar with no range contributes nothing rather than dividing by zero.
    flow = (
        pl.when(span.is_null() | close.is_null() | volume.is_null())
        .then(None)
        .when(span > 0.0)
        .then(((close - low) - (high - close)) / span * volume)
        .otherwise(0.0)
    )
    return flow.cum_sum()


def _adosc_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    fast_period: int,
    slow_period: int,
) -> pl.Expr:
    line = _ad_expr(high, low, close, volume)
    # TA-Lib seeds both averages with the first A/D reading, not with an SMA.
    fast = ema(line, fast_period, mode="recursive")
    slow = ema(line, slow_period, mode="recursive")
    return fast - slow


def _obv_expr(close: pl.Expr, volume: pl.Expr) -> pl.Expr:
    change = close.diff()
    signed = (
        pl.when(close.is_null() | volume.is_null())
        .then(None)
        .when(change > 0.0)
        .then(volume)
        .when(change < 0.0)
        .then(-volume)
        .otherwise(0.0)
    )
    # The first bar seeds the running total with its own volume.
    return (
        pl.when(pl.int_range(pl.len()) == 0).then(volume).otherwise(signed)
    ).cum_sum()


@overload
def ad(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    volume: str | pl.Expr,
) -> pl.Expr: ...


@overload
def ad(
    high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series
) -> pl.Series: ...


def ad(
    high: IntoColumn, low: IntoColumn, close: IntoColumn, volume: IntoColumn
) -> pl.Expr | pl.Series:
    """Chaikin Accumulation/Distribution Line.

    Each bar's volume is signed by where the close finished inside the bar's
    range, then accumulated. A close at the high counts the full volume as
    accumulation, a close at the low the full volume as distribution.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        volume: Column name, expression, or series of traded volume.

    Returns:
        A running total in volume units: a ``pl.Series`` when every input is a
        series, otherwise a ``pl.Expr``. There is no warm-up, and a bar with no
        range contributes nothing.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((high, low, close, volume), _ad_expr)


@overload
def adosc(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    volume: str | pl.Expr,
    fast_period: int = 3,
    slow_period: int = 10,
) -> pl.Expr: ...


@overload
def adosc(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    volume: pl.Series,
    fast_period: int = 3,
    slow_period: int = 10,
) -> pl.Series: ...


def adosc(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    volume: IntoColumn,
    fast_period: int = 3,
    slow_period: int = 10,
) -> pl.Expr | pl.Series:
    """Chaikin A/D Oscillator: a MACD built on the A/D line instead of price.

    Turning the open-ended :func:`ad` total into the gap between two of its own
    averages gives a bounded, oscillating reading of buying pressure.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        volume: Column name, expression, or series of traded volume.
        fast_period: Period of the faster exponential average.
        slow_period: Period of the slower exponential average.

    Returns:
        A ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``slow_period - 1`` rows are null. Both averages are seeded
        with the first A/D reading, as TA-Lib does, rather than with an SMA.

    Raises:
        ValueError: If a period is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    return apply_to_columns(
        (high, low, close, volume),
        lambda h, low_, c, v: _adosc_expr(h, low_, c, v, fast_period, slow_period),
    )


@overload
def obv(close: str | pl.Expr, volume: str | pl.Expr) -> pl.Expr: ...


@overload
def obv(close: pl.Series, volume: pl.Series) -> pl.Series: ...


def obv(close: IntoColumn, volume: IntoColumn) -> pl.Expr | pl.Series:
    """On Balance Volume: volume added on up bars and subtracted on down bars.

    Only the sign of the close-to-close change matters, so the line measures
    participation rather than price. The first bar seeds the total with its own
    volume, as TA-Lib does.

    Args:
        close: Column name, expression, or series of closing prices.
        volume: Column name, expression, or series of traded volume.

    Returns:
        A running total in volume units: a ``pl.Series`` when every input is a
        series, otherwise a ``pl.Expr``. There is no warm-up, and an unchanged
        close contributes nothing.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((close, volume), _obv_expr)
