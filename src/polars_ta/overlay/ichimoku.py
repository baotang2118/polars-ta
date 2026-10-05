"""Ichimoku Kinko Hyo."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window

ICHIMOKU_FIELDS = ("conversion", "base", "span_a", "span_b", "lagging")


def _midpoint(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    highest = high.rolling_max(window_size=window, min_samples=window)
    lowest = low.rolling_min(window_size=window, min_samples=window)
    return (highest + lowest) / 2.0


def _ichimoku_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    conversion_period: int,
    base_period: int,
    span_b_period: int,
    displacement: int,
) -> pl.Expr:
    conversion = _midpoint(high, low, conversion_period)
    base = _midpoint(high, low, base_period)
    return pl.struct(
        conversion=conversion,
        base=base,
        span_a=((conversion + base) / 2.0).shift(displacement),
        span_b=_midpoint(high, low, span_b_period).shift(displacement),
        lagging=close.shift(-displacement),
    )


@overload
def ichimoku(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> pl.Expr: ...


@overload
def ichimoku(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> pl.Series: ...


def ichimoku(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> pl.Expr | pl.Series:
    """Ichimoku Kinko Hyo: five lines, two of which form the cloud.

    The two leading spans are displaced forward and the lagging span backward,
    as they are plotted. **The lagging span therefore contains future closes
    relative to its row** and must not be fed into a backtest signal directly.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        conversion_period: Lookback of the conversion line (Tenkan-sen).
        base_period: Lookback of the base line (Kijun-sen).
        span_b_period: Lookback of leading span B (Senkou Span B).
        displacement: Bars the spans are shifted by (Senkou/Chikou offset).

    Returns:
        A struct with fields ``conversion``, ``base``, ``span_a``, ``span_b``,
        and ``lagging``: a ``pl.Series`` when every input is a series,
        otherwise a ``pl.Expr``. Each field carries its own warm-up.

    Raises:
        ValueError: If any period or the displacement is invalid.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(conversion_period)
    validate_window(base_period)
    validate_window(span_b_period)
    validate_window(displacement)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _ichimoku_expr(
            h, low_, c, conversion_period, base_period, span_b_period, displacement
        ),
    )
