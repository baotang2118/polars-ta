"""Absolute and percentage price oscillators."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window
from polars_ta.overlay.dispatch import MaType, _ma_expr, validate_ma_type


def _po_expr(
    values: pl.Expr,
    fast_period: int,
    slow_period: int,
    ma_type: MaType,
    percentage: bool,
) -> pl.Expr:
    # TA-Lib orders the periods rather than returning a sign-flipped result.
    fast_period, slow_period = min(fast_period, slow_period), max(
        fast_period, slow_period
    )
    fast = _ma_expr(values, fast_period, ma_type)
    slow = _ma_expr(values, slow_period, ma_type)
    if not percentage:
        return fast - slow
    return (
        pl.when(fast.is_null() | slow.is_null())
        .then(None)
        .when(slow != 0.0)
        .then((fast - slow) / slow * 100.0)
        .otherwise(0.0)
    )


@overload
def apo(
    column: str | pl.Expr,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr: ...


@overload
def apo(
    column: pl.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Series: ...


def apo(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr | pl.Series:
    """Absolute Price Oscillator: the gap between a fast and a slow average.

    The same construction as the MACD line, but with a selectable average and
    no signal line. The periods are ordered before use, so swapping them
    changes nothing.

    Args:
        column: Column name, expression, or series holding the input values.
        fast_period: Period of the faster average.
        slow_period: Period of the slower average.
        ma_type: Which average to apply; see :func:`~polars_ta.overlay.ma`.

    Returns:
        A difference in price units: a ``pl.Series`` when ``column`` is a
        series, otherwise a ``pl.Expr``. Output starts once the slower average
        does.

    Raises:
        ValueError: If a period is invalid or ``ma_type`` is unknown.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_ma_type(ma_type)
    return apply_to_column(
        column,
        lambda values: _po_expr(values, fast_period, slow_period, ma_type, False),
    )


@overload
def ppo(
    column: str | pl.Expr,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr: ...


@overload
def ppo(
    column: pl.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Series: ...


def ppo(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr | pl.Series:
    """Percentage Price Oscillator: :func:`apo` scaled by the slow average.

    Expressing the gap as a percentage makes readings comparable across
    instruments and across time, which a raw price difference is not.

    Args:
        column: Column name, expression, or series holding the input values.
        fast_period: Period of the faster average.
        slow_period: Period of the slower average.
        ma_type: Which average to apply; see :func:`~polars_ta.overlay.ma`.

    Returns:
        A percentage: a ``pl.Series`` when ``column`` is a series, otherwise a
        ``pl.Expr``. A zero slow average reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If a period is invalid or ``ma_type`` is unknown.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_ma_type(ma_type)
    return apply_to_column(
        column,
        lambda values: _po_expr(values, fast_period, slow_period, ma_type, True),
    )
