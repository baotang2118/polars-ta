"""Absolute and percentage price oscillators."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window
from polars_ta.overlay.dispatch import MaType, _ma_expr, validate_ma_type


def _po_expr(
    values: pl.Expr,
    fast_period: int,
    slow_period: int,
    ma_type: MaType,
    percentage: bool,
) -> pl.Expr:
    # TA-Lib orders the periods rather than returning a sign-flipped result.
    fast_period, slow_period = (
        min(fast_period, slow_period),
        max(fast_period, slow_period),
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


def apo(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr:
    """Absolute Price Oscillator: the gap between a fast and a slow average.

    The same construction as the MACD line, but with a selectable average and
    no signal line. The periods are ordered before use, so swapping them
    changes nothing.

    Args:
        column: Column name or expression holding the input values.
        fast_period: Period of the faster average.
        slow_period: Period of the slower average.
        ma_type: Which average to apply; see :func:`~polars_ta.overlay.ma`.

    Returns:
        A ``pl.Expr`` yielding a difference in price units. Output starts once
        the slower average does.

    Raises:
        ValueError: If a period is invalid or ``ma_type`` is unknown.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_ma_type(ma_type)
    return _po_expr(to_expr(column), fast_period, slow_period, ma_type, False)


def ppo(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr:
    """Percentage Price Oscillator: :func:`apo` scaled by the slow average.

    Expressing the gap as a percentage makes readings comparable across
    instruments and across time, which a raw price difference is not.

    Args:
        column: Column name or expression holding the input values.
        fast_period: Period of the faster average.
        slow_period: Period of the slower average.
        ma_type: Which average to apply; see :func:`~polars_ta.overlay.ma`.

    Returns:
        A ``pl.Expr`` yielding a percentage. A zero slow average reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If a period is invalid or ``ma_type`` is unknown.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_ma_type(ma_type)
    return _po_expr(to_expr(column), fast_period, slow_period, ma_type, True)


def pvo(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    *,
    ma_type: MaType = "ema",
) -> pl.Expr:
    """Percentage Volume Oscillator: :func:`ppo` applied to volume.

    The same construction read on volume instead of price, so it says whether
    participation is picking up or draining away, independently of which way
    price went. It defaults to exponential averages, unlike :func:`ppo`.

    Args:
        column: Column name or expression holding traded volume.
        fast_period: Period of the faster average.
        slow_period: Period of the slower average.
        ma_type: Which average to apply; see :func:`~polars_ta.overlay.ma`.

    Returns:
        A ``pl.Expr`` yielding a percentage. A zero slow average reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If a period is invalid or ``ma_type`` is unknown.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_ma_type(ma_type)
    return _po_expr(to_expr(column), fast_period, slow_period, ma_type, True)
