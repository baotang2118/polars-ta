"""Awesome Oscillator."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window
from polars_ta.overlay.ma import _sma_expr
from polars_ta.overlay.transform import _medprice_expr


def _ao_expr(
    high: pl.Expr, low: pl.Expr, fast_period: int, slow_period: int
) -> pl.Expr:
    median = _medprice_expr(high, low)
    return _sma_expr(median, fast_period) - _sma_expr(median, slow_period)


def ao(
    high: IntoColumn,
    low: IntoColumn,
    fast_period: int = 5,
    slow_period: int = 34,
) -> pl.Expr:
    """Awesome Oscillator: two simple averages of the median price, differenced.

    Building on the bar's midpoint rather than its close keeps the reading out
    of the hands of a single print, and the gap between a short and a long
    average says whether recent momentum is running ahead of the longer trend.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        fast_period: Period of the faster average.
        slow_period: Period of the slower average.

    Returns:
        A ``pl.Expr`` yielding a difference in price units. The first
        ``slow_period - 1`` rows are null.

    Raises:
        ValueError: If a period is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    high_expr, low_expr = to_exprs(high, low)
    return _ao_expr(high_expr, low_expr, fast_period, slow_period)
