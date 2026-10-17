"""Vertical Horizontal Filter."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window


def _vhf_expr(values: pl.Expr, window: int) -> pl.Expr:
    highest = values.rolling_max(window_size=window, min_samples=window)
    lowest = values.rolling_min(window_size=window, min_samples=window)
    travelled = values.diff().abs().rolling_sum(window_size=window, min_samples=window)
    return (
        pl.when(highest.is_null() | lowest.is_null() | travelled.is_null())
        .then(None)
        .when(travelled != 0.0)
        .then((highest - lowest) / travelled)
        .otherwise(0.0)
    )


def vhf(column: IntoColumn, window: int = 28) -> pl.Expr:
    """Vertical Horizontal Filter: net travel over total travel.

    The window's range is divided by the distance price actually covered to
    get there. A straight-line move spends all its travel on the range and
    reads near ``1``; a market thrashing inside the same range reads near
    ``0``. It says whether a trend-following or a range-bound tool suits the
    current market, not which way to trade it.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the range and the travel sum.

    Returns:
        A ``pl.Expr`` yielding a ratio in ``[0, 1]``. The first ``window`` rows
        are null: one for the differences and ``window - 1`` for the sums. A
        window that never moved reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _vhf_expr(to_expr(column), window)
