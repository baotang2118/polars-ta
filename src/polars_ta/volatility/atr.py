"""True Range and Average True Range."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, to_exprs, validate_window
from polars_ta.overlay.ma import ema


def _true_range_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    previous_close = close.shift(1)
    # max_horizontal ignores nulls, so guard the inputs rather than relying on it.
    known = high.is_not_null() & low.is_not_null() & previous_close.is_not_null()
    return (
        pl.when(known)
        .then(
            pl.max_horizontal(
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            )
        )
        .otherwise(None)
    )


def _atr_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    return ema(_true_range_expr(high, low, close), window, alpha=1.0 / window)


def _natr_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    average = _atr_expr(high, low, close, window)
    return (
        pl.when(average.is_null() | close.is_null())
        .then(None)
        .when(close != 0.0)
        .then(100.0 * average / close)
        .otherwise(0.0)
    )


def true_range(high: IntoColumn, low: IntoColumn, close: IntoColumn) -> pl.Expr:
    """True Range: the widest of today's span and either gap from yesterday's close.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding a non-negative range. The first row is null,
        since it has no previous close.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    return _true_range_expr(*to_exprs(high, low, close))


def atr(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Average True Range: Wilder-smoothed true range, a pure volatility measure.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods to smooth over.

    Returns:
        A ``pl.Expr`` yielding a non-negative volatility measure in price
        units. The first ``window`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _atr_expr(to_expr(high), to_expr(low), to_expr(close), window)


def natr(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Normalized Average True Range: the ATR as a percentage of the close.

    Expressing volatility relative to price makes readings comparable across
    instruments and across long stretches of history, which the raw ATR is not.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods to smooth over.

    Returns:
        A ``pl.Expr`` yielding a non-negative percentage. The first ``window``
        rows are null, and a zero close reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _natr_expr(to_expr(high), to_expr(low), to_expr(close), window)
