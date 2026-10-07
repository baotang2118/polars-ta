"""Aroon and the Aroon Oscillator."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window

AROON_FIELDS = ("down", "up")


def _bars_since(values: pl.Expr, extreme: pl.Expr, window: int) -> pl.Expr:
    """How many bars back the window's extreme last occurred, ties to the newest."""
    chain = pl.when(extreme.is_null()).then(None)
    for offset in range(window + 1):
        chain = chain.when(values.shift(offset) == extreme).then(float(offset))
    return chain.otherwise(None)


def _aroon_pair(high: pl.Expr, low: pl.Expr, window: int) -> tuple[pl.Expr, pl.Expr]:
    # The window spans the last `window` bars plus today, so `window + 1` rows.
    span = window + 1
    highest = high.rolling_max(window_size=span, min_samples=span)
    lowest = low.rolling_min(window_size=span, min_samples=span)
    since_high = _bars_since(high, highest, window)
    since_low = _bars_since(low, lowest, window)
    factor = 100.0 / window
    return factor * (window - since_low), factor * (window - since_high)


def _aroon_expr(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    down, up = _aroon_pair(high, low, window)
    return pl.struct(down=down, up=up)


def _aroonosc_expr(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    down, up = _aroon_pair(high, low, window)
    return up - down


def aroon(high: IntoColumn, low: IntoColumn, window: int = 14) -> pl.Expr:
    """Aroon: how recently the window's high and low were set.

    ``up`` reaches ``100`` on the bar that sets a new window high and decays by
    ``100 / window`` for every bar since; ``down`` does the same for the low.
    A reading says nothing about the size of a move, only its freshness.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        window: Number of periods looked back, in addition to the current bar.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``down`` and ``up``, each
        in ``[0, 100]``. The first ``window`` rows are null. Repeated extremes
        count from the most recent occurrence, as in TA-Lib.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    high_expr, low_expr = to_exprs(high, low)
    return _aroon_expr(high_expr, low_expr, window)


def aroonosc(high: IntoColumn, low: IntoColumn, window: int = 14) -> pl.Expr:
    """Aroon Oscillator: the Aroon up line minus the down line.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        window: Number of periods looked back, in addition to the current bar.

    Returns:
        A ``pl.Expr`` yielding a value in ``[-100, 100]``. The first
        ``window`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    high_expr, low_expr = to_exprs(high, low)
    return _aroonosc_expr(high_expr, low_expr, window)
