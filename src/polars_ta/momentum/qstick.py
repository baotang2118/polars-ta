"""Qstick."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window
from polars_ta.overlay.ma import _sma_expr


def _qstick_expr(open_: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    return _sma_expr(close - open_, window)


def qstick(open_: IntoColumn, close: IntoColumn, window: int = 10) -> pl.Expr:
    """Qstick: the average candle body over a window.

    Averaging ``close - open`` measures how the sessions themselves finished,
    which is a different question from where price ended up: a market can grind
    higher on a run of red candles, and Qstick reads that as weakness.

    Args:
        open_: Column name or expression of opening prices.
        close: Column name or expression of closing prices.
        window: Number of periods averaged.

    Returns:
        A ``pl.Expr`` yielding a difference in price units; positive while
        candles close above their opens on balance. The first ``window - 1``
        rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    open_expr, close_expr = to_exprs(open_, close)
    return _qstick_expr(open_expr, close_expr, window)
