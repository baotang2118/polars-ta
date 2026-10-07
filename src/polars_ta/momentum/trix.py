"""TRIX: the rate of change of a triple-smoothed exponential average."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr
from polars_ta.momentum.roc import _change_expr
from polars_ta.overlay.ma import EmaMode, _ema_expr, _resolve_alpha


def _trix_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    stage = values
    for _ in range(3):
        stage = _ema_expr(stage, window, alpha, mode)
    return _change_expr(stage, 1, lambda now, before: (now / before - 1.0) * 100.0)


def trix(column: IntoColumn, window: int = 30, *, mode: EmaMode = "talib") -> pl.Expr:
    """TRIX: the one-period rate of change of a triple exponential average.

    Three smoothing passes strip out cycles shorter than ``window``, so the
    remaining slope isolates the dominant trend and crosses zero when it turns.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods used for all three EMA passes.
        mode: EMA seeding convention; see :func:`~polars_ta.overlay.ma.ema`.

    Returns:
        A ``pl.Expr`` yielding a percentage. The first ``3 * (window - 1) + 1``
        rows are null.

    Raises:
        ValueError: If ``window`` or ``mode`` is invalid.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    smoothing = _resolve_alpha(window, None, mode)
    return _trix_expr(to_expr(column), window, smoothing, mode)
