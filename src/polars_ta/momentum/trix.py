"""TRIX: the rate of change of a triple-smoothed exponential average."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column
from polars_ta.momentum.roc import _change_expr
from polars_ta.overlay.ma import EmaMode, _ema_expr, _resolve_alpha


def _trix_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    stage = values
    for _ in range(3):
        stage = _ema_expr(stage, window, alpha, mode)
    return _change_expr(stage, 1, lambda now, before: (now / before - 1.0) * 100.0)


@overload
def trix(
    column: str | pl.Expr, window: int = 30, *, mode: EmaMode = "talib"
) -> pl.Expr: ...


@overload
def trix(
    column: pl.Series, window: int = 30, *, mode: EmaMode = "talib"
) -> pl.Series: ...


def trix(
    column: IntoColumn, window: int = 30, *, mode: EmaMode = "talib"
) -> pl.Expr | pl.Series:
    """TRIX: the one-period rate of change of a triple exponential average.

    Three smoothing passes strip out cycles shorter than ``window``, so the
    remaining slope isolates the dominant trend and crosses zero when it turns.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods used for all three EMA passes.
        mode: EMA seeding convention; see :func:`~polars_ta.overlay.ma.ema`.

    Returns:
        A percentage: a ``pl.Series`` when ``column`` is a series, otherwise a
        ``pl.Expr``. The first ``3 * (window - 1) + 1`` rows are null.

    Raises:
        ValueError: If ``window`` or ``mode`` is invalid.
    """
    smoothing = _resolve_alpha(window, None, mode)
    return apply_to_column(
        column, lambda values: _trix_expr(values, window, smoothing, mode)
    )
