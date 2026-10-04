"""Moving Average Convergence/Divergence."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window
from polars_ta.overlay.ma import EmaMode, ema

MACD_FIELDS = ("macd", "signal", "histogram")


def _macd_expr(
    values: pl.Expr,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    mode: EmaMode,
) -> pl.Expr:
    fast = ema(values, fast_period, mode=mode)
    slow = ema(values, slow_period, mode=mode)
    macd_line = fast - slow
    signal_line = ema(macd_line, signal_period, mode=mode)
    # TA-Lib emits all three series from the same bar, so hold the faster
    # MACD line back until its signal exists.
    aligned = pl.when(signal_line.is_null()).then(None)
    return pl.struct(
        macd=aligned.otherwise(macd_line),
        signal=signal_line,
        histogram=aligned.otherwise(macd_line - signal_line),
    )


@overload
def macd(
    column: str | pl.Expr,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    mode: EmaMode = "talib",
) -> pl.Expr: ...


@overload
def macd(
    column: pl.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    mode: EmaMode = "talib",
) -> pl.Series: ...


def macd(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    mode: EmaMode = "talib",
) -> pl.Expr | pl.Series:
    """MACD: the gap between a fast and a slow EMA, against its own signal line.

    Args:
        column: Column name, expression, or series holding the input values.
        fast_period: Period of the faster EMA.
        slow_period: Period of the slower EMA.
        signal_period: Period of the EMA applied to the MACD line.
        mode: EMA seeding convention shared by all three passes; see
            :func:`polars_ta.overlay.ma.ema`.

    Returns:
        A struct with fields ``macd``, ``signal``, and ``histogram``: a
        ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        All three fields start on the same row, after
        ``(slow_period - 1) + (signal_period - 1)`` leading nulls.

    Raises:
        ValueError: If any period is invalid, or if ``mode`` is unknown.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_window(signal_period)
    return apply_to_column(
        column,
        lambda values: _macd_expr(
            values, fast_period, slow_period, signal_period, mode
        ),
    )
