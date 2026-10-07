"""Moving Average Convergence/Divergence."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window
from polars_ta.overlay.dispatch import MaType, _ma_expr, validate_ma_type
from polars_ta.overlay.ma import EmaMode, ema

MACD_FIELDS = ("macd", "signal", "histogram")


def _macd_struct(macd_line: pl.Expr, signal_line: pl.Expr) -> pl.Expr:
    # TA-Lib emits all three series from the same bar, so hold the faster
    # MACD line back until its signal exists.
    aligned = pl.when(signal_line.is_null()).then(None)
    return pl.struct(
        macd=aligned.otherwise(macd_line),
        signal=signal_line,
        histogram=aligned.otherwise(macd_line - signal_line),
    )


def _macd_expr(
    values: pl.Expr,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    mode: EmaMode,
) -> pl.Expr:
    macd_line = ema(values, fast_period, mode=mode) - ema(
        values, slow_period, mode=mode
    )
    return _macd_struct(macd_line, ema(macd_line, signal_period, mode=mode))


def _macdext_expr(
    values: pl.Expr,
    fast_period: int,
    fast_ma_type: MaType,
    slow_period: int,
    slow_ma_type: MaType,
    signal_period: int,
    signal_ma_type: MaType,
) -> pl.Expr:
    macd_line = _ma_expr(values, fast_period, fast_ma_type) - _ma_expr(
        values, slow_period, slow_ma_type
    )
    return _macd_struct(macd_line, _ma_expr(macd_line, signal_period, signal_ma_type))


def macd(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    mode: EmaMode = "talib",
) -> pl.Expr:
    """MACD: the gap between a fast and a slow EMA, against its own signal line.

    Args:
        column: Column name or expression holding the input values.
        fast_period: Period of the faster EMA.
        slow_period: Period of the slower EMA.
        signal_period: Period of the EMA applied to the MACD line.
        mode: EMA seeding convention shared by all three passes; see
            :func:`polars_ta.overlay.ma.ema`.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``macd``, ``signal``, and
        ``histogram``. All three fields start on the same row, after
        ``(slow_period - 1) + (signal_period - 1)`` leading nulls.

    Raises:
        ValueError: If any period is invalid, or if ``mode`` is unknown.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_window(signal_period)
    return _macd_expr(to_expr(column), fast_period, slow_period, signal_period, mode)


def macdext(
    column: IntoColumn,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    *,
    fast_ma_type: MaType = "sma",
    slow_ma_type: MaType = "sma",
    signal_ma_type: MaType = "sma",
) -> pl.Expr:
    """MACD with a separately selectable average for each of its three passes.

    :func:`macd` fixes all three to exponential averages; this allows, for
    example, a simple fast line against a weighted signal.

    Args:
        column: Column name or expression holding the input values.
        fast_period: Period of the faster average.
        slow_period: Period of the slower average.
        signal_period: Period of the average applied to the MACD line.
        fast_ma_type: Average used for the fast line.
        slow_ma_type: Average used for the slow line.
        signal_ma_type: Average used for the signal line.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``macd``, ``signal``, and
        ``histogram``. All three fields start on the same row, once the slow
        and signal averages are both available.

    Raises:
        ValueError: If a period is invalid or an ``ma_type`` is unknown.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    validate_window(signal_period)
    for ma_type in (fast_ma_type, slow_ma_type, signal_ma_type):
        validate_ma_type(ma_type)
    return _macdext_expr(
        to_expr(column),
        fast_period,
        fast_ma_type,
        slow_period,
        slow_ma_type,
        signal_period,
        signal_ma_type,
    )


def macdfix(
    column: IntoColumn, signal_period: int = 9, *, mode: EmaMode = "talib"
) -> pl.Expr:
    """MACD fixed at the classic 12/26 periods, leaving only the signal tunable.

    Args:
        column: Column name or expression holding the input values.
        signal_period: Period of the EMA applied to the MACD line.
        mode: EMA seeding convention shared by all three passes; see
            :func:`polars_ta.overlay.ma.ema`.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``macd``, ``signal``, and
        ``histogram``. All three fields start after ``25 + (signal_period - 1)``
        leading nulls.

    Raises:
        ValueError: If ``signal_period`` is invalid, or if ``mode`` is unknown.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    return macd(column, 12, 26, signal_period, mode=mode)
