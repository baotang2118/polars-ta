"""Moving-average dispatch: one entry point over every average in the package."""

from __future__ import annotations

from typing import Literal, get_args, overload

import polars as pl

from polars_ta._common import (
    IntoColumn,
    apply_to_column,
    apply_to_columns,
    validate_window,
)
from polars_ta.overlay.adaptive import _kama_expr, _mama_expr
from polars_ta.overlay.ma import (
    _dema_expr,
    _ema_expr,
    _sma_expr,
    _t3_expr,
    _tema_expr,
    _trima_expr,
    _wma_expr,
)

MaType = Literal["sma", "ema", "wma", "dema", "tema", "trima", "kama", "mama", "t3"]
"""Moving-average kinds accepted by :func:`ma` and the indicators built on it."""

MA_TYPES: tuple[str, ...] = get_args(MaType)


def validate_ma_type(ma_type: str) -> None:
    """Raise ``ValueError`` unless ``ma_type`` names a supported average."""
    if ma_type not in MA_TYPES:
        raise ValueError(f"ma_type must be one of {list(MA_TYPES)}, got {ma_type!r}")


def _ma_expr(values: pl.Expr, window: int, ma_type: MaType) -> pl.Expr:
    if ma_type == "sma":
        return _sma_expr(values, window)
    if ma_type == "wma":
        return _wma_expr(values, window)
    if ma_type == "trima":
        return _trima_expr(values, window)
    if ma_type == "kama":
        return _kama_expr(values, window, 2, 30)
    if ma_type == "mama":
        # TA-Lib's MA takes the MAMA line and ignores the window entirely.
        return _mama_expr(values, 0.5, 0.05).struct.field("mama")
    alpha = 2.0 / (window + 1.0)
    if ma_type == "ema":
        return _ema_expr(values, window, alpha, "talib")
    if ma_type == "dema":
        return _dema_expr(values, window, alpha, "talib")
    if ma_type == "tema":
        return _tema_expr(values, window, alpha, "talib")
    return _t3_expr(values, window, alpha, "talib", 0.7)


def _mavp_expr(
    values: pl.Expr,
    periods: pl.Expr,
    min_period: int,
    max_period: int,
    ma_type: MaType,
) -> pl.Expr:
    wanted = periods.cast(pl.Int64).clip(min_period, max_period)
    # One average per candidate period, selected row by row.
    chosen = pl.when(wanted == min_period).then(_ma_expr(values, min_period, ma_type))
    for window in range(min_period + 1, max_period + 1):
        chosen = chosen.when(wanted == window).then(_ma_expr(values, window, ma_type))
    longest = _ma_expr(values, max_period, ma_type)
    # TA-Lib emits nothing until the longest permitted average is available.
    return pl.when(longest.is_null() | wanted.is_null()).then(None).otherwise(chosen)


@overload
def ma(
    column: str | pl.Expr, window: int = 30, *, ma_type: MaType = "sma"
) -> pl.Expr: ...


@overload
def ma(
    column: pl.Series, window: int = 30, *, ma_type: MaType = "sma"
) -> pl.Series: ...


def ma(
    column: IntoColumn, window: int = 30, *, ma_type: MaType = "sma"
) -> pl.Expr | pl.Series:
    """Moving average of the requested kind.

    A single entry point over :func:`~polars_ta.overlay.ma.sma`,
    :func:`~polars_ta.overlay.ma.ema`, :func:`~polars_ta.overlay.ma.wma`,
    :func:`~polars_ta.overlay.ma.dema`, :func:`~polars_ta.overlay.ma.tema`,
    :func:`~polars_ta.overlay.ma.trima`, :func:`~polars_ta.overlay.ma.t3`, and
    :func:`~polars_ta.overlay.adaptive.kama`, so that an average can be chosen
    at runtime. Each kind keeps its own default parameters and its own warm-up.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods passed to the chosen average.
        ma_type: Which average to apply; one of :data:`MA_TYPES`.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.

    Raises:
        ValueError: If ``window`` is invalid or ``ma_type`` is unknown.
    """
    validate_window(window)
    validate_ma_type(ma_type)
    return apply_to_column(column, lambda values: _ma_expr(values, window, ma_type))


@overload
def mavp(
    column: str | pl.Expr,
    periods: str | pl.Expr,
    min_period: int = 2,
    max_period: int = 30,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr: ...


@overload
def mavp(
    column: pl.Series,
    periods: pl.Series,
    min_period: int = 2,
    max_period: int = 30,
    *,
    ma_type: MaType = "sma",
) -> pl.Series: ...


def mavp(
    column: IntoColumn,
    periods: IntoColumn,
    min_period: int = 2,
    max_period: int = 30,
    *,
    ma_type: MaType = "sma",
) -> pl.Expr | pl.Series:
    """Moving average whose period is read from a second column, row by row.

    Each row's period is truncated to an integer and clamped to
    ``[min_period, max_period]``. One average is built per candidate period, so
    the expression grows linearly with ``max_period - min_period``; keep that
    span small, especially for the chained averages.

    Args:
        column: Column name, expression, or series holding the input values.
        periods: Column name, expression, or series of per-row periods.
        min_period: Smallest period a row may request.
        max_period: Largest period a row may request.
        ma_type: Which average to apply; one of :data:`MA_TYPES`.

    Returns:
        A ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        Output starts only once the ``max_period`` average is available, as in
        TA-Lib, so the warm-up does not change from row to row.

    Raises:
        ValueError: If a period bound is invalid, ``min_period`` exceeds
            ``max_period``, or ``ma_type`` is unknown.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(min_period)
    validate_window(max_period)
    validate_ma_type(ma_type)
    if min_period > max_period:
        raise ValueError(
            f"min_period must not exceed max_period, got {min_period} > {max_period}"
        )
    return apply_to_columns(
        (column, periods),
        lambda values, wanted: _mavp_expr(
            values, wanted, min_period, max_period, ma_type
        ),
    )
