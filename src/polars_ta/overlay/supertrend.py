"""Supertrend overlay."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import (
    IntoColumn,
    apply_to_columns,
    validate_positive,
    validate_window,
)
from polars_ta.volatility.atr import _atr_expr

SUPERTREND_FIELDS = ("supertrend", "direction")

_RETURN_DTYPE = pl.Struct({"supertrend": pl.Float64, "direction": pl.Int8})


def _supertrend_scan(bands: pl.Series) -> pl.Series:
    """Carry the final bands and trend direction forward, bar by bar."""
    close = bands.struct.field("close").to_list()
    upper = bands.struct.field("upper").to_list()
    lower = bands.struct.field("lower").to_list()

    trend: list[float | None] = [None] * len(close)
    direction: list[int | None] = [None] * len(close)
    previous = None
    for index in range(len(close)):
        if close[index] is None or upper[index] is None or lower[index] is None:
            continue
        if previous is None:
            direction[index] = 1
            trend[index] = lower[index]
            previous = index
            continue
        if close[index] > upper[previous]:
            direction[index] = 1
        elif close[index] < lower[previous]:
            direction[index] = -1
        else:
            direction[index] = direction[previous]
            if direction[index] == 1 and lower[index] < lower[previous]:
                lower[index] = lower[previous]
            if direction[index] == -1 and upper[index] > upper[previous]:
                upper[index] = upper[previous]
        trend[index] = lower[index] if direction[index] == 1 else upper[index]
        previous = index

    return pl.Series(
        values=[{"supertrend": t, "direction": d} for t, d in zip(trend, direction)],
        dtype=_RETURN_DTYPE,
    )


def _supertrend_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
    multiplier: float,
) -> pl.Expr:
    offset = multiplier * _atr_expr(high, low, close, window)
    median = (high + low) / 2.0
    return pl.struct(
        close=close, upper=median + offset, lower=median - offset
    ).map_batches(_supertrend_scan, return_dtype=_RETURN_DTYPE)


@overload
def supertrend(
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
    window: int = 10,
    multiplier: float = 3.0,
) -> pl.Expr: ...


@overload
def supertrend(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    window: int = 10,
    multiplier: float = 3.0,
) -> pl.Series: ...


def supertrend(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 10,
    multiplier: float = 3.0,
) -> pl.Expr | pl.Series:
    """Supertrend: an ATR band that flips sides when price closes through it.

    Bands are drawn ``multiplier`` ATRs either side of the bar's median price,
    then ratcheted so they only ever tighten toward price while the trend
    holds. Closing beyond the opposite band flips the trend.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.
        window: Number of periods for the ATR.
        multiplier: Band width in ATRs.

    Returns:
        A struct with fields ``supertrend`` (the active band, in price units)
        and ``direction`` (``1`` while rising, ``-1`` while falling): a
        ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null, following the ATR.

    Raises:
        ValueError: If ``window`` or ``multiplier`` is invalid.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    validate_positive("multiplier", multiplier)
    return apply_to_columns(
        (high, low, close),
        lambda h, low_, c: _supertrend_expr(h, low_, c, window, multiplier),
    )
