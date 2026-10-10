"""Hikkake patterns, inside-bar traps that may be confirmed within three bars."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn
from polars_ta.pattern._candles import NEAR, PATTERN_DTYPE, Bars

_HIKKAKE_LOOKBACK = 5
_HIKKAKEMOD_LOOKBACK = 10
# TA-Lib primes the state machine three bars before its first output.
_HIKKAKE_SCAN_START = _HIKKAKE_LOOKBACK - 3
_HIKKAKEMOD_SCAN_START = _HIKKAKEMOD_LOOKBACK - 3


def _hikkake_scan(frame: pl.Series) -> pl.Series:
    """Walk the bars, carrying the open hikkake forward to its confirmation."""
    high = frame.struct.field("high").to_list()
    low = frame.struct.field("low").to_list()
    close = frame.struct.field("close").to_list()

    result: list[int] = [0] * len(high)
    pattern_idx = 0
    pattern_result = 0
    for i in range(_HIKKAKE_SCAN_START, len(high)):
        hi, hi1, hi2 = high[i], high[i - 1], high[i - 2]
        lo, lo1, lo2 = low[i], low[i - 1], low[i - 2]
        cl = close[i]
        if (
            hi is None
            or hi1 is None
            or hi2 is None
            or lo is None
            or lo1 is None
            or lo2 is None
            or cl is None
        ):
            continue
        inside = hi1 < hi2 and lo1 > lo2
        bull = hi < hi1 and lo < lo1
        bear = hi > hi1 and lo > lo1
        if inside and (bull or bear):
            pattern_result = 100 if bull else -100
            pattern_idx = i
            result[i] = pattern_result
            continue
        # pattern_idx of 0 means nothing is open, so bar -1 is never read.
        if pattern_idx > 0 and i <= pattern_idx + 3:
            prior_high = high[pattern_idx - 1]
            prior_low = low[pattern_idx - 1]
            broke_up = pattern_result > 0 and prior_high is not None and cl > prior_high
            broke_down = pattern_result < 0 and prior_low is not None and cl < prior_low
            if broke_up or broke_down:
                result[i] = pattern_result + (100 if pattern_result > 0 else -100)
                pattern_idx = 0

    return pl.Series(values=result, dtype=PATTERN_DTYPE)


def _hikkakemod_scan(frame: pl.Series) -> pl.Series:
    """Walk the bars, carrying the open modified hikkake to its confirmation."""
    high = frame.struct.field("high").to_list()
    low = frame.struct.field("low").to_list()
    close = frame.struct.field("close").to_list()
    near = frame.struct.field("near").to_list()

    result: list[int] = [0] * len(high)
    pattern_idx = 0
    pattern_result = 0
    for i in range(_HIKKAKEMOD_SCAN_START, len(high)):
        hi, hi1, hi2, hi3 = high[i], high[i - 1], high[i - 2], high[i - 3]
        lo, lo1, lo2, lo3 = low[i], low[i - 1], low[i - 2], low[i - 3]
        cl, cl2, span = close[i], close[i - 2], near[i]
        if (
            hi is None
            or hi1 is None
            or hi2 is None
            or hi3 is None
            or lo is None
            or lo1 is None
            or lo2 is None
            or lo3 is None
            or cl is None
            or cl2 is None
            or span is None
        ):
            continue
        inside = hi2 < hi3 and lo2 > lo3 and hi1 < hi2 and lo1 > lo2
        bull = hi < hi1 and lo < lo1 and cl2 <= lo2 + span
        bear = hi > hi1 and lo > lo1 and cl2 >= hi2 - span
        if inside and (bull or bear):
            pattern_result = 100 if bull else -100
            pattern_idx = i
            result[i] = pattern_result
            continue
        # pattern_idx of 0 means nothing is open, so bar -1 is never read.
        if pattern_idx > 0 and i <= pattern_idx + 3:
            prior_high = high[pattern_idx - 1]
            prior_low = low[pattern_idx - 1]
            broke_up = pattern_result > 0 and prior_high is not None and cl > prior_high
            broke_down = pattern_result < 0 and prior_low is not None and cl < prior_low
            if broke_up or broke_down:
                result[i] = pattern_result + (100 if pattern_result > 0 else -100)
                pattern_idx = 0

    return pl.Series(values=result, dtype=PATTERN_DTYPE)


def cdlhikkake(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Hikkake: an inside bar, then a break out of it that may be a trap.

    The second bar trades entirely inside the first, and the third breaks out
    of that narrow range. The break is recorded when it happens, and again on
    any of the next three bars that closes back past the far side of the
    inside bar, which is the point at which the first break looks like a trap.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a break below the inside bar,
        ``-100`` on a break above it, ``±200`` on a bar that later closes back
        through the inside bar's opposite extreme (carrying the sign of the
        break it reverses), and ``0`` otherwise. The first 5 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    scanned = pl.struct(
        high=bars.high(0), low=bars.low(0), close=bars.close(0)
    ).map_batches(_hikkake_scan, return_dtype=PATTERN_DTYPE)
    return (
        pl.when(bars.unusable(_HIKKAKE_LOOKBACK))
        .then(None)
        .otherwise(scanned)
        .cast(PATTERN_DTYPE)
    )


def cdlhikkakemod(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Modified Hikkake: a hikkake whose inside bar closed at one extreme.

    A reference bar is followed by a narrower bar that closes near its own low
    or high, then by an inside bar, then by a break out of that inside bar.
    Requiring the close near an extreme makes this the reversal reading of the
    hikkake, where the plain pattern can also continue the move. Confirmation
    works the same way: any of the next three bars closing back past the far
    side of the inside bar is recorded too.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a break below the inside bar,
        ``-100`` on a break above it, ``±200`` on a bar that later closes back
        through the inside bar's opposite extreme (carrying the sign of the
        break it reverses), and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    scanned = pl.struct(
        high=bars.high(0),
        low=bars.low(0),
        close=bars.close(0),
        near=bars.average(NEAR, 2),
    ).map_batches(_hikkakemod_scan, return_dtype=PATTERN_DTYPE)
    return (
        pl.when(bars.unusable(_HIKKAKEMOD_LOOKBACK))
        .then(None)
        .otherwise(scanned)
        .cast(PATTERN_DTYPE)
    )
