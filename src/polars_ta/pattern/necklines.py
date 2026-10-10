"""Rebounds that close back into a prior long black bar without clearing it."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn
from polars_ta.pattern._candles import (
    BODY_LONG,
    BODY_SHORT,
    EQUAL,
    Bars,
    signal,
)


def _long_black_then_white_below_low(bars: Bars) -> pl.Expr:
    """A long black bar, then a white bar opening below its low."""
    return (
        bars.black(1)
        & (bars.body(1) > bars.average(BODY_LONG, 1))
        & bars.white(0)
        & (bars.open(0) < bars.low(1))
    )


def cdlinneck(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """In-Neck: a white bar that closes only just inside a long black body.

    The rally opens below the black bar's low and recovers barely past its
    close, so the ground regained is slight.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        _long_black_then_white_below_low(bars)
        & (bars.close(0) <= bars.close(1) + bars.average(EQUAL, 1))
        & (bars.close(0) >= bars.close(1))
    )
    return signal(bars, 11, pattern, -100)


def cdlonneck(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """On-Neck: a white bar that closes level with a long black bar's low.

    The rally opens below that low and stops right at it, failing to reach
    into the black bar's body at all.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    equal = bars.average(EQUAL, 1)
    pattern = (
        _long_black_then_white_below_low(bars)
        & (bars.close(0) <= bars.low(1) + equal)
        & (bars.close(0) >= bars.low(1) - equal)
    )
    return signal(bars, 11, pattern, -100)


def cdlthrusting(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Thrusting: a white bar that closes into a long black body's lower half.

    The rally opens below the black bar's low and pushes clearly past its
    close, yet still stops short of halving the decline.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        _long_black_then_white_below_low(bars)
        & (bars.close(0) > bars.close(1) + bars.average(EQUAL, 1))
        & (bars.close(0) <= bars.close(1) + bars.body(1) * 0.5)
    )
    return signal(bars, 11, pattern, -100)


def cdlhomingpigeon(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Homing Pigeon: a small black bar held entirely inside a long black one.

    The second session repeats the first's direction on a much smaller range
    and stays within its body, so the decline loses width.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.black(1)
        & bars.black(0)
        & (bars.body(1) > bars.average(BODY_LONG, 1))
        & (bars.body(0) <= bars.average(BODY_SHORT, 0))
        & (bars.open(0) < bars.open(1))
        & (bars.close(0) > bars.close(1))
    )
    return signal(bars, 11, pattern, 100)
