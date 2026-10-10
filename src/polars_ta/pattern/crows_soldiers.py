"""Crows and soldiers: runs of same-coloured bars, and the blocks that stall them."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn
from polars_ta.pattern._candles import (
    BODY_LONG,
    BODY_SHORT,
    EQUAL,
    FAR,
    NEAR,
    SHADOW_LONG,
    SHADOW_SHORT,
    SHADOW_VERY_SHORT,
    Bars,
    signal,
)


def cdl2crows(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Two Crows: a gap up after a long white bar, then two black bars that fill it.

    The second black bar opens inside the first black bar's body and closes
    back inside the white bar's, undoing the gap and with it the advance.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.white(2)
        & (bars.body(2) > bars.average(BODY_LONG, 2))
        & bars.black(1)
        & bars.body_gap_up(1, 2)
        & bars.black(0)
        & (bars.open(0) < bars.open(1))
        & (bars.open(0) > bars.close(1))
        & (bars.close(0) > bars.open(2))
        & (bars.close(0) < bars.close(2))
    )
    return signal(bars, 12, pattern, -100)


def cdl3blackcrows(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Three Black Crows: three black bars stepping down from a white bar's high.

    Each bar opens inside the previous body and closes near its own low, so
    three sessions in a row end with sellers holding the ground they took.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 13 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.white(3)
        & bars.black(2)
        & (bars.lower_shadow(2) < bars.average(SHADOW_VERY_SHORT, 2))
        & bars.black(1)
        & (bars.lower_shadow(1) < bars.average(SHADOW_VERY_SHORT, 1))
        & bars.black(0)
        & (bars.lower_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & (bars.open(1) < bars.open(2))
        & (bars.open(1) > bars.close(2))
        & (bars.open(0) < bars.open(1))
        & (bars.open(0) > bars.close(1))
        & (bars.high(3) > bars.close(2))
        & (bars.close(2) > bars.close(1))
        & (bars.close(1) > bars.close(0))
    )
    return signal(bars, 13, pattern, -100)


def cdlidentical3crows(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Identical Three Crows: three declining black bars opening at the prior close.

    Each bar starts where the one before it finished rather than gapping, so
    the decline proceeds in even steps with no recovery between sessions.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.black(2)
        & (bars.lower_shadow(2) < bars.average(SHADOW_VERY_SHORT, 2))
        & bars.black(1)
        & (bars.lower_shadow(1) < bars.average(SHADOW_VERY_SHORT, 1))
        & bars.black(0)
        & (bars.lower_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & (bars.close(2) > bars.close(1))
        & (bars.close(1) > bars.close(0))
        & (bars.open(1) <= bars.close(2) + bars.average(EQUAL, 2))
        & (bars.open(1) >= bars.close(2) - bars.average(EQUAL, 2))
        & (bars.open(0) <= bars.close(1) + bars.average(EQUAL, 1))
        & (bars.open(0) >= bars.close(1) - bars.average(EQUAL, 1))
    )
    return signal(bars, 12, pattern, -100)


def _three_advancing_whites(bars: Bars) -> pl.Expr:
    """Three white bars closing higher, each opening within or near the previous."""
    return (
        bars.white(2)
        & bars.white(1)
        & bars.white(0)
        & (bars.close(0) > bars.close(1))
        & (bars.close(1) > bars.close(2))
        & (bars.open(1) > bars.open(2))
        & (bars.open(1) <= bars.close(2) + bars.average(NEAR, 2))
        & (bars.open(0) > bars.open(1))
        & (bars.open(0) <= bars.close(1) + bars.average(NEAR, 1))
    )


def cdl3whitesoldiers(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Three White Soldiers: three white bars advancing in steady, full strides.

    Each bar opens inside the previous body, closes near its own high, and
    keeps roughly the size of the one before it, so the advance holds its pace.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        _three_advancing_whites(bars)
        & (bars.upper_shadow(2) < bars.average(SHADOW_VERY_SHORT, 2))
        & (bars.upper_shadow(1) < bars.average(SHADOW_VERY_SHORT, 1))
        & (bars.upper_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & (bars.body(1) > bars.body(2) - bars.average(FAR, 2))
        & (bars.body(0) > bars.body(1) - bars.average(FAR, 1))
        & (bars.body(0) > bars.average(BODY_SHORT, 0))
    )
    return signal(bars, 12, pattern, 100)


def cdladvanceblock(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Advance Block: three white bars whose strides shorten as they rise.

    The advance starts on a long bar with little upper shadow, then the later
    bodies shrink or grow upper shadows, so each session gives more back.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    weakening = (
        (
            (bars.body(1) < bars.body(2) - bars.average(FAR, 2))
            & (bars.body(0) < bars.body(1) + bars.average(NEAR, 1))
        )
        | (bars.body(0) < bars.body(1) - bars.average(FAR, 1))
        | (
            (bars.body(0) < bars.body(1))
            & (bars.body(1) < bars.body(2))
            & (
                (bars.upper_shadow(0) > bars.average(SHADOW_SHORT, 0))
                | (bars.upper_shadow(1) > bars.average(SHADOW_SHORT, 1))
            )
        )
        | (
            (bars.body(0) < bars.body(1))
            & (bars.upper_shadow(0) > bars.average(SHADOW_LONG, 0))
        )
    )
    pattern = (
        _three_advancing_whites(bars)
        & (bars.body(2) > bars.average(BODY_LONG, 2))
        & (bars.upper_shadow(2) < bars.average(SHADOW_SHORT, 2))
        & weakening
    )
    return signal(bars, 12, pattern, -100)


def cdlstalledpattern(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Stalled Pattern: two long white bars, then a small one riding on top.

    The third bar opens at the upper end of the second body but covers almost
    no ground of its own, so the advance arrives at a standstill.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.white(2)
        & bars.white(1)
        & bars.white(0)
        & (bars.close(0) > bars.close(1))
        & (bars.close(1) > bars.close(2))
        & (bars.body(2) > bars.average(BODY_LONG, 2))
        & (bars.body(1) > bars.average(BODY_LONG, 1))
        & (bars.upper_shadow(1) < bars.average(SHADOW_VERY_SHORT, 1))
        & (bars.open(1) > bars.open(2))
        & (bars.open(1) <= bars.close(2) + bars.average(NEAR, 2))
        & (bars.body(0) < bars.average(BODY_SHORT, 0))
        & (bars.open(0) >= bars.close(1) - bars.body(0) - bars.average(NEAR, 1))
    )
    return signal(bars, 12, pattern, -100)


def cdlupsidegap2crows(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Upside Gap Two Crows: a gap up met by two black bars, the second wider.

    The small black bar that opens the gap is swallowed by a larger one, which
    still closes above the white bar, leaving the gap open but the advance checked.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.white(2)
        & (bars.body(2) > bars.average(BODY_LONG, 2))
        & bars.black(1)
        & (bars.body(1) <= bars.average(BODY_SHORT, 1))
        & bars.body_gap_up(1, 2)
        & bars.black(0)
        & (bars.open(0) > bars.open(1))
        & (bars.close(0) < bars.close(1))
        & (bars.close(0) > bars.close(2))
    )
    return signal(bars, 12, pattern, -100)
