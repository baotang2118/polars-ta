"""Gap patterns: bodies that jump clear of the previous bar, and what follows."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn
from polars_ta.pattern._candles import BODY_LONG, EQUAL, NEAR, Bars, signal


def cdlgapsidesidewhite(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Up/Down-gap Side-by-Side White Lines: twin white bars across one gap.

    Both white bars gap the same way from the bar before them, and they are
    close enough in size and opening price to look like a matched pair, so
    the gap stays open rather than being filled.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` after an upside gap, ``-100`` after a
        downside gap, and ``0`` otherwise. The first 7 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    gap_up = bars.body_gap_up(1, 2) & bars.body_gap_up(0, 2)
    gap_down = bars.body_gap_down(1, 2) & bars.body_gap_down(0, 2)
    near = bars.average(NEAR, 1)
    equal = bars.average(EQUAL, 1)
    pattern = (
        (gap_up | gap_down)
        & bars.white(1)
        & bars.white(0)
        & (bars.body(0) >= bars.body(1) - near)
        & (bars.body(0) <= bars.body(1) + near)
        & (bars.open(0) >= bars.open(1) - equal)
        & (bars.open(0) <= bars.open(1) + equal)
    )
    value = pl.when(bars.body_gap_up(1, 2)).then(100).otherwise(-100)
    return signal(bars, 7, pattern, value)


def cdltasukigap(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Tasuki Gap: a gap, then an opposite bar that probes it without closing it.

    The second bar opens inside the body that made the gap and runs back
    through it, but stops short of the far side, leaving the gap intact.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` for an upside gap, ``-100`` for a
        downside gap, and ``0`` otherwise. The first 7 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    similar = (bars.body(1) - bars.body(0)).abs() < bars.average(NEAR, 1)
    upside = (
        bars.body_gap_up(1, 2)
        & bars.white(1)
        & bars.black(0)
        & (bars.open(0) < bars.close(1))
        & (bars.open(0) > bars.open(1))
        & (bars.close(0) < bars.open(1))
        & (bars.close(0) > bars.body_high(2))
    )
    downside = (
        bars.body_gap_down(1, 2)
        & bars.black(1)
        & bars.white(0)
        & (bars.open(0) < bars.open(1))
        & (bars.open(0) > bars.close(1))
        & (bars.close(0) > bars.open(1))
        & (bars.close(0) < bars.body_low(2))
    )
    pattern = (upside | downside) & similar
    return signal(bars, 7, pattern, bars.color(1) * 100)


def cdlxsidegap3methods(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Upside/Downside Gap Three Methods: a gap bridged by one opposite bar.

    Two same-coloured bars open a gap between their bodies, then a single
    opposite bar spans it, opening inside the second body and closing inside
    the first.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` for an upside gap, ``-100`` for a
        downside gap, and ``0`` otherwise. The first 2 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    gap = (bars.white(2) & bars.body_gap_up(1, 2)) | (
        bars.black(2) & bars.body_gap_down(1, 2)
    )
    pattern = (
        (bars.color(2) == bars.color(1))
        & (bars.color(1) == -bars.color(0))
        & (bars.open(0) < bars.body_high(1))
        & (bars.open(0) > bars.body_low(1))
        & (bars.close(0) < bars.body_high(2))
        & (bars.close(0) > bars.body_low(2))
        & gap
    )
    return signal(bars, 2, pattern, bars.color(2) * 100)


def cdlbreakaway(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Breakaway: a gap, three bars drifting on, then one bar back into the gap.

    A long bar gaps away and the next three continue in the same direction
    without filling it, until a final opposite bar closes back inside the
    gap and undoes the move.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a bullish match, ``-100`` on a
        bearish one, and ``0`` otherwise. The first 14 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    bullish = (
        bars.black(4)
        & bars.body_gap_down(3, 4)
        & (bars.high(2) < bars.high(3))
        & (bars.low(2) < bars.low(3))
        & (bars.high(1) < bars.high(2))
        & (bars.low(1) < bars.low(2))
        & (bars.close(0) > bars.open(3))
        & (bars.close(0) < bars.close(4))
    )
    bearish = (
        bars.white(4)
        & bars.body_gap_up(3, 4)
        & (bars.high(2) > bars.high(3))
        & (bars.low(2) > bars.low(3))
        & (bars.high(1) > bars.high(2))
        & (bars.low(1) > bars.low(2))
        & (bars.close(0) < bars.open(3))
        & (bars.close(0) > bars.close(4))
    )
    pattern = (
        (bars.body(4) > bars.average(BODY_LONG, 4))
        & (bars.color(4) == bars.color(3))
        & (bars.color(3) == bars.color(1))
        & (bars.color(1) == -bars.color(0))
        & (bullish | bearish)
    )
    return signal(bars, 14, pattern, bars.color(0) * 100)
