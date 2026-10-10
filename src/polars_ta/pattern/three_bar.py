"""Three-bar reversals built from engulfing, strike, and swallow shapes."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn
from polars_ta.pattern._candles import (
    BODY_LONG,
    BODY_SHORT,
    EQUAL,
    NEAR,
    SHADOW_VERY_SHORT,
    Bars,
    signal,
)


def cdl3inside(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Three Inside Up/Down: a harami, then a bar that closes past its first.

    A long bar is followed by a short one wholly inside it, and the third bar
    closes beyond the long bar's open, resolving the pause against the side
    that set the long bar.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` for three inside up, ``-100`` for three
        inside down, and ``0`` otherwise. The first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(2) > bars.average(BODY_LONG, 2))
        & (bars.body(1) <= bars.average(BODY_SHORT, 1))
        & (bars.body_high(1) < bars.body_high(2))
        & (bars.body_low(1) > bars.body_low(2))
        & (
            (bars.white(2) & bars.black(0) & (bars.close(0) < bars.open(2)))
            | (bars.black(2) & bars.white(0) & (bars.close(0) > bars.open(2)))
        )
    )
    return signal(bars, 12, pattern, -bars.color(2) * 100)


def cdl3outside(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Three Outside Up/Down: an engulfing pair, then a bar that extends it.

    The second bar's body swallows the first bar's body in the opposite
    colour, and the third bar closes further in the engulfing direction,
    confirming that the takeover held.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` for three outside up, ``-100`` for
        three outside down, and ``0`` otherwise. The first 3 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    up = (
        bars.white(1)
        & bars.black(2)
        & (bars.close(1) > bars.open(2))
        & (bars.open(1) < bars.close(2))
        & (bars.close(0) > bars.close(1))
    )
    down = (
        bars.black(1)
        & bars.white(2)
        & (bars.open(1) > bars.close(2))
        & (bars.close(1) < bars.open(2))
        & (bars.close(0) < bars.close(1))
    )
    return signal(bars, 3, up | down, bars.color(1) * 100)


def cdl3linestrike(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Three-Line Strike: three bars one way, then one bar undoing them all.

    Three same-coloured bars extend in one direction, each opening within or
    near its predecessor's body, and the fourth bar opens past the third's
    close and runs back through the whole run to finish beyond the first
    bar's open.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` after three white bars, ``-100`` after
        three black ones, and ``0`` otherwise. The first 8 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    near_third = bars.average(NEAR, 3)
    near_second = bars.average(NEAR, 2)
    white_run = (
        bars.white(1)
        & (bars.close(1) > bars.close(2))
        & (bars.close(2) > bars.close(3))
        & (bars.open(0) > bars.close(1))
        & (bars.close(0) < bars.open(3))
    )
    black_run = (
        bars.black(1)
        & (bars.close(1) < bars.close(2))
        & (bars.close(2) < bars.close(3))
        & (bars.open(0) < bars.close(1))
        & (bars.close(0) > bars.open(3))
    )
    pattern = (
        (bars.color(3) == bars.color(2))
        & (bars.color(2) == bars.color(1))
        & (bars.color(0) == -bars.color(1))
        & (bars.open(2) >= bars.body_low(3) - near_third)
        & (bars.open(2) <= bars.body_high(3) + near_third)
        & (bars.open(1) >= bars.body_low(2) - near_second)
        & (bars.open(1) <= bars.body_high(2) + near_second)
        & (white_run | black_run)
    )
    return signal(bars, 8, pattern, bars.color(1) * 100)


def cdlunique3river(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Unique 3 River: a long black bar, a probing harami, then a small white.

    The second black bar holds inside the first bar's body yet digs out a new
    low, and the small white bar that follows opens above that low, so the
    selling that made the low found no continuation.

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
        (bars.body(2) > bars.average(BODY_LONG, 2))
        & bars.black(2)
        & bars.black(1)
        & (bars.close(1) > bars.close(2))
        & (bars.open(1) <= bars.open(2))
        & (bars.low(1) < bars.low(2))
        & (bars.body(0) < bars.average(BODY_SHORT, 0))
        & bars.white(0)
        & (bars.open(0) > bars.low(1))
    )
    return signal(bars, 12, pattern, 100)


def cdlsticksandwich(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Stick Sandwich: two black bars closing alike around a white one.

    The white middle bar trades entirely above the first bar's close, and the
    third black bar comes back to close at that same level, so the sell-off
    stops twice at one price.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 7 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    tolerance = bars.average(EQUAL, 2)
    pattern = (
        bars.black(2)
        & bars.white(1)
        & bars.black(0)
        & (bars.low(1) > bars.close(2))
        & (bars.close(0) <= bars.close(2) + tolerance)
        & (bars.close(0) >= bars.close(2) - tolerance)
    )
    return signal(bars, 7, pattern, 100)


def cdlconcealbabyswall(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Concealing Baby Swallow: four black bars, the last swallowing the third.

    Two shadowless black bars carry the decline, the third gaps down but
    reaches back up into the second bar's body, and the fourth bar covers the
    third's entire range, so the drop ends inside a single session.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 13 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.black(3)
        & bars.black(2)
        & bars.black(1)
        & bars.black(0)
        & (bars.lower_shadow(3) < bars.average(SHADOW_VERY_SHORT, 3))
        & (bars.upper_shadow(3) < bars.average(SHADOW_VERY_SHORT, 3))
        & (bars.lower_shadow(2) < bars.average(SHADOW_VERY_SHORT, 2))
        & (bars.upper_shadow(2) < bars.average(SHADOW_VERY_SHORT, 2))
        & bars.body_gap_down(1, 2)
        & (bars.upper_shadow(1) > bars.average(SHADOW_VERY_SHORT, 1))
        & (bars.high(1) > bars.close(2))
        & (bars.high(0) > bars.high(1))
        & (bars.low(0) < bars.low(1))
    )
    return signal(bars, 13, pattern, 100)
