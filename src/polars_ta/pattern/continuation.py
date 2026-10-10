"""Five-bar patterns where a brief pause interrupts a move that then resumes."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, validate_positive
from polars_ta.pattern._candles import (
    BODY_LONG,
    BODY_SHORT,
    SHADOW_VERY_SHORT,
    Bars,
    signal,
)


def cdlrisefall3methods(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Rising/Falling Three Methods: a long bar, three small ones, another long bar.

    The three small bars run against the first bar's direction but stay inside
    its range, and the fifth bar resumes that direction and closes beyond the
    first bar's close.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a rising match, ``-100`` on a falling
        one, and ``0`` otherwise. The first 14 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    direction = bars.color(4)
    pattern = (
        (bars.body(4) > bars.average(BODY_LONG, 4))
        & (bars.body(3) < bars.average(BODY_SHORT, 3))
        & (bars.body(2) < bars.average(BODY_SHORT, 2))
        & (bars.body(1) < bars.average(BODY_SHORT, 1))
        & (bars.body(0) > bars.average(BODY_LONG, 0))
        & (direction == -bars.color(3))
        & (bars.color(3) == bars.color(2))
        & (bars.color(2) == bars.color(1))
        & (bars.color(1) == -bars.color(0))
        & (bars.body_low(3) < bars.high(4))
        & (bars.body_high(3) > bars.low(4))
        & (bars.body_low(2) < bars.high(4))
        & (bars.body_high(2) > bars.low(4))
        & (bars.body_low(1) < bars.high(4))
        & (bars.body_high(1) > bars.low(4))
        & (bars.close(2) * direction < bars.close(3) * direction)
        & (bars.close(1) * direction < bars.close(2) * direction)
        & (bars.open(0) * direction > bars.close(1) * direction)
        & (bars.close(0) * direction > bars.close(4) * direction)
    )
    return signal(bars, 14, pattern, bars.color(0) * 100)


def cdlmathold(
    open_: IntoColumn,
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    penetration: float = 0.5,
) -> pl.Expr:
    """Mat Hold: a long white bar, a gap up, three small bars, then a new high.

    The three small bars drift down without giving back much of the long white
    body, and the fifth bar closes above the highest of them.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        penetration: Fraction of the first body the small bars may give back,
            matching TA-Lib's ``optInPenetration``.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 14 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
        ValueError: If ``penetration`` is not a positive finite number.
    """
    validate_positive("penetration", penetration)
    bars = Bars.of(open_, high, low, close)
    floor = bars.close(4) - bars.body(4) * penetration
    pattern = (
        (bars.body(4) > bars.average(BODY_LONG, 4))
        & (bars.body(3) < bars.average(BODY_SHORT, 3))
        & (bars.body(2) < bars.average(BODY_SHORT, 2))
        & (bars.body(1) < bars.average(BODY_SHORT, 1))
        & bars.white(4)
        & bars.black(3)
        & bars.white(0)
        & bars.body_gap_up(3, 4)
        & (bars.body_low(2) < bars.close(4))
        & (bars.body_low(1) < bars.close(4))
        & (bars.body_low(2) > floor)
        & (bars.body_low(1) > floor)
        & (bars.body_high(2) < bars.open(3))
        & (bars.body_high(1) < bars.body_high(2))
        & (bars.open(0) > bars.close(1))
        & (bars.close(0) > pl.max_horizontal(bars.high(3), bars.high(2), bars.high(1)))
    )
    return signal(bars, 14, pattern, 100)


def cdlladderbottom(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Ladder Bottom: three stepping-down black bars, a probe higher, then a rally.

    The fourth black bar trades above its open before closing weak, and the
    fifth bar opens above it and closes above its high.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 14 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.black(4)
        & bars.black(3)
        & bars.black(2)
        & (bars.open(4) > bars.open(3))
        & (bars.open(3) > bars.open(2))
        & (bars.close(4) > bars.close(3))
        & (bars.close(3) > bars.close(2))
        & bars.black(1)
        & (bars.upper_shadow(1) > bars.average(SHADOW_VERY_SHORT, 1))
        & bars.white(0)
        & (bars.open(0) > bars.open(1))
        & (bars.close(0) > bars.high(1))
    )
    return signal(bars, 14, pattern, 100)
