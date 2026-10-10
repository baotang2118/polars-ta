"""Star patterns, where a small gapped body interrupts a run of long ones."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, validate_positive
from polars_ta.pattern._candles import (
    BODY_DOJI,
    BODY_LONG,
    BODY_SHORT,
    SHADOW_LONG,
    SHADOW_VERY_SHORT,
    Bars,
    signal,
)


def cdlmorningstar(
    open_: IntoColumn,
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    penetration: float = 0.3,
) -> pl.Expr:
    """Morning Star: a long black bar, a small gapped-down body, then a rally.

    The decline ends in a bar too small to extend it, and the next session
    opens higher and closes back inside the black bar's body.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        penetration: Fraction of the first body the third bar must close
            beyond, matching TA-Lib's ``optInPenetration``.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
        ValueError: If ``penetration`` is not a positive finite number.
    """
    validate_positive("penetration", penetration)
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(2) > bars.average(BODY_LONG, 2))
        & bars.black(2)
        & (bars.body(1) <= bars.average(BODY_SHORT, 1))
        & bars.body_gap_down(1, 2)
        & (bars.body(0) > bars.average(BODY_SHORT, 0))
        & bars.white(0)
        & (bars.close(0) > bars.close(2) + bars.body(2) * penetration)
    )
    return signal(bars, 12, pattern, 100)


def cdleveningstar(
    open_: IntoColumn,
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    penetration: float = 0.3,
) -> pl.Expr:
    """Evening Star: a long white bar, a small gapped-up body, then a decline.

    The advance ends in a bar too small to extend it, and the next session
    opens lower and closes back inside the white bar's body.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        penetration: Fraction of the first body the third bar must close
            beyond, matching TA-Lib's ``optInPenetration``.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
        ValueError: If ``penetration`` is not a positive finite number.
    """
    validate_positive("penetration", penetration)
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(2) > bars.average(BODY_LONG, 2))
        & bars.white(2)
        & (bars.body(1) <= bars.average(BODY_SHORT, 1))
        & bars.body_gap_up(1, 2)
        & (bars.body(0) > bars.average(BODY_SHORT, 0))
        & bars.black(0)
        & (bars.close(0) < bars.close(2) - bars.body(2) * penetration)
    )
    return signal(bars, 12, pattern, -100)


def cdlmorningdojistar(
    open_: IntoColumn,
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    penetration: float = 0.3,
) -> pl.Expr:
    """Morning Doji Star: a Morning Star whose middle bar is a doji.

    The middle session gaps below the long black bar and closes where it
    opened, so the decline pauses on an even balance before the rebound.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        penetration: Fraction of the first body the third bar must close
            beyond, matching TA-Lib's ``optInPenetration``.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
        ValueError: If ``penetration`` is not a positive finite number.
    """
    validate_positive("penetration", penetration)
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(2) > bars.average(BODY_LONG, 2))
        & bars.black(2)
        & (bars.body(1) <= bars.average(BODY_DOJI, 1))
        & bars.body_gap_down(1, 2)
        & (bars.body(0) > bars.average(BODY_SHORT, 0))
        & bars.white(0)
        & (bars.close(0) > bars.close(2) + bars.body(2) * penetration)
    )
    return signal(bars, 12, pattern, 100)


def cdleveningdojistar(
    open_: IntoColumn,
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    penetration: float = 0.3,
) -> pl.Expr:
    """Evening Doji Star: an Evening Star whose middle bar is a doji.

    The middle session gaps above the long white bar and closes where it
    opened, so the advance pauses on an even balance before the setback.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        penetration: Fraction of the first body the third bar must close
            beyond, matching TA-Lib's ``optInPenetration``.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
        ValueError: If ``penetration`` is not a positive finite number.
    """
    validate_positive("penetration", penetration)
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(2) > bars.average(BODY_LONG, 2))
        & bars.white(2)
        & (bars.body(1) <= bars.average(BODY_DOJI, 1))
        & bars.body_gap_up(1, 2)
        & (bars.body(0) > bars.average(BODY_SHORT, 0))
        & bars.black(0)
        & (bars.close(0) < bars.close(2) - bars.body(2) * penetration)
    )
    return signal(bars, 12, pattern, -100)


def cdlabandonedbaby(
    open_: IntoColumn,
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    penetration: float = 0.3,
) -> pl.Expr:
    """Abandoned Baby: a doji isolated by gaps between two long opposite bars.

    Neither shadow of the doji touches its neighbours, so the session that
    reversed the move stands entirely apart from the bars on either side.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        penetration: Fraction of the first body the third bar must close
            beyond, matching TA-Lib's ``optInPenetration``.

    Returns:
        A ``pl.Expr`` yielding ``100`` after a black first bar, ``-100``
        after a white one, and ``0`` otherwise. The first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
        ValueError: If ``penetration`` is not a positive finite number.
    """
    validate_positive("penetration", penetration)
    bars = Bars.of(open_, high, low, close)
    top = (
        bars.white(2)
        & bars.black(0)
        & (bars.close(0) < bars.close(2) - bars.body(2) * penetration)
        & bars.gap_up(1, 2)
        & bars.gap_down(0, 1)
    )
    bottom = (
        bars.black(2)
        & bars.white(0)
        & (bars.close(0) > bars.close(2) + bars.body(2) * penetration)
        & bars.gap_down(1, 2)
        & bars.gap_up(0, 1)
    )
    pattern = (
        (bars.body(2) > bars.average(BODY_LONG, 2))
        & (bars.body(1) <= bars.average(BODY_DOJI, 1))
        & (bars.body(0) > bars.average(BODY_SHORT, 0))
        & (top | bottom)
    )
    return signal(bars, 12, pattern, bars.color(0) * 100)


def cdltristar(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Tristar: three doji in a row, the middle one gapping away.

    Three sessions each close where they opened, and the middle one trades
    clear of its neighbours' bodies, so the gap settles nothing either.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` when the middle doji gaps up, ``100``
        when it gaps down, and ``0`` otherwise. The first 12 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    doji = bars.average(BODY_DOJI, 2)
    pattern = (bars.body(2) <= doji) & (bars.body(1) <= doji) & (bars.body(0) <= doji)
    direction = (
        pl.when(bars.body_gap_up(1, 2) & (bars.body_high(0) < bars.body_high(1)))
        .then(-100)
        .when(bars.body_gap_down(1, 2) & (bars.body_low(0) > bars.body_low(1)))
        .then(100)
        .otherwise(0)
    )
    return signal(bars, 12, pattern, direction)


def cdl3starsinsouth(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Three Stars In The South: three black bars of shrinking range.

    Each session opens above the last close, probes lower, and gives up less
    ground than the one before, until the third is a small bar held entirely
    inside the second's range.

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
    very_short = bars.average(SHADOW_VERY_SHORT, 0)
    pattern = (
        bars.black(2)
        & bars.black(1)
        & bars.black(0)
        & (bars.body(2) > bars.average(BODY_LONG, 2))
        & (bars.lower_shadow(2) > bars.average(SHADOW_LONG, 2))
        & (bars.body(1) < bars.body(2))
        & (bars.open(1) > bars.close(2))
        & (bars.open(1) <= bars.high(2))
        & (bars.low(1) < bars.close(2))
        & (bars.low(1) >= bars.low(2))
        & (bars.lower_shadow(1) > bars.average(SHADOW_VERY_SHORT, 1))
        & (bars.body(0) < bars.average(BODY_SHORT, 0))
        & (bars.lower_shadow(0) < very_short)
        & (bars.upper_shadow(0) < very_short)
        & (bars.low(0) > bars.low(1))
        & (bars.high(0) < bars.high(1))
    )
    return signal(bars, 12, pattern, 100)
