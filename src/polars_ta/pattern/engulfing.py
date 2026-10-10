"""Two-bar reversals where the second bar answers the first."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, validate_positive
from polars_ta.pattern._candles import (
    BODY_DOJI,
    BODY_LONG,
    BODY_SHORT,
    EQUAL,
    SHADOW_VERY_SHORT,
    Bars,
    signal,
)


def cdlengulfing(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Engulfing Pattern: a bar whose body swallows the previous one's.

    The second bar opens beyond one end of the first bar's body and closes
    beyond the other, so a full session of the opposite colour is reversed in
    a single day.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` when a white bar engulfs a black one,
        ``-100`` when a black bar engulfs a white one, and ``0`` otherwise.
        The first 2 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.white(0)
        & bars.black(1)
        & (bars.close(0) > bars.open(1))
        & (bars.open(0) < bars.close(1))
    ) | (
        bars.black(0)
        & bars.white(1)
        & (bars.open(0) > bars.close(1))
        & (bars.close(0) < bars.open(1))
    )
    return signal(bars, 2, pattern, bars.color(0) * 100)


def cdlharami(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Harami: a small body held inside the previous long one.

    After a wide session the next one trades entirely within its body, so the
    move that produced the long bar has stopped extending.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` after a black first bar, ``-100`` after
        a white one, and ``0`` otherwise. The first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(1) > bars.average(BODY_LONG, 1))
        & (bars.body(0) <= bars.average(BODY_SHORT, 0))
        & (bars.body_high(0) < bars.body_high(1))
        & (bars.body_low(0) > bars.body_low(1))
    )
    return signal(bars, 11, pattern, -bars.color(1) * 100)


def cdlharamicross(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Harami Cross: a harami whose inside bar is a doji.

    The contained session not only stays within the long body but also closes
    where it opened, leaving the standoff sharper than in a plain harami.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` after a black first bar, ``-100`` after
        a white one, and ``0`` otherwise. The first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(1) > bars.average(BODY_LONG, 1))
        & (bars.body(0) <= bars.average(BODY_DOJI, 0))
        & (bars.body_high(0) < bars.body_high(1))
        & (bars.body_low(0) > bars.body_low(1))
    )
    return signal(bars, 11, pattern, -bars.color(1) * 100)


def cdldarkcloudcover(
    open_: IntoColumn,
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    penetration: float = 0.5,
) -> pl.Expr:
    """Dark Cloud Cover: a black bar opening above a long white bar's high.

    The session starts above everything the previous one reached and still
    closes deep inside its body, giving back more than ``penetration`` of the
    advance it had made.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        penetration: Fraction of the white body the close must give back.

    Returns:
        A ``pl.Expr`` yielding ``-100`` on a match and ``0`` otherwise. The
        first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
        ValueError: If ``penetration`` is not a positive finite number.
    """
    validate_positive("penetration", penetration)
    bars = Bars.of(open_, high, low, close)
    pattern = (
        bars.white(1)
        & (bars.body(1) > bars.average(BODY_LONG, 1))
        & bars.black(0)
        & (bars.open(0) > bars.high(1))
        & (bars.close(0) > bars.open(1))
        & (bars.close(0) < bars.close(1) - bars.body(1) * penetration)
    )
    return signal(bars, 11, pattern, -100)


def cdlpiercing(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Piercing Pattern: a long white bar opening below a long black bar's low.

    The session opens under everything the previous one reached and closes
    back above the midpoint of its body, recovering more than half of the
    decline.

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
        & (bars.body(1) > bars.average(BODY_LONG, 1))
        & bars.white(0)
        & (bars.body(0) > bars.average(BODY_LONG, 0))
        & (bars.open(0) < bars.low(1))
        & (bars.close(0) < bars.open(1))
        & (bars.close(0) > bars.close(1) + bars.body(1) * 0.5)
    )
    return signal(bars, 11, pattern, 100)


def cdlcounterattack(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Counterattack: two long bars of opposite colour closing at the same price.

    The second session moves as far as the first did but the other way, and
    ends exactly where the first ended, cancelling it out.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` when the second bar is white, ``-100``
        when it is black, and ``0`` otherwise. The first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    equal = bars.average(EQUAL, 1)
    pattern = (
        (bars.color(1) == -bars.color(0))
        & (bars.body(1) > bars.average(BODY_LONG, 1))
        & (bars.body(0) > bars.average(BODY_LONG, 0))
        & (bars.close(0) <= bars.close(1) + equal)
        & (bars.close(0) >= bars.close(1) - equal)
    )
    return signal(bars, 11, pattern, bars.color(0) * 100)


def cdlseparatinglines(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Separating Lines: a belt hold opening where the opposite bar opened.

    The second session starts at the same price as the first but runs the
    other way, opening at one extreme of its range and closing near the
    other.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` when the second bar is white, ``-100``
        when it is black, and ``0`` otherwise. The first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    equal = bars.average(EQUAL, 1)
    very_short = bars.average(SHADOW_VERY_SHORT, 0)
    pattern = (
        (bars.color(1) == -bars.color(0))
        & (bars.open(0) <= bars.open(1) + equal)
        & (bars.open(0) >= bars.open(1) - equal)
        & (bars.body(0) > bars.average(BODY_LONG, 0))
        & (
            (bars.white(0) & (bars.lower_shadow(0) < very_short))
            | (bars.black(0) & (bars.upper_shadow(0) < very_short))
        )
    )
    return signal(bars, 11, pattern, bars.color(0) * 100)


def _kicking(bars: Bars) -> pl.Expr:
    """Two opposite marubozu separated by a gap in the second bar's direction."""
    return (
        (bars.color(1) == -bars.color(0))
        & (bars.body(1) > bars.average(BODY_LONG, 1))
        & (bars.upper_shadow(1) < bars.average(SHADOW_VERY_SHORT, 1))
        & (bars.lower_shadow(1) < bars.average(SHADOW_VERY_SHORT, 1))
        & (bars.body(0) > bars.average(BODY_LONG, 0))
        & (bars.upper_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & (bars.lower_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & ((bars.black(1) & bars.gap_up(0, 1)) | (bars.white(1) & bars.gap_down(0, 1)))
    )


def cdlkicking(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Kicking: two shadowless bars of opposite colour separated by a gap.

    Each session runs one way from open to close with almost no shadow, and
    the second one starts clear of the first's range, so price changes
    direction without trading in between.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` when the second bar is white, ``-100``
        when it is black, and ``0`` otherwise. The first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    return signal(bars, 11, _kicking(bars), bars.color(0) * 100)


def cdlkickingbylength(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Kicking by Length: a kicking scored by the longer of its two bars.

    The pattern is the same as a kicking, but the direction reported is that
    of whichever of the two sessions covered more ground rather than that of
    the later one.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` when the longer bar is white, ``-100``
        when it is black, and ``0`` otherwise. The first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    longer = (
        pl.when(bars.body(0) > bars.body(1))
        .then(bars.color(0))
        .otherwise(bars.color(1))
    )
    return signal(bars, 11, _kicking(bars), longer * 100)


def cdlmatchinglow(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Matching Low: two black bars closing at the same price.

    A second declining session stops at exactly the level where the first
    one stopped, marking that price as one sellers could not push through.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 6 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    equal = bars.average(EQUAL, 1)
    pattern = (
        bars.black(1)
        & bars.black(0)
        & (bars.close(0) <= bars.close(1) + equal)
        & (bars.close(0) >= bars.close(1) - equal)
    )
    return signal(bars, 6, pattern, 100)
