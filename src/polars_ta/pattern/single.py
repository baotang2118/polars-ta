"""Single-bar patterns read from one candle's body and shadows."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn
from polars_ta.pattern._candles import (
    BODY_LONG,
    BODY_SHORT,
    NEAR,
    SHADOW_LONG,
    SHADOW_SHORT,
    SHADOW_VERY_LONG,
    SHADOW_VERY_SHORT,
    Bars,
    signal,
)


def _small_body_long_lower_shadow(bars: Bars) -> pl.Expr:
    return (
        (bars.body(0) < bars.average(BODY_SHORT, 0))
        & (bars.lower_shadow(0) > bars.average(SHADOW_LONG, 0))
        & (bars.upper_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
    )


def _small_body_long_upper_shadow(bars: Bars) -> pl.Expr:
    return (
        (bars.body(0) < bars.average(BODY_SHORT, 0))
        & (bars.upper_shadow(0) > bars.average(SHADOW_LONG, 0))
        & (bars.lower_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
    )


def cdlbelthold(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Belt-hold: a long bar that opens at one extreme of its range.

    The session opens at its low and rises, or opens at its high and falls, so
    one side held the bar from the first trade to the last.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a white bar, ``-100`` on a black one,
        and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    very_short = bars.average(SHADOW_VERY_SHORT, 0)
    pattern = (bars.body(0) > bars.average(BODY_LONG, 0)) & (
        (bars.white(0) & (bars.lower_shadow(0) < very_short))
        | (bars.black(0) & (bars.upper_shadow(0) < very_short))
    )
    return signal(bars, 10, pattern, bars.color(0) * 100)


def cdlclosingmarubozu(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Closing Marubozu: a long bar that closes at one extreme of its range.

    Whatever ground was given up during the session was recovered by the close,
    which lands on the bar's high or its low.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a white bar, ``-100`` on a black one,
        and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    very_short = bars.average(SHADOW_VERY_SHORT, 0)
    pattern = (bars.body(0) > bars.average(BODY_LONG, 0)) & (
        (bars.white(0) & (bars.upper_shadow(0) < very_short))
        | (bars.black(0) & (bars.lower_shadow(0) < very_short))
    )
    return signal(bars, 10, pattern, bars.color(0) * 100)


def cdlmarubozu(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Marubozu: a long bar that is all body and no shadow.

    Price moved in one direction from the open to the close without trading
    beyond either end of the body.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a white bar, ``-100`` on a black one,
        and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    very_short = bars.average(SHADOW_VERY_SHORT, 0)
    pattern = (
        (bars.body(0) > bars.average(BODY_LONG, 0))
        & (bars.upper_shadow(0) < very_short)
        & (bars.lower_shadow(0) < very_short)
    )
    return signal(bars, 10, pattern, bars.color(0) * 100)


def cdllongline(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Long Line Candle: a long body with short shadows at both ends.

    Most of the session's range was covered between the open and the close,
    with little trading outside that span.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a white bar, ``-100`` on a black one,
        and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    short = bars.average(SHADOW_SHORT, 0)
    pattern = (
        (bars.body(0) > bars.average(BODY_LONG, 0))
        & (bars.upper_shadow(0) < short)
        & (bars.lower_shadow(0) < short)
    )
    return signal(bars, 10, pattern, bars.color(0) * 100)


def cdlshortline(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Short Line Candle: a small body with short shadows at both ends.

    The whole session stayed within a narrow range, so the bar records little
    activity in either direction.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a white bar, ``-100`` on a black one,
        and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    short = bars.average(SHADOW_SHORT, 0)
    pattern = (
        (bars.body(0) < bars.average(BODY_SHORT, 0))
        & (bars.upper_shadow(0) < short)
        & (bars.lower_shadow(0) < short)
    )
    return signal(bars, 10, pattern, bars.color(0) * 100)


def cdlhighwave(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """High-Wave Candle: a small body between two very long shadows.

    Price travelled far above and far below the body during the session and
    came back, so the range was wide but the net move was not.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a white bar, ``-100`` on a black one,
        and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    very_long = bars.average(SHADOW_VERY_LONG, 0)
    pattern = (
        (bars.body(0) < bars.average(BODY_SHORT, 0))
        & (bars.upper_shadow(0) > very_long)
        & (bars.lower_shadow(0) > very_long)
    )
    return signal(bars, 10, pattern, bars.color(0) * 100)


def cdlspinningtop(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Spinning Top: a small body with a shadow longer than it on each side.

    Both sides pushed price away from the body and neither held the gain, so
    the session ended close to where it began.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a white bar, ``-100`` on a black one,
        and ``0`` otherwise. The first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(0) < bars.average(BODY_SHORT, 0))
        & (bars.upper_shadow(0) > bars.body(0))
        & (bars.lower_shadow(0) > bars.body(0))
    )
    return signal(bars, 10, pattern, bars.color(0) * 100)


def cdlhammer(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Hammer: a small body at the top of the range, near the prior low.

    Price was sold well below the previous bar's low and bought back before the
    close, leaving a long lower shadow and a body near the session's high.

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
    pattern = _small_body_long_lower_shadow(bars) & (
        bars.body_low(0) <= bars.low(1) + bars.average(NEAR, 1)
    )
    return signal(bars, 11, pattern, 100)


def cdlhangingman(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Hanging Man: a small body at the top of the range, near the prior high.

    The shape matches a hammer, but the body sits near the previous bar's high
    rather than its low, so the long lower shadow follows an advance.

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
    pattern = _small_body_long_lower_shadow(bars) & (
        bars.body_low(0) >= bars.high(1) - bars.average(NEAR, 1)
    )
    return signal(bars, 11, pattern, -100)


def cdlinvertedhammer(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Inverted Hammer: a small body at the bottom of the range, after a gap down.

    The bar opens below the previous body, rallies far above it, and gives the
    rally back by the close, leaving a long upper shadow.

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
    pattern = _small_body_long_upper_shadow(bars) & bars.body_gap_down(0, 1)
    return signal(bars, 11, pattern, 100)


def cdlshootingstar(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Shooting Star: a small body at the bottom of the range, after a gap up.

    The bar opens above the previous body and runs higher still, then returns
    to the bottom of its range by the close.

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
    pattern = _small_body_long_upper_shadow(bars) & bars.body_gap_up(0, 1)
    return signal(bars, 11, pattern, -100)
