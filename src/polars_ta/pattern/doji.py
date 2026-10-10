"""Doji bars, where the open and close meet, and their named variants."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn
from polars_ta.pattern._candles import (
    BODY_DOJI,
    BODY_LONG,
    NEAR,
    SHADOW_LONG,
    SHADOW_VERY_LONG,
    SHADOW_VERY_SHORT,
    Bars,
    signal,
)


def cdldoji(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Doji: a bar that closes where it opened.

    Buyers and sellers finished the session where they started it, so the bar
    records a standoff rather than a direction.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = bars.body(0) <= bars.average(BODY_DOJI, 0)
    return signal(bars, 10, pattern, 100)


def cdldojistar(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Doji Star: a long bar, then a doji gapping away from it.

    The move that drove the long bar leaves a gap and then stalls, the doji
    showing that the side in control no longer is.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``-100`` after a white first bar, ``100`` after
        a black one, and ``0`` otherwise. The first 11 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(1) > bars.average(BODY_LONG, 1))
        & (bars.body(0) <= bars.average(BODY_DOJI, 0))
        & (
            (bars.white(1) & bars.body_gap_up(0, 1))
            | (bars.black(1) & bars.body_gap_down(0, 1))
        )
    )
    return signal(bars, 11, pattern, -bars.color(1) * 100)


def cdldragonflydoji(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Dragonfly Doji: a doji that opens and closes at the high of the day.

    Price was sold far down during the session and recovered all of it by the
    close, leaving a long lower shadow and almost no upper one.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(0) <= bars.average(BODY_DOJI, 0))
        & (bars.upper_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & (bars.lower_shadow(0) > bars.average(SHADOW_VERY_SHORT, 0))
    )
    return signal(bars, 10, pattern, 100)


def cdlgravestonedoji(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Gravestone Doji: a doji that opens and closes at the low of the day.

    Price was bought far up during the session and gave all of it back by the
    close, leaving a long upper shadow and almost no lower one.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(0) <= bars.average(BODY_DOJI, 0))
        & (bars.lower_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & (bars.upper_shadow(0) > bars.average(SHADOW_VERY_SHORT, 0))
    )
    return signal(bars, 10, pattern, 100)


def cdllongleggeddoji(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Long Legged Doji: a doji with at least one long shadow.

    Price ranged widely during the session and still returned to its opening
    level, so the day's movement settled nothing.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (bars.body(0) <= bars.average(BODY_DOJI, 0)) & (
        (bars.lower_shadow(0) > bars.average(SHADOW_LONG, 0))
        | (bars.upper_shadow(0) > bars.average(SHADOW_LONG, 0))
    )
    return signal(bars, 10, pattern, 100)


def cdlrickshawman(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Rickshaw Man: a doji with two long shadows and its body centred.

    Price swung as far one way as the other and closed where it opened, in the
    middle of the range, with neither side ending the session ahead.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    midpoint = bars.low(0) + bars.hl_range(0) / 2
    near = bars.average(NEAR, 0)
    pattern = (
        (bars.body(0) <= bars.average(BODY_DOJI, 0))
        & (bars.lower_shadow(0) > bars.average(SHADOW_LONG, 0))
        & (bars.upper_shadow(0) > bars.average(SHADOW_LONG, 0))
        & (bars.body_low(0) <= midpoint + near)
        & (bars.body_high(0) >= midpoint - near)
    )
    return signal(bars, 10, pattern, 100)


def cdltakuri(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Takuri: a dragonfly doji whose lower shadow is very long.

    The same opening and closing at the high of the day, but the session's dip
    reached much further down before it was bought back.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding ``100`` on a match and ``0`` otherwise. The
        first 10 rows are null.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    bars = Bars.of(open_, high, low, close)
    pattern = (
        (bars.body(0) <= bars.average(BODY_DOJI, 0))
        & (bars.upper_shadow(0) < bars.average(SHADOW_VERY_SHORT, 0))
        & (bars.lower_shadow(0) > bars.average(SHADOW_VERY_LONG, 0))
    )
    return signal(bars, 10, pattern, 100)
