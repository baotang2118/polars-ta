"""Shared candlestick geometry, settings, and output conventions.

Candlestick recognisers all describe the same handful of measurements -- how
long a real body is, how short a shadow is, how wide a gap is -- so the raw
geometry lives here once and every pattern module builds on it.

A measurement only means something relative to the bars around it: a "long"
body is long compared with recent bodies. Each :class:`CandleSetting` names one
such comparison, pairing the quantity being measured (:class:`RangeType`) with
the number of preceding bars it is averaged over and the multiple that counts
as a match. The defaults below are TA-Lib's and are not configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import polars as pl

from polars_ta._common import IntoColumn, to_exprs

PATTERN_DTYPE = pl.Int32
"""Dtype every recogniser returns: ``0``, or ``±100``/``±200`` on a match."""


class RangeType(Enum):
    """The quantity a :class:`CandleSetting` measures on a single bar."""

    REAL_BODY = "real_body"
    HIGH_LOW = "high_low"
    SHADOWS = "shadows"


@dataclass(frozen=True)
class CandleSetting:
    """One named comparison: what to measure, over how many bars, times what.

    Args:
        range_type: The per-bar quantity being measured.
        avg_period: Preceding bars averaged over; ``0`` compares with the
            reference bar itself rather than with its predecessors.
        factor: The multiple of that average which marks the boundary.
    """

    range_type: RangeType
    avg_period: int
    factor: float


# A body is long when it exceeds the average body of the 10 bars before it.
BODY_LONG = CandleSetting(RangeType.REAL_BODY, 10, 1.0)
# ... and very long at three times that average.
BODY_VERY_LONG = CandleSetting(RangeType.REAL_BODY, 10, 3.0)
# A body is short when it falls below that same average.
BODY_SHORT = CandleSetting(RangeType.REAL_BODY, 10, 1.0)
# A body is a doji's when it is under a tenth of the recent high-low range.
BODY_DOJI = CandleSetting(RangeType.HIGH_LOW, 10, 0.1)
# A shadow is long when it exceeds its own bar's body, very long at twice it.
SHADOW_LONG = CandleSetting(RangeType.REAL_BODY, 0, 1.0)
SHADOW_VERY_LONG = CandleSetting(RangeType.REAL_BODY, 0, 2.0)
# A shadow is short when under half the recent average of both shadows summed.
SHADOW_SHORT = CandleSetting(RangeType.SHADOWS, 10, 1.0)
# ... and very short under a tenth of the recent high-low range.
SHADOW_VERY_SHORT = CandleSetting(RangeType.HIGH_LOW, 10, 0.1)
# Distances between candles: "near" is 20% of the recent high-low range,
# "far" 60%, and "equal" 5%.
NEAR = CandleSetting(RangeType.HIGH_LOW, 5, 0.2)
FAR = CandleSetting(RangeType.HIGH_LOW, 5, 0.6)
EQUAL = CandleSetting(RangeType.HIGH_LOW, 5, 0.05)


def _shift(expr: pl.Expr, offset: int) -> pl.Expr:
    return expr if offset == 0 else expr.shift(offset)


class Bars:
    """OHLC expressions plus the candlestick measurements taken from them.

    Every method takes an ``offset`` counted backwards from the bar being
    classified, so ``offset=0`` is that bar, ``offset=1`` the one before it.
    """

    __slots__ = ("_close", "_high", "_low", "_open")

    def __init__(
        self, open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr
    ) -> None:
        self._open = open_
        self._high = high
        self._low = low
        self._close = close

    @classmethod
    def of(
        cls, open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
    ) -> Bars:
        """Build from column names or expressions."""
        return cls(*to_exprs(open_, high, low, close))

    def open(self, offset: int = 0) -> pl.Expr:
        return _shift(self._open, offset)

    def high(self, offset: int = 0) -> pl.Expr:
        return _shift(self._high, offset)

    def low(self, offset: int = 0) -> pl.Expr:
        return _shift(self._low, offset)

    def close(self, offset: int = 0) -> pl.Expr:
        return _shift(self._close, offset)

    def body(self, offset: int = 0) -> pl.Expr:
        """Real body: the distance between open and close."""
        return (self.close(offset) - self.open(offset)).abs()

    def body_high(self, offset: int = 0) -> pl.Expr:
        """Top of the real body."""
        return pl.max_horizontal(self.open(offset), self.close(offset))

    def body_low(self, offset: int = 0) -> pl.Expr:
        """Bottom of the real body."""
        return pl.min_horizontal(self.open(offset), self.close(offset))

    def upper_shadow(self, offset: int = 0) -> pl.Expr:
        return self.high(offset) - self.body_high(offset)

    def lower_shadow(self, offset: int = 0) -> pl.Expr:
        return self.body_low(offset) - self.low(offset)

    def hl_range(self, offset: int = 0) -> pl.Expr:
        return self.high(offset) - self.low(offset)

    def white(self, offset: int = 0) -> pl.Expr:
        """Whether the bar closed at or above its open."""
        return self.close(offset) >= self.open(offset)

    def black(self, offset: int = 0) -> pl.Expr:
        return self.close(offset) < self.open(offset)

    def color(self, offset: int = 0) -> pl.Expr:
        """``1`` for a white bar, ``-1`` for a black one."""
        return pl.when(self.white(offset)).then(1).otherwise(-1)

    def gap_up(self, newer: int, older: int) -> pl.Expr:
        """Whether the ``newer`` bar's whole range sits above the ``older``."""
        return self.low(newer) > self.high(older)

    def gap_down(self, newer: int, older: int) -> pl.Expr:
        return self.high(newer) < self.low(older)

    def body_gap_up(self, newer: int, older: int) -> pl.Expr:
        """Whether the ``newer`` bar's real body sits above the ``older``."""
        return self.body_low(newer) > self.body_high(older)

    def body_gap_down(self, newer: int, older: int) -> pl.Expr:
        return self.body_high(newer) < self.body_low(older)

    def _range(self, range_type: RangeType, offset: int) -> pl.Expr:
        if range_type is RangeType.REAL_BODY:
            return self.body(offset)
        if range_type is RangeType.HIGH_LOW:
            return self.hl_range(offset)
        return self.upper_shadow(offset) + self.lower_shadow(offset)

    def average(self, setting: CandleSetting, offset: int = 0) -> pl.Expr:
        """The threshold ``setting`` defines, as seen from bar ``offset``.

        With a non-zero ``avg_period`` the average covers the bars *before*
        the reference bar, never the bar itself, so a pattern is always judged
        against history rather than against its own size.
        """
        if setting.avg_period == 0:
            base = self._range(setting.range_type, offset)
        else:
            rolling = self._range(setting.range_type, 0).rolling_mean(
                window_size=setting.avg_period, min_samples=setting.avg_period
            )
            base = _shift(rolling, offset + 1)
        divisor = 2.0 if setting.range_type is RangeType.SHADOWS else 1.0
        return base * setting.factor / divisor

    def unusable(self, lookback: int) -> pl.Expr:
        """Whether the bar lacks ``lookback`` complete bars of history behind it."""
        window = lookback + 1
        missing = (
            self._open.is_null()
            | self._high.is_null()
            | self._low.is_null()
            | self._close.is_null()
        ).cast(pl.UInt32)
        counted = missing.rolling_sum(window_size=window, min_samples=window)
        return counted.is_null() | (counted > 0)


def signal(
    bars: Bars, lookback: int, condition: pl.Expr, value: int | pl.Expr = 100
) -> pl.Expr:
    """Score a recogniser: ``value`` where ``condition`` holds, ``0`` elsewhere.

    Args:
        bars: The bars the condition was built from.
        lookback: Bars of history the condition needs; those rows report null.
        condition: Whether the pattern completes on each bar.
        value: The score a match carries, as a constant or an expression.
    """
    return (
        pl.when(bars.unusable(lookback))
        .then(None)
        .when(condition)
        .then(value)
        .otherwise(0)
        .cast(PATTERN_DTYPE)
    )
