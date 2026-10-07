"""Shared market data for the indicator test suite.

Every test module draws its bars from here, so indicators are always compared
against one another on identical data. Three datasets are published, each for a
distinct reason:

* ``CLOSE``/``HIGH``/``LOW``/``VOLUME`` - the canonical series, long enough to
  exercise every indicator's default parameters.
* ``HAND_CHECKED`` - a short series whose indicator values were worked out by
  hand; kept verbatim so those assertions stay genuine checks rather than
  change detectors.
* ``WILDER_CLOSE`` - Wilder's published worked example, an external anchor for
  RSI.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

LENGTH: int = 120

_SEED_CLOSE: list[float] = [
    9.0,
    10.0,
    11.0,
    10.0,
    9.0,
    10.0,
    11.0,
    12.0,
    11.0,
    10.0,
    12.0,
    13.0,
    11.0,
    10.0,
    9.0,
    11.0,
]


def _walk(seed: list[float], count: int) -> list[float]:
    """Extend a series with a reproducible pseudo-random walk."""
    state = 20240219
    values = list(seed)
    for _ in range(count):
        state = (1103515245 * state + 12345) % 2147483648
        step = (state / 2147483648.0 - 0.5) * 5.0
        values.append(round(max(values[-1] + step, 1.0), 2))
    return values


CLOSE: list[float] = _walk(_SEED_CLOSE, LENGTH - len(_SEED_CLOSE))

# Bars are symmetric about the close, which keeps the typical price equal to it
# and so keeps the MFI and CCI hand-checks tractable.
_SPREAD: list[float] = [1.0 + (index % 4) * 0.25 for index in range(LENGTH)]

HIGH: list[float] = [round(close + spread, 4) for close, spread in zip(CLOSE, _SPREAD)]
LOW: list[float] = [round(close - spread, 4) for close, spread in zip(CLOSE, _SPREAD)]
# Opens sit inside each bar's range on a repeating 0.5 / 0.0 / -0.5 offset.
OPEN: list[float] = [
    round(close + spread * (0.5 - (index % 3) * 0.5), 4)
    for index, (close, spread) in enumerate(zip(CLOSE, _SPREAD))
]
VOLUME: list[float] = [float(100 * (1 + index % 9) + 50) for index in range(LENGTH)]

HAND_CHECKED: list[float] = [
    1.0,
    3.0,
    2.0,
    6.0,
    5.0,
    9.0,
    4.0,
    8.0,
    7.0,
    11.0,
    6.0,
    10.0,
    9.0,
    13.0,
    8.0,
    12.0,
]

# Wilder, "New Concepts in Technical Trading Systems"; RSI(14) first value is 70.4641.
WILDER_CLOSE: list[float] = [
    44.34,
    44.09,
    44.15,
    43.61,
    44.33,
    44.83,
    45.10,
    45.42,
    45.84,
    46.08,
    45.89,
    46.03,
    45.61,
    46.28,
    46.28,
    46.00,
    46.03,
    46.41,
    46.22,
    45.64,
]


def frame(
    high: list | None = None,
    low: list | None = None,
    close: list | None = None,
    volume: list | None = None,
    open_: list | None = None,
) -> pl.DataFrame:
    """Build an OHLCV frame, substituting the canonical data for any omitted column.

    Shorter replacement columns truncate the canonical ones to match, so a test
    can pass a handful of synthetic bars without supplying every column.
    """
    supplied = [
        column for column in (high, low, close, volume, open_) if column is not None
    ]
    size = min((len(column) for column in supplied), default=LENGTH)
    if size > LENGTH:
        raise ValueError(f"replacement columns may hold at most {LENGTH} bars")
    return pl.DataFrame(
        {
            "open": OPEN[:size] if open_ is None else open_,
            "high": HIGH[:size] if high is None else high,
            "low": LOW[:size] if low is None else low,
            "close": CLOSE[:size] if close is None else close,
            "volume": VOLUME[:size] if volume is None else volume,
        }
    )


def frame_from(closes: Sequence[float | None], spread: float = 1.0) -> pl.DataFrame:
    """Build an OHLCV frame whose bars are centred on the given close series."""
    return pl.DataFrame(
        {
            "open": list(closes),
            "high": [None if c is None else c + spread for c in closes],
            "low": [None if c is None else c - spread for c in closes],
            "close": list(closes),
            "volume": [100.0] * len(closes),
        }
    )


def ramp_up(count: int, start: float = 1.0) -> list[float]:
    """A strictly increasing series."""
    return [start + index for index in range(count)]


def ramp_down(count: int, start: float | None = None) -> list[float]:
    """A strictly decreasing series."""
    top = float(count) if start is None else start
    return [top - index for index in range(count)]


def constant(count: int, value: float = 5.0) -> list[float]:
    """A series that never moves."""
    return [value] * count


def with_null(values: list[float], *indices: int) -> list[float | None]:
    """Copy a series with the given positions blanked out."""
    result: list[float | None] = list(values)
    for index in indices:
        result[index] = None
    return result
