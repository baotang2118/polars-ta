"""Bar builders for the candlestick formation tests.

A formation is written as the handful of bars that make up the pattern. Those
bars are laid after a run of identical baseline bars, which is what gives the
averaged candle settings something to measure against: after the baseline a
body of more than 1.0 is "long" and one of 1.0 or less is "short", a body of
0.2 or less is a doji's, a shadow under 0.2 is "very short", "near" is 0.4,
"far" 1.2, and "equal" 0.1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl

Bar = tuple[float, float, float, float]
"""One bar, as ``(open, high, low, close)``."""

BASELINE: Bar = (100.0, 101.5, 99.5, 101.0)
BASELINE_BARS: int = 20


def frame_of(bars: Sequence[Bar]) -> pl.DataFrame:
    """An OHLC frame of ``bars`` preceded by the baseline run."""
    rows = [BASELINE] * BASELINE_BARS + list(bars)
    return pl.DataFrame(
        {
            "open": [bar[0] for bar in rows],
            "high": [bar[1] for bar in rows],
            "low": [bar[2] for bar in rows],
            "close": [bar[3] for bar in rows],
        }
    )


def score(
    recogniser: Callable[..., pl.Expr],
    bars: Sequence[Bar],
    index: int = -1,
    **options: float,
) -> int | None:
    """What ``recogniser`` reports at ``index`` within the formation."""
    expr = recogniser("open", "high", "low", "close", **options)
    values = frame_of(bars).select(expr.alias("out")).to_series().to_list()
    return values[BASELINE_BARS:][index]
