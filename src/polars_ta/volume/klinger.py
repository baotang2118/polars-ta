"""Klinger Volume Oscillator."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window
from polars_ta.overlay.ma import ema

_RETURN_DTYPE = pl.Float64


def _volume_force_scan(bars: pl.Series) -> pl.Series:
    """Accumulate the trend's daily ranges, resetting whenever the trend flips."""
    pivot = bars.struct.field("pivot").to_list()
    span = bars.struct.field("span").to_list()
    volume = bars.struct.field("volume").to_list()

    force: list[float | None] = [None] * len(pivot)
    previous = None
    trend = 0
    cumulative = 0.0
    for index in range(len(pivot)):
        if pivot[index] is None or span[index] is None or volume[index] is None:
            previous = None
            trend = 0
            continue
        if previous is None:
            previous = index
            continue
        direction = 1 if pivot[index] > pivot[previous] else -1
        if direction != trend:
            # A new trend restarts the sum from the bar that ended the old one.
            cumulative = span[previous]
            trend = direction
        cumulative += span[index]
        ratio = (
            0.0 if cumulative == 0.0 else abs(2.0 * (span[index] / cumulative - 1.0))
        )
        force[index] = volume[index] * ratio * 100.0 * direction
        previous = index
    return pl.Series(values=force, dtype=_RETURN_DTYPE)


def _kvo_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    fast_period: int,
    slow_period: int,
) -> pl.Expr:
    force = pl.struct(
        pivot=high + low + close, span=high - low, volume=volume
    ).map_batches(_volume_force_scan, return_dtype=_RETURN_DTYPE)
    fast = ema(force, fast_period, mode="recursive")
    slow = ema(force, slow_period, mode="recursive")
    return fast - slow


def kvo(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    volume: IntoColumn,
    fast_period: int = 34,
    slow_period: int = 55,
) -> pl.Expr:
    """Klinger Volume Oscillator: volume signed by trend, then differenced.

    Klinger's volume force signs each bar's volume by whether the high-low-close
    sum rose or fell, and scales it by how the bar's own range compares with the
    ranges accumulated since the trend began. A bar that widens an established
    move counts for far more than one that merely repeats it. The gap between a
    fast and a slow average of that force is the oscillator.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        volume: Column name or expression of traded volume.
        fast_period: Period of the faster exponential average.
        slow_period: Period of the slower exponential average.

    Returns:
        A ``pl.Expr`` in volume units. The first ``slow_period`` rows are null:
        one for the trend comparison and ``slow_period - 1`` for the average.
        Both averages are seeded with the first force reading, as the reference
        implementations do, and a trend with no accumulated range contributes
        nothing rather than dividing.

    Raises:
        ValueError: If a period is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(fast_period)
    validate_window(slow_period)
    high_expr, low_expr, close_expr, volume_expr = to_exprs(high, low, close, volume)
    return _kvo_expr(
        high_expr, low_expr, close_expr, volume_expr, fast_period, slow_period
    )
