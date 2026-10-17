"""Ehlers' Fisher Transform."""

from __future__ import annotations

import math

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window
from polars_ta.overlay.transform import _medprice_expr

FISHER_FIELDS = ("fisher", "signal")

# Ehlers' own weights: a third of the new reading against two thirds of the old.
_POSITION_WEIGHT = 0.33
_CARRY_WEIGHT = 0.67
# The transform diverges at +/-1, so the smoothed position stops just short.
_CLAMP = 0.999

_RETURN_DTYPE = pl.Struct({"fisher": pl.Float64, "signal": pl.Float64})


def _fisher_scan(positions: pl.Series) -> pl.Series:
    """Smooth the scaled position, then run it through the transform."""
    values = positions.to_list()

    fishers: list[float | None] = [None] * len(values)
    position = 0.0
    previous = 0.0
    for index, value in enumerate(values):
        if value is None:
            position = 0.0
            previous = 0.0
            continue
        position = _POSITION_WEIGHT * value + _CARRY_WEIGHT * position
        position = max(-_CLAMP, min(_CLAMP, position))
        previous = 0.5 * math.log((1.0 + position) / (1.0 - position)) + 0.5 * previous
        fishers[index] = previous
    signals: list[float | None] = [None, *fishers[:-1]]
    return pl.Series(
        values=[{"fisher": f, "signal": s} for f, s in zip(fishers, signals)],
        dtype=_RETURN_DTYPE,
    )


def _fisher_expr(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    median = _medprice_expr(high, low)
    lowest = median.rolling_min(window_size=window, min_samples=window)
    highest = median.rolling_max(window_size=window, min_samples=window)
    span = highest - lowest
    # A flat window has no position to read, so it contributes nothing.
    position = (
        pl.when(span.is_null() | median.is_null())
        .then(None)
        .when(span > 0.0)
        .then(2.0 * ((median - lowest) / span - 0.5))
        .otherwise(0.0)
    )
    return position.map_batches(_fisher_scan, return_dtype=_RETURN_DTYPE)


def fisher(high: IntoColumn, low: IntoColumn, window: int = 9) -> pl.Expr:
    """Fisher Transform: the bar's place in its range, stretched towards normality.

    Price spends most of its time near the middle of a range, so a plain
    stochastic reading clusters there and extremes are hard to call. Mapping
    the scaled position through ``0.5 * ln((1 + x) / (1 - x))`` pulls the
    middle apart and sends the edges off towards infinity, which turns a turn
    at an extreme into a sharp, unambiguous spike.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        window: Number of periods in the high-low range.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``fisher`` and ``signal``,
        the latter being the previous bar's ``fisher``. The first
        ``window - 1`` rows are null, and ``signal`` is null for one row more.
        The smoothed position is clamped to ``+/-0.999`` so the transform
        stays finite, and a flat window reads as the range's midpoint.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    high_expr, low_expr = to_exprs(high, low)
    return _fisher_expr(high_expr, low_expr, window)
