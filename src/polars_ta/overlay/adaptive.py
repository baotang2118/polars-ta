"""Adaptive moving averages, whose smoothing factor varies bar by bar."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window
from polars_ta._hilbert import SHORT_LOOKBACK, hilbert_transform, mask_lookback

MAMA_FIELDS = ("mama", "fama")

_RETURN_DTYPE = pl.Float64
_MAMA_DTYPE = pl.Struct({"mama": pl.Float64, "fama": pl.Float64})


def _kama_scan(inputs: pl.Series) -> pl.Series:
    """Run the KAMA recursion, whose smoothing factor changes every bar."""
    values = inputs.struct.field("value").to_list()
    factors = inputs.struct.field("factor").to_list()
    seeds = inputs.struct.field("seed").to_list()

    result: list[float | None] = [None] * len(values)
    previous: float | None = None
    for index in range(len(values)):
        if factors[index] is None or values[index] is None:
            previous = None
            continue
        if previous is None:
            previous = seeds[index]
            if previous is None:
                continue
        previous += factors[index] * (values[index] - previous)
        result[index] = previous
    return pl.Series(values=result, dtype=_RETURN_DTYPE)


def _kama_expr(
    values: pl.Expr, window: int, fast_period: int, slow_period: int
) -> pl.Expr:
    volatility = values.diff().abs().rolling_sum(window_size=window, min_samples=window)
    direction = values - values.shift(window)
    # TA-Lib treats a move at least as large as the path that produced it as
    # perfectly efficient, which also covers the zero-volatility case.
    efficiency = (
        pl.when(volatility.is_null() | direction.is_null())
        .then(None)
        .when((volatility <= direction) | (volatility == 0.0))
        .then(1.0)
        .otherwise((direction / volatility).abs())
    )
    fastest = 2.0 / (fast_period + 1.0)
    slowest = 2.0 / (slow_period + 1.0)
    smoothing = (efficiency * (fastest - slowest) + slowest) ** 2
    return pl.struct(value=values, factor=smoothing, seed=values.shift(1)).map_batches(
        _kama_scan, return_dtype=_RETURN_DTYPE
    )


def kama(
    column: IntoColumn,
    window: int = 30,
    *,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pl.Expr:
    """Kaufman's Adaptive Moving Average, which speeds up in a trending market.

    Each bar's efficiency ratio compares the net move over ``window`` periods
    against the total distance travelled. A straight-line move scores ``1`` and
    smooths with the ``fast_period`` factor; a directionless one scores ``0``
    and smooths with the ``slow_period`` factor.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the efficiency-ratio lookback.
        fast_period: Period behind the fastest permitted smoothing factor.
        slow_period: Period behind the slowest permitted smoothing factor.

    Returns:
        A ``pl.Expr``. The first ``window`` rows are null; the recursion is
        seeded with the value immediately before the first output.

    Raises:
        ValueError: If any period is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    validate_window(fast_period)
    validate_window(slow_period)
    return _kama_expr(to_expr(column), window, fast_period, slow_period)


def _mama_expr(values: pl.Expr, fast_limit: float, slow_limit: float) -> pl.Expr:
    def scan(column: pl.Series) -> pl.Series:
        series = hilbert_transform(column.to_list(), fast_limit, slow_limit)
        fast = mask_lookback(series.mama, SHORT_LOOKBACK)
        slow = mask_lookback(series.fama, SHORT_LOOKBACK)
        return pl.Series(
            values=[{"mama": m, "fama": f} for m, f in zip(fast, slow)],
            dtype=_MAMA_DTYPE,
        )

    return values.map_batches(scan, return_dtype=_MAMA_DTYPE)


def _validate_limit(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float, got {type(value).__name__}")
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} <= 1, got {value}")


def mama(
    column: IntoColumn, *, fast_limit: float = 0.5, slow_limit: float = 0.05
) -> pl.Expr:
    """Ehlers' MESA Adaptive Moving Average and its following average.

    The smoothing factor is driven by how fast the Hilbert transform's phase is
    turning: a sharp turn in phase marks a new trend and lets the average jump,
    while a steady phase slows it to ``slow_limit``. ``fama`` is a second,
    half-speed pass whose crossings with ``mama`` are the usual signal.

    Args:
        column: Column name or expression holding the input values.
        fast_limit: Largest smoothing factor the average may use.
        slow_limit: Smallest smoothing factor the average may use.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``mama`` and ``fama``, both
        in price units. The first 32 rows are null, and a null input ends the
        recursion.

    Raises:
        ValueError: If a limit is outside ``(0, 1]``.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    _validate_limit("fast_limit", fast_limit)
    _validate_limit("slow_limit", slow_limit)
    return _mama_expr(to_expr(column), float(fast_limit), float(slow_limit))
