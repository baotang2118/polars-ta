"""Adaptive moving averages, whose smoothing factor varies bar by bar."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_alpha, validate_window
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


def _vidya_scan(inputs: pl.Series) -> pl.Series:
    """Run the VIDYA recursion, seeding with the first bar that has a factor."""
    values = inputs.struct.field("value").to_list()
    factors = inputs.struct.field("factor").to_list()

    result: list[float | None] = [None] * len(values)
    previous: float | None = None
    for index in range(len(values)):
        if factors[index] is None or values[index] is None:
            previous = None
            continue
        if previous is None:
            previous = values[index]
        else:
            previous += factors[index] * (values[index] - previous)
        result[index] = previous
    return pl.Series(values=result, dtype=_RETURN_DTYPE)


def _vidya_expr(
    values: pl.Expr, short_period: int, long_period: int, alpha: float
) -> pl.Expr:
    short_std = values.rolling_std(
        window_size=short_period, min_samples=short_period, ddof=0
    )
    long_std = values.rolling_std(
        window_size=long_period, min_samples=long_period, ddof=0
    )
    # A flat long window cannot say the short one is volatile, so nothing adapts.
    ratio = (
        pl.when(short_std.is_null() | long_std.is_null())
        .then(None)
        .when(long_std != 0.0)
        .then(short_std / long_std)
        .otherwise(0.0)
    )
    return pl.struct(value=values, factor=alpha * ratio).map_batches(
        _vidya_scan, return_dtype=_RETURN_DTYPE
    )


def vidya(
    column: IntoColumn,
    short_period: int = 12,
    long_period: int = 26,
    *,
    alpha: float = 0.2,
) -> pl.Expr:
    """Chande's Variable Index Dynamic Average, which speeds up when volatility does.

    The smoothing factor is ``alpha`` scaled by the ratio of a short standard
    deviation to a long one. A burst of volatility pushes that ratio above one
    and lets the average track price closely; a quiet stretch pulls it down and
    the average flattens.

    Args:
        column: Column name or expression holding the input values.
        short_period: Number of periods in the faster standard deviation.
        long_period: Number of periods in the slower standard deviation; also
            sets the warm-up.
        alpha: Base smoothing factor the volatility ratio scales.

    Returns:
        A ``pl.Expr``. The first ``long_period - 1`` rows are null; the first
        live row is the input value itself, which seeds the recursion. A long
        window with no dispersion holds the average flat rather than dividing,
        and a null input ends the recursion.

    Raises:
        ValueError: If a period is invalid, ``short_period`` exceeds
            ``long_period``, or ``alpha`` is outside ``(0, 1]``.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(short_period)
    validate_window(long_period)
    validate_alpha(alpha)
    if short_period > long_period:
        raise ValueError(
            "short_period must not exceed long_period, "
            f"got {short_period} > {long_period}"
        )
    return _vidya_expr(to_expr(column), short_period, long_period, float(alpha))
