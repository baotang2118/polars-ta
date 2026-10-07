"""The directional movement family: DM, DI, DX, ADX, and ADXR."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window
from polars_ta.overlay.ma import ema
from polars_ta.volatility.atr import _true_range_expr

ADX_FIELDS = ("adx", "plus_di", "minus_di")


def _directional_moves(high: pl.Expr, low: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Raw +DM and -DM: a bar counts toward only the larger outward move."""
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus = (
        pl.when(up_move.is_null())
        .then(None)
        .when((up_move > down_move) & (up_move > 0.0))
        .then(up_move)
        .otherwise(0.0)
    )
    minus = (
        pl.when(down_move.is_null())
        .then(None)
        .when((down_move > up_move) & (down_move > 0.0))
        .then(down_move)
        .otherwise(0.0)
    )
    return plus, minus


def _wilder_sum(values: pl.Expr, window: int) -> pl.Expr:
    """TA-Lib's running Wilder sum, seeded with ``window - 1`` terms.

    The seed deliberately holds one term fewer than the window, which is what
    gives ``plus_dm`` its ``window - 1`` lookback rather than ``window``.
    """
    if window == 1:
        return values
    smoothing = 1.0 / window
    seed = values.rolling_sum(window_size=window - 1, min_samples=window - 1)
    complete = seed.is_not_null().cum_sum()
    seeded = (
        pl.when(complete == 0)
        .then(None)
        .when(seed.is_not_null() & (complete == 1))
        .then(seed * smoothing)
        .otherwise(values)
    )
    running = seeded.ewm_mean(
        alpha=smoothing, adjust=False, ignore_nulls=False, min_samples=1
    )
    return window * running


def _di_expr(dm_sum: pl.Expr, tr_sum: pl.Expr, window: int) -> pl.Expr:
    known = dm_sum.is_not_null() & tr_sum.is_not_null()
    if window > 1:
        # TA-Lib emits the first DI one bar after the running sums begin.
        known = known & tr_sum.shift(1).is_not_null()
    return (
        pl.when(~known)
        .then(None)
        .when(tr_sum > 0.0)
        .then(100.0 * dm_sum / tr_sum)
        .otherwise(0.0)
    )


def _di_pair(
    high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int
) -> tuple[pl.Expr, pl.Expr]:
    plus, minus = _directional_moves(high, low)
    ranges = _wilder_sum(_true_range_expr(high, low, close), window)
    return (
        _di_expr(_wilder_sum(plus, window), ranges, window),
        _di_expr(_wilder_sum(minus, window), ranges, window),
    )


def _dx_expr(positive: pl.Expr, negative: pl.Expr) -> pl.Expr:
    total = positive + negative
    return (
        pl.when(total.is_null())
        .then(None)
        .when(total > 0.0)
        .then(100.0 * (positive - negative).abs() / total)
        .otherwise(0.0)
    )


def _adx_line(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    return ema(
        _dx_expr(*_di_pair(high, low, close, window)), window, alpha=1.0 / window
    )


def _adx_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    positive, negative = _di_pair(high, low, close, window)
    return pl.struct(
        adx=ema(_dx_expr(positive, negative), window, alpha=1.0 / window),
        plus_di=positive,
        minus_di=negative,
    )


def plus_dm(high: IntoColumn, low: IntoColumn, window: int = 14) -> pl.Expr:
    """Plus Directional Movement: the Wilder-smoothed running sum of ``+DM``.

    A bar contributes its gain in highs only when that gain exceeds the
    matching loss in lows, so at most one of ``+DM`` and ``-DM`` is non-zero.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        window: Number of periods in Wilder's smoothing.

    Returns:
        A ``pl.Expr`` yielding a non-negative running sum in price units. The
        first ``window - 1`` rows are null, matching TA-Lib, whose seed holds
        one term fewer than the window.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _wilder_sum(_directional_moves(*to_exprs(high, low))[0], window)


def minus_dm(high: IntoColumn, low: IntoColumn, window: int = 14) -> pl.Expr:
    """Minus Directional Movement: the Wilder-smoothed running sum of ``-DM``.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        window: Number of periods in Wilder's smoothing.

    Returns:
        A ``pl.Expr`` yielding a non-negative running sum in price units. The
        first ``window - 1`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _wilder_sum(_directional_moves(*to_exprs(high, low))[1], window)


def plus_di(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Plus Directional Indicator: ``+DM`` as a percentage of the true range.

    Normalizing by range makes the reading comparable across instruments and
    volatility regimes, which the raw ``+DM`` sum is not.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods in Wilder's smoothing.

    Returns:
        A ``pl.Expr`` yielding a value in ``[0, 100]``. The first ``window``
        rows are null, and a window with no range at all reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _di_pair(*to_exprs(high, low, close), window)[0]


def minus_di(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Minus Directional Indicator: ``-DM`` as a percentage of the true range.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods in Wilder's smoothing.

    Returns:
        A ``pl.Expr`` yielding a value in ``[0, 100]``. The first ``window``
        rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _di_pair(*to_exprs(high, low, close), window)[1]


def dx(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Directional Movement Index: the normalized gap between the two indicators.

    The unsmoothed input to :func:`adx`, and correspondingly jumpier.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods in Wilder's smoothing.

    Returns:
        A ``pl.Expr`` yielding a value in ``[0, 100]``. The first ``window``
        rows are null, and a bar where neither indicator moved reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _dx_expr(*_di_pair(*to_exprs(high, low, close), window))


def adx(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Average Directional Index with its two Directional Indicators.

    ``plus_di`` and ``minus_di`` measure the strength of upward and downward
    movement; ``adx`` smooths the normalized gap between them into a single
    trend-strength reading that ignores direction.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods for all three Wilder-smoothing passes.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``adx``, ``plus_di``, and
        ``minus_di``, each in ``[0, 100]``. The indicators start after
        ``window`` rows and ``adx`` after ``2 * window - 1``, matching the
        separate TA-Lib lookbacks.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _adx_expr(*to_exprs(high, low, close), window)


def adxr(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Average Directional Index Rating: today's ADX averaged with an older one.

    Averaging across ``window - 1`` bars smooths the ADX further, which makes
    turns in trend strength easier to read at the cost of more lag.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        window: Number of periods for the underlying ADX and for the lookback.

    Returns:
        A ``pl.Expr`` yielding a value in ``[0, 100]``. The first
        ``3 * window - 2`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    line = _adx_line(*to_exprs(high, low, close), window)
    return (line + line.shift(window - 1)) / 2.0
