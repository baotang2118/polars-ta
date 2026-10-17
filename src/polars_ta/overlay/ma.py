"""Moving-average overlays: simple, weighted, and exponential families."""

from __future__ import annotations

import math
from typing import Literal

import polars as pl

from polars_ta._common import (
    IntoColumn,
    to_expr,
    to_exprs,
    validate_alpha,
    validate_window,
)

EmaMode = Literal["talib", "adjust", "recursive"]

_EMA_MODES: frozenset[str] = frozenset(("talib", "adjust", "recursive"))


def _sma_expr(values: pl.Expr, window: int) -> pl.Expr:
    return values.rolling_mean(window_size=window, min_samples=window)


def _wma_expr(values: pl.Expr, window: int) -> pl.Expr:
    weights = [float(weight) for weight in range(1, window + 1)]
    # Weighted rolling_mean panics on nulls, so fill them and mask the result instead.
    weighted = values.cast(pl.Float64).fill_null(0.0)
    nulls_in_window = (
        values.is_null()
        .cast(pl.UInt32)
        .rolling_sum(window_size=window, min_samples=window)
    )
    return (
        pl.when(nulls_in_window > 0)
        .then(None)
        .otherwise(
            weighted.rolling_mean(
                window_size=window, weights=weights, min_samples=window
            )
        )
    )


def _ema_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    if mode != "talib":
        return values.ewm_mean(
            alpha=alpha,
            adjust=mode == "adjust",
            ignore_nulls=False,
            min_samples=window,
        )

    rolling = _sma_expr(values, window)
    complete_windows = rolling.is_not_null().cum_sum()
    seeded = (
        pl.when(complete_windows == 0)
        .then(None)
        .when(rolling.is_not_null() & (complete_windows == 1))
        .then(rolling)
        .otherwise(values)
    )
    return seeded.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False, min_samples=1)


def _hma_expr(values: pl.Expr, window: int) -> pl.Expr:
    half = max(window // 2, 1)
    root = max(int(math.sqrt(window)), 1)
    return _wma_expr(2.0 * _wma_expr(values, half) - _wma_expr(values, window), root)


def _zlema_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    lag = (window - 1) // 2
    return _ema_expr(2.0 * values - values.shift(lag), window, alpha, mode)


def _vwma_expr(values: pl.Expr, volume: pl.Expr, window: int) -> pl.Expr:
    weighted = (values * volume).rolling_sum(window_size=window, min_samples=window)
    traded = volume.rolling_sum(window_size=window, min_samples=window)
    # A window that traded nothing has no volume-weighted price, so it stays null.
    return (
        pl.when(weighted.is_null() | traded.is_null() | (traded == 0.0))
        .then(None)
        .otherwise(weighted / traded)
    )


def _dema_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    first = _ema_expr(values, window, alpha, mode)
    second = _ema_expr(first, window, alpha, mode)
    return 2.0 * first - second


def _tema_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    first = _ema_expr(values, window, alpha, mode)
    second = _ema_expr(first, window, alpha, mode)
    third = _ema_expr(second, window, alpha, mode)
    return 3.0 * first - 3.0 * second + third


def _trima_periods(window: int) -> tuple[int, int]:
    """The two SMA passes whose convolution is the triangular weight ramp."""
    half = window // 2
    if window % 2 == 1:
        return half + 1, half + 1
    return half + 1, half


def _trima_expr(values: pl.Expr, window: int) -> pl.Expr:
    first, second = _trima_periods(window)
    return _sma_expr(_sma_expr(values, first), second)


def _t3_expr(
    values: pl.Expr, window: int, alpha: float, mode: EmaMode, vfactor: float
) -> pl.Expr:
    stage = values
    passes = []
    for _ in range(6):
        stage = _ema_expr(stage, window, alpha, mode)
        passes.append(stage)
    third, fourth, fifth, sixth = passes[2:]
    squared = vfactor * vfactor
    cubed = squared * vfactor
    return (
        -cubed * sixth
        + (3.0 * squared + 3.0 * cubed) * fifth
        + (-6.0 * squared - 3.0 * vfactor - 3.0 * cubed) * fourth
        + (1.0 + 3.0 * vfactor + cubed + 3.0 * squared) * third
    )


def _resolve_alpha(window: int, alpha: float | None, mode: EmaMode) -> float:
    """Validate the shared EMA arguments and return the effective smoothing factor."""
    validate_window(window)
    if alpha is None:
        alpha = 2.0 / (window + 1.0)
    else:
        validate_alpha(alpha)
    if mode not in _EMA_MODES:
        raise ValueError(f"mode must be one of {sorted(_EMA_MODES)}, got {mode!r}")
    return alpha


def sma(column: IntoColumn, window: int) -> pl.Expr:
    """Simple moving average: the arithmetic mean of the last ``window`` values.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the averaging window; must be at least 1.

    Returns:
        A ``pl.Expr``. The first ``window - 1`` rows are null, as is any row
        whose window contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _sma_expr(to_expr(column), window)


def ema(
    column: IntoColumn,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr:
    """Exponential moving average weighting recent values most heavily.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods; also sets the smoothing factor and warm-up.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: Seeding convention. ``"talib"`` seeds the recursion with the
            simple moving average of the first complete window, matching
            TA-Lib. ``"recursive"`` and ``"adjust"`` match pandas
            ``ewm(adjust=False)`` and ``ewm(adjust=True)`` respectively.

    Returns:
        A ``pl.Expr``. The first ``window - 1`` rows are null in every mode.

    Raises:
        ValueError: If ``window``, ``alpha``, or ``mode`` is invalid.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    return _ema_expr(to_expr(column), window, smoothing, mode)


def wma(column: IntoColumn, window: int) -> pl.Expr:
    """Weighted moving average with linearly decaying weights ``window..1``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the averaging window; must be at least 1.

    Returns:
        A ``pl.Expr``. The first ``window - 1`` rows are null, as is any row
        whose window contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _wma_expr(to_expr(column), window)


def hma(column: IntoColumn, window: int = 9) -> pl.Expr:
    """Hull Moving Average: a weighted average built to cut lag, not just noise.

    A half-length weighted average is doubled and the full-length one
    subtracted, which overshoots the current price; a final weighted average
    over ``sqrt(window)`` periods smooths that overshoot back out.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the longer weighted average; must be at
            least 2. The two other passes use ``window // 2`` and
            ``int(sqrt(window))`` periods.

    Returns:
        A ``pl.Expr``. The first ``window + int(sqrt(window)) - 2`` rows are
        null, as is any row whose windows contain a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 2.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    if window < 2:
        raise ValueError(f"window must be >= 2 for a Hull average, got {window}")
    return _hma_expr(to_expr(column), window)


def zlema(
    column: IntoColumn,
    window: int = 10,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr:
    """Zero-Lag Exponential Moving Average: an EMA of a de-lagged series.

    An EMA of period ``n`` lags price by about ``(n - 1) / 2`` bars. Adding
    that lag's worth of recent change back onto each bar before smoothing
    cancels most of it, at the cost of a sharper reaction to single outliers.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods; also sets the smoothing factor and the
            ``(window - 1) // 2`` bar de-lagging shift.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: EMA seeding convention; see :func:`ema`.

    Returns:
        A ``pl.Expr``. The first ``(window - 1) // 2 + window - 1`` rows are null.

    Raises:
        ValueError: If ``window``, ``alpha``, or ``mode`` is invalid.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    return _zlema_expr(to_expr(column), window, smoothing, mode)


def wilders(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Wilder's smoothing: the exponential average behind RSI, ATR, and ADX.

    Wilder's recursion carries ``1 / window`` of each new value instead of the
    usual ``2 / (window + 1)``, so it reacts about half as fast as an EMA of
    the same length. It is :func:`ema` with ``alpha=1 / window``, named here
    because the indicators that use it rarely spell that out.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods; also sets the smoothing factor and warm-up.

    Returns:
        A ``pl.Expr``. The first ``window - 1`` rows are null. The recursion is
        seeded with the simple moving average of the first complete window.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _ema_expr(to_expr(column), window, 1.0 / window, "talib")


def vwma(close: IntoColumn, volume: IntoColumn, window: int = 20) -> pl.Expr:
    """Volume Weighted Moving Average: each close weighted by its own volume.

    Unlike :func:`~polars_ta.volume.vwap`, which averages the typical price,
    this tracks the close, so it can be read against a plain :func:`sma` of the
    same length to see whether the heavily traded bars sat above or below the
    unweighted mean.

    Args:
        close: Column name or expression of closing prices.
        volume: Column name or expression of traded volume.
        window: Number of periods averaged.

    Returns:
        A ``pl.Expr`` yielding a price. The first ``window - 1`` rows are null,
        as is any window that traded no volume.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    close_expr, volume_expr = to_exprs(close, volume)
    return _vwma_expr(close_expr, volume_expr, window)


def dema(
    column: IntoColumn,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr:
    """Double exponential moving average: ``2 * EMA - EMA(EMA)``.

    Removes most of the lag of a plain EMA by subtracting the residual lag of a
    second EMA applied to the first.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods used for both EMA passes.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: EMA seeding convention; see :func:`ema`.

    Returns:
        A ``pl.Expr``. The first ``2 * (window - 1)`` rows are null.

    Raises:
        ValueError: If ``window``, ``alpha``, or ``mode`` is invalid.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    return _dema_expr(to_expr(column), window, smoothing, mode)


def tema(
    column: IntoColumn,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr:
    """Triple exponential moving average: ``3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods used for all three EMA passes.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: EMA seeding convention; see :func:`ema`.

    Returns:
        A ``pl.Expr``. The first ``3 * (window - 1)`` rows are null.

    Raises:
        ValueError: If ``window``, ``alpha``, or ``mode`` is invalid.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    return _tema_expr(to_expr(column), window, smoothing, mode)


def trima(column: IntoColumn, window: int = 30) -> pl.Expr:
    """Triangular moving average: an SMA of an SMA, weighting the window centre.

    The weights rise linearly to the middle of the window and fall away again,
    which smooths more heavily than an SMA of the same length at the cost of
    extra lag.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods spanned by the triangular weights.

    Returns:
        A ``pl.Expr``. The first ``window - 1`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _trima_expr(to_expr(column), window)


def t3(
    column: IntoColumn,
    window: int = 5,
    *,
    vfactor: float = 0.7,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr:
    """Tillson's T3: a weighted blend of the third through sixth EMA passes.

    ``vfactor`` controls how much lag is cancelled. At ``0.0`` the result is
    the plain triple-smoothed EMA; at ``1.0`` it reduces to a TEMA of a TEMA,
    which is faster but overshoots more.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods used for all six EMA passes.
        vfactor: Volume factor in ``[0, 1]`` weighting the lag cancellation.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: EMA seeding convention; see :func:`ema`.

    Returns:
        A ``pl.Expr``. The first ``6 * (window - 1)`` rows are null.

    Raises:
        ValueError: If ``window``, ``vfactor``, ``alpha``, or ``mode`` is invalid.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    if isinstance(vfactor, bool) or not isinstance(vfactor, (int, float)):
        raise ValueError(f"vfactor must be a float, got {type(vfactor).__name__}")
    if not 0.0 <= vfactor <= 1.0:
        raise ValueError(f"vfactor must satisfy 0 <= vfactor <= 1, got {vfactor}")
    return _t3_expr(to_expr(column), window, smoothing, mode, float(vfactor))
