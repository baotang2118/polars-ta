"""Moving-average overlays: simple, weighted, and exponential families."""

from __future__ import annotations

from typing import Literal, overload

import polars as pl

from polars_ta._common import (
    IntoColumn,
    apply_to_column,
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


def _dema_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    first = _ema_expr(values, window, alpha, mode)
    second = _ema_expr(first, window, alpha, mode)
    return 2.0 * first - second


def _tema_expr(values: pl.Expr, window: int, alpha: float, mode: EmaMode) -> pl.Expr:
    first = _ema_expr(values, window, alpha, mode)
    second = _ema_expr(first, window, alpha, mode)
    third = _ema_expr(second, window, alpha, mode)
    return 3.0 * first - 3.0 * second + third


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


@overload
def sma(column: str | pl.Expr, window: int) -> pl.Expr: ...


@overload
def sma(column: pl.Series, window: int) -> pl.Series: ...


def sma(column: IntoColumn, window: int) -> pl.Expr | pl.Series:
    """Simple moving average: the arithmetic mean of the last ``window`` values.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods in the averaging window; must be at least 1.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window - 1`` rows are null, as is any row whose window
        contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(column, lambda values: _sma_expr(values, window))


@overload
def ema(
    column: str | pl.Expr,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr: ...


@overload
def ema(
    column: pl.Series,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Series: ...


def ema(
    column: IntoColumn,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr | pl.Series:
    """Exponential moving average weighting recent values most heavily.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods; also sets the smoothing factor and warm-up.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: Seeding convention. ``"talib"`` seeds the recursion with the
            simple moving average of the first complete window, matching
            TA-Lib. ``"recursive"`` and ``"adjust"`` match pandas
            ``ewm(adjust=False)`` and ``ewm(adjust=True)`` respectively.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window - 1`` rows are null in every mode.

    Raises:
        ValueError: If ``window``, ``alpha``, or ``mode`` is invalid.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    return apply_to_column(
        column, lambda values: _ema_expr(values, window, smoothing, mode)
    )


@overload
def wma(column: str | pl.Expr, window: int) -> pl.Expr: ...


@overload
def wma(column: pl.Series, window: int) -> pl.Series: ...


def wma(column: IntoColumn, window: int) -> pl.Expr | pl.Series:
    """Weighted moving average with linearly decaying weights ``window..1``.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods in the averaging window; must be at least 1.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window - 1`` rows are null, as is any row whose window
        contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(column, lambda values: _wma_expr(values, window))


@overload
def dema(
    column: str | pl.Expr,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr: ...


@overload
def dema(
    column: pl.Series,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Series: ...


def dema(
    column: IntoColumn,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr | pl.Series:
    """Double exponential moving average: ``2 * EMA - EMA(EMA)``.

    Removes most of the lag of a plain EMA by subtracting the residual lag of a
    second EMA applied to the first.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods used for both EMA passes.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: EMA seeding convention; see :func:`ema`.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``2 * (window - 1)`` rows are null.

    Raises:
        ValueError: If ``window``, ``alpha``, or ``mode`` is invalid.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    return apply_to_column(
        column, lambda values: _dema_expr(values, window, smoothing, mode)
    )


@overload
def tema(
    column: str | pl.Expr,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr: ...


@overload
def tema(
    column: pl.Series,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Series: ...


def tema(
    column: IntoColumn,
    window: int,
    *,
    alpha: float | None = None,
    mode: EmaMode = "talib",
) -> pl.Expr | pl.Series:
    """Triple exponential moving average: ``3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))``.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods used for all three EMA passes.
        alpha: Smoothing factor overriding the default ``2 / (window + 1)``.
        mode: EMA seeding convention; see :func:`ema`.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``3 * (window - 1)`` rows are null.

    Raises:
        ValueError: If ``window``, ``alpha``, or ``mode`` is invalid.
    """
    smoothing = _resolve_alpha(window, alpha, mode)
    return apply_to_column(
        column, lambda values: _tema_expr(values, window, smoothing, mode)
    )
