"""Moving-average overlays: simple and exponential moving averages."""

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
    validate_window(window)
    if alpha is None:
        alpha = 2.0 / (window + 1.0)
    else:
        validate_alpha(alpha)
    if mode not in _EMA_MODES:
        raise ValueError(f"mode must be one of {sorted(_EMA_MODES)}, got {mode!r}")
    return apply_to_column(
        column, lambda values: _ema_expr(values, window, alpha, mode)
    )
