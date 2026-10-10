"""Measures of how two series move together."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window

# TA-Lib's dead zone for a denominator that has collapsed to rounding noise.
_EPSILON = 1e-14


def _rolling_sums(
    first: pl.Expr, second: pl.Expr, window: int
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    def total(expr: pl.Expr) -> pl.Expr:
        return expr.rolling_sum(window_size=window, min_samples=window)

    return (
        total(first),
        total(second),
        total(first * first),
        total(second * second),
        total(first * second),
    )


def _correl_expr(first: pl.Expr, second: pl.Expr, window: int) -> pl.Expr:
    sum_x, sum_y, sum_xx, sum_yy, sum_xy = _rolling_sums(first, second, window)
    spread_x = sum_xx - sum_x * sum_x / window
    spread_y = sum_yy - sum_y * sum_y / window
    product = spread_x * spread_y
    covariance = sum_xy - sum_x * sum_y / window
    return (
        pl.when(product.is_null())
        .then(None)
        .when(product < _EPSILON)
        .then(0.0)
        .otherwise(covariance / product.sqrt())
    )


def _returns(values: pl.Expr) -> pl.Expr:
    previous = values.shift(1)
    return (
        pl.when(previous.is_null())
        .then(None)
        .when(previous.abs() < _EPSILON)
        .then(0.0)
        .otherwise(values / previous - 1.0)
    )


def _beta_expr(asset: pl.Expr, market: pl.Expr, window: int) -> pl.Expr:
    # The reference series is the regression's x axis, so its spread divides.
    sum_x, sum_y, sum_xx, _, sum_xy = _rolling_sums(
        _returns(market), _returns(asset), window
    )
    divisor = window * sum_xx - sum_x * sum_x
    return (
        pl.when(divisor.is_null())
        .then(None)
        .when(divisor.abs() < _EPSILON)
        .then(0.0)
        .otherwise((window * sum_xy - sum_x * sum_y) / divisor)
    )


def correl(first: IntoColumn, second: IntoColumn, window: int = 30) -> pl.Expr:
    """Pearson's Correlation Coefficient: how tightly two series move together.

    Measures the strength of a straight-line relationship only, and says
    nothing about its slope: ``1`` and ``-1`` mean the points fall exactly on
    a rising or falling line, whatever its steepness.

    Args:
        first: Column name or expression of the first series.
        second: Column name or expression of the second series.
        window: Number of periods in the window, including the current row.

    Returns:
        A ``pl.Expr`` yielding a value in ``[-1, 1]``. The first
        ``window - 1`` rows are null, and a window in which either series
        never moves reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    first_expr, second_expr = to_exprs(first, second)
    return _correl_expr(first_expr, second_expr, window)


def beta(asset: IntoColumn, market: IntoColumn, window: int = 5) -> pl.Expr:
    """Beta: how far an asset moves for each move of a reference series.

    Both series are reduced to simple returns, and the slope of a line fitted
    through the resulting pairs is reported. ``1`` means the asset matched the
    reference move for move; above ``1`` it amplified it, below ``1`` it
    damped it.

    Args:
        asset: Column name or expression of the asset's prices.
        market: Column name or expression of the reference prices.
        window: Number of returns in the window.

    Returns:
        A ``pl.Expr`` yielding a slope. The first ``window`` rows are null,
        since one bar is consumed turning prices into returns, and a window in
        which the reference never moves reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    asset_expr, market_expr = to_exprs(asset, market)
    return _beta_expr(asset_expr, market_expr, window)
