"""Least-squares lines fitted to a rolling window, and the values read off them."""

from __future__ import annotations

import math

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window

_DEGREES_PER_RADIAN = 180.0 / math.pi


def _validate_regression_window(window: int) -> None:
    validate_window(window)
    if window < 2:
        raise ValueError(f"window must be >= 2 to fit a line, got {window}")


def _fit(values: pl.Expr, window: int) -> tuple[pl.Expr, pl.Expr]:
    """The window's fitted slope per bar and its value at the window's first bar."""
    total = values.rolling_sum(window_size=window, min_samples=window)
    # Position within the window, oldest bar first, so the slope runs forwards.
    weights = [float(position) for position in range(window)]
    weighted = values.rolling_sum(
        window_size=window, weights=weights, min_samples=window
    )
    midpoint = (window - 1) / 2.0
    spread = window * (window - 1) * (window + 1)
    slope = 12.0 * (weighted - midpoint * total) / spread
    intercept = total / window - slope * midpoint
    return slope, intercept


def _linearreg_expr(values: pl.Expr, window: int) -> pl.Expr:
    slope, intercept = _fit(values, window)
    return intercept + slope * (window - 1)


def _tsf_expr(values: pl.Expr, window: int) -> pl.Expr:
    slope, intercept = _fit(values, window)
    return intercept + slope * window


def linearreg(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Linear Regression: the fitted line's value at the current bar.

    A least-squares line is drawn through the window and read at its newest
    point, which smooths the series without the lag a moving average carries.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods the line is fitted over.

    Returns:
        A ``pl.Expr`` on the scale of the input. The first ``window - 1`` rows
        are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 2.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    _validate_regression_window(window)
    return _linearreg_expr(to_expr(column), window)


def linearreg_slope(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Linear Regression Slope: the fitted line's change per bar.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods the line is fitted over.

    Returns:
        A ``pl.Expr`` yielding units of the input per bar. The first
        ``window - 1`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 2.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    _validate_regression_window(window)
    return _fit(to_expr(column), window)[0]


def linearreg_intercept(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Linear Regression Intercept: the fitted line's value at the window's first bar.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods the line is fitted over.

    Returns:
        A ``pl.Expr`` on the scale of the input. The first ``window - 1`` rows
        are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 2.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    _validate_regression_window(window)
    return _fit(to_expr(column), window)[1]


def linearreg_angle(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Linear Regression Angle: the fitted slope restated in degrees.

    The angle depends on the input's scale, since one bar on the horizontal
    axis is compared against one price unit on the vertical: a series quoted
    in cents tilts far more steeply than the same series quoted in dollars.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods the line is fitted over.

    Returns:
        A ``pl.Expr`` yielding degrees in ``(-90, 90)``. The first
        ``window - 1`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 2.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    _validate_regression_window(window)
    return _fit(to_expr(column), window)[0].arctan() * _DEGREES_PER_RADIAN


def tsf(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Time Series Forecast: the fitted line extended one bar past the window.

    Identical to :func:`linearreg` but read one bar further along, so it leads
    that line by exactly one slope step.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods the line is fitted over.

    Returns:
        A ``pl.Expr`` on the scale of the input. The first ``window - 1`` rows
        are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 2.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    _validate_regression_window(window)
    return _tsf_expr(to_expr(column), window)
