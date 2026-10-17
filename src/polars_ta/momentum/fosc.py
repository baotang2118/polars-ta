"""Forecast Oscillator."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr
from polars_ta.statistic.regression import _tsf_expr, _validate_regression_window


def _fosc_expr(values: pl.Expr, window: int) -> pl.Expr:
    # The forecast for this bar is the one the previous bar's window made.
    forecast = _tsf_expr(values, window).shift(1)
    return (
        pl.when(forecast.is_null() | values.is_null())
        .then(None)
        .when(values != 0.0)
        .then((values - forecast) / values * 100.0)
        .otherwise(0.0)
    )


def fosc(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Forecast Oscillator: how far price came in above the forecast it was given.

    Each bar is compared against the value
    :func:`~polars_ta.statistic.regression.tsf` projected for it from the
    preceding window. A run of positive readings says price keeps beating its
    own trend line, and a cross through zero marks the trend losing its grip.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods the forecasting line is fitted over.

    Returns:
        A ``pl.Expr`` yielding a percentage of the current value. The first
        ``window`` rows are null, and a zero current value reports ``0.0``
        rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 2.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    _validate_regression_window(window)
    return _fosc_expr(to_expr(column), window)
