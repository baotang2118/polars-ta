"""Variance and standard deviation over a rolling window."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_positive, validate_window


def _var_expr(values: pl.Expr, window: int) -> pl.Expr:
    mean = values.rolling_mean(window_size=window, min_samples=window)
    mean_of_squares = (values * values).rolling_mean(
        window_size=window, min_samples=window
    )
    # Rounding can push a window of near-identical values a hair below zero.
    return (mean_of_squares - mean * mean).clip(lower_bound=0.0)


def _stddev_expr(values: pl.Expr, window: int, nbdev: float) -> pl.Expr:
    return _var_expr(values, window).sqrt() * nbdev


def _validate_ddof(ddof: int) -> None:
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise ValueError(f"ddof must be an int, got {type(ddof).__name__}")
    if ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, got {ddof}")


def _zscore_expr(values: pl.Expr, window: int, ddof: int) -> pl.Expr:
    variance = _var_expr(values, window)
    if ddof == 1 and window > 1:
        variance = variance * (window / (window - 1))
    mean = values.rolling_mean(window_size=window, min_samples=window)
    # The mean-of-squares variance keeps rounding noise on a flat window, so the
    # spread is tested on the raw values to avoid dividing by that noise.
    spread = values.rolling_max(
        window_size=window, min_samples=window
    ) - values.rolling_min(window_size=window, min_samples=window)
    return (
        pl.when(spread > 0.0)
        .then((values - mean) / variance.sqrt())
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def var(column: IntoColumn, window: int = 5) -> pl.Expr:
    """Variance: the mean squared deviation from the window's mean.

    The window is divided by in full rather than by ``window - 1``, so this is
    the population variance of the bars in view, not an estimate of a wider
    population's.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the window, including the current row.

    Returns:
        A ``pl.Expr`` yielding a non-negative value. The first ``window - 1``
        rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _var_expr(to_expr(column), window)


def stddev(column: IntoColumn, window: int = 5, nbdev: float = 1.0) -> pl.Expr:
    """Standard Deviation: the square root of the variance, scaled by ``nbdev``.

    ``nbdev`` is the band width Bollinger Bands and similar envelopes ask for:
    ``2.0`` returns two standard deviations rather than one.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the window, including the current row.
        nbdev: Multiple of the standard deviation to report.

    Returns:
        A ``pl.Expr`` yielding a non-negative value. The first ``window - 1``
        rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1, or if
            ``nbdev`` is not a positive finite number.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    validate_positive("nbdev", nbdev)
    return _stddev_expr(to_expr(column), window, nbdev)


def zscore(column: IntoColumn, window: int = 20, ddof: int = 0) -> pl.Expr:
    """Z-Score: how many standard deviations the current value sits from its mean.

    Both the mean and the standard deviation are taken over the same rolling
    window, so the score says where the latest bar falls within its own recent
    history: ``0`` is the average, ``2`` is two standard deviations above it.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in the window, including the current row.
        ddof: Delta degrees of freedom. ``0`` divides by ``window`` for the
            population standard deviation, matching :func:`var` and
            :func:`stddev`; ``1`` divides by ``window - 1`` for the sample one.

    Returns:
        A ``pl.Expr`` yielding a signed value. The first ``window - 1`` rows are
        null, as is any row whose window has no spread, the score being
        undefined there.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1, or if ``ddof``
            is not ``0`` or ``1``.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    _validate_ddof(ddof)
    return _zscore_expr(to_expr(column), window, ddof)
