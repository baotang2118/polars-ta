"""Relative Strength Index and the Chande Momentum Oscillator."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window
from polars_ta.overlay.ma import ema

NEUTRAL_RSI = 50.0
"""Value reported when a window contains neither a gain nor a loss."""


def _smoothed_gain_loss(values: pl.Expr, window: int) -> tuple[pl.Expr, pl.Expr]:
    """Wilder-smoothed average gain and average loss over ``window`` periods."""
    change = values.diff()
    smoothing = 1.0 / window
    average_gain = ema(change.clip(lower_bound=0.0), window, alpha=smoothing)
    average_loss = ema((-change).clip(lower_bound=0.0), window, alpha=smoothing)
    return average_gain, average_loss


def _rsi_expr(values: pl.Expr, window: int) -> pl.Expr:
    average_gain, average_loss = _smoothed_gain_loss(values, window)
    total = average_gain + average_loss
    return (
        pl.when(total.is_null())
        .then(None)
        .when(total > 0.0)
        .then(100.0 * average_gain / total)
        .otherwise(NEUTRAL_RSI)
    )


def _cmo_expr(values: pl.Expr, window: int) -> pl.Expr:
    average_gain, average_loss = _smoothed_gain_loss(values, window)
    total = average_gain + average_loss
    # TA-Lib reports 0.0 for a flat window here, not the neutral 50.0 of RSI.
    return (
        pl.when(total.is_null())
        .then(None)
        .when(total > 0.0)
        .then(100.0 * (average_gain - average_loss) / total)
        .otherwise(0.0)
    )


def rsi(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Relative Strength Index: the share of recent movement that was upward.

    Wilder's smoothing is applied to the average gain and average loss, seeded
    with the mean of the first ``window`` changes, matching TA-Lib's ``RSI``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods used to average gains and losses.

    Returns:
        A ``pl.Expr`` yielding a value in ``[0, 100]``. The first ``window``
        rows are null. A window with neither a gain nor a loss reports the
        neutral ``50.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _rsi_expr(to_expr(column), window)


def cmo(column: IntoColumn, window: int = 14) -> pl.Expr:
    """Chande Momentum Oscillator: gains minus losses over their total.

    The same Wilder-smoothed gains and losses as :func:`rsi`, but scaled to
    ``[-100, 100]`` around a zero centre line rather than to ``[0, 100]``
    around fifty. Away from a flat window, ``cmo == 2 * rsi - 100``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods used to average gains and losses.

    Returns:
        A ``pl.Expr`` yielding a value in ``[-100, 100]``. The first
        ``window`` rows are null. A window with neither a gain nor a loss
        reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _cmo_expr(to_expr(column), window)
