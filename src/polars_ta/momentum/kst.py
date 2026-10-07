"""KST Oscillator (Know Sure Thing)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window
from polars_ta.momentum.roc import _change_expr
from polars_ta.overlay.ma import _sma_expr

KST_FIELDS = ("kst", "signal")

ROC_PERIODS = (10, 15, 20, 30)
SMA_PERIODS = (10, 10, 10, 15)


def _validate_periods(name: str, periods: Sequence[int]) -> None:
    if len(periods) != 4:
        raise ValueError(f"{name} must hold exactly 4 periods, got {len(periods)}")
    for period in periods:
        validate_window(period)


def _kst_line(
    values: pl.Expr, roc_periods: Sequence[int], sma_periods: Sequence[int]
) -> pl.Expr:
    total = None
    for weight, (roc_period, sma_period) in enumerate(
        zip(roc_periods, sma_periods), start=1
    ):
        change = _change_expr(
            values, roc_period, lambda now, before: now / before - 1.0
        )
        term = float(weight) * _sma_expr(change, sma_period)
        total = term if total is None else total + term
    return 100.0 * total


def _kst_expr(
    values: pl.Expr,
    roc_periods: Sequence[int],
    sma_periods: Sequence[int],
    signal_period: int,
) -> pl.Expr:
    line = _kst_line(values, roc_periods, sma_periods)
    return pl.struct(kst=line, signal=_sma_expr(line, signal_period))


@overload
def kst(
    column: str | pl.Expr,
    roc_periods: Sequence[int] = ROC_PERIODS,
    sma_periods: Sequence[int] = SMA_PERIODS,
    signal_period: int = 9,
) -> pl.Expr: ...


@overload
def kst(
    column: pl.Series,
    roc_periods: Sequence[int] = ROC_PERIODS,
    sma_periods: Sequence[int] = SMA_PERIODS,
    signal_period: int = 9,
) -> pl.Series: ...


def kst(
    column: IntoColumn,
    roc_periods: Sequence[int] = ROC_PERIODS,
    sma_periods: Sequence[int] = SMA_PERIODS,
    signal_period: int = 9,
) -> pl.Expr | pl.Series:
    """KST Oscillator: four smoothed rates of change, weighted by their horizon.

    One rate of change only sees one cycle length. Stacking four of them, with
    the slowest counting four times as much as the fastest, gives a reading
    that turns on short-term momentum but is anchored by the long term.

    Args:
        column: Column name, expression, or series holding the input values.
        roc_periods: Four rate-of-change look-backs, shortest first.
        sma_periods: Four averaging periods, one per rate of change.
        signal_period: Number of periods in the signal average.

    Returns:
        A struct with fields ``kst`` and ``signal``, both percentages: a
        ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        ``kst`` starts once its slowest term does, and ``signal`` a further
        ``signal_period - 1`` rows later; the fields keep separate warm-ups so
        no good ``kst`` data is discarded.

    Raises:
        ValueError: If a period is invalid or a sequence does not hold four.
    """
    _validate_periods("roc_periods", roc_periods)
    _validate_periods("sma_periods", sma_periods)
    validate_window(signal_period)
    return apply_to_column(
        column,
        lambda values: _kst_expr(
            values, tuple(roc_periods), tuple(sma_periods), signal_period
        ),
    )
