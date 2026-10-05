"""Momentum and the rate-of-change family."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column, validate_window


def _mom_expr(values: pl.Expr, window: int) -> pl.Expr:
    return values - values.shift(window)


def _change_expr(
    values: pl.Expr,
    window: int,
    builder: Callable[[pl.Expr, pl.Expr], pl.Expr],
) -> pl.Expr:
    previous = values.shift(window)
    known = values.is_not_null() & previous.is_not_null()
    # TA-Lib reports 0.0 rather than dividing by a zero reference price.
    return (
        pl.when(~known)
        .then(None)
        .when(previous != 0.0)
        .then(builder(values, previous))
        .otherwise(0.0)
    )


@overload
def mom(column: str | pl.Expr, window: int = 10) -> pl.Expr: ...


@overload
def mom(column: pl.Series, window: int = 10) -> pl.Series: ...


def mom(column: IntoColumn, window: int = 10) -> pl.Expr | pl.Series:
    """Momentum: the absolute change over ``window`` periods.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods to look back.

    Returns:
        ``P_t - P_{t-window}``: a ``pl.Series`` when ``column`` is a series,
        otherwise a ``pl.Expr``. The first ``window`` rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(column, lambda values: _mom_expr(values, window))


@overload
def roc(column: str | pl.Expr, window: int = 10) -> pl.Expr: ...


@overload
def roc(column: pl.Series, window: int = 10) -> pl.Series: ...


def roc(column: IntoColumn, window: int = 10) -> pl.Expr | pl.Series:
    """Rate of change as a percentage: ``(P_t / P_{t-window} - 1) * 100``.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null, and a zero reference value reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(
        column,
        lambda values: _change_expr(
            values, window, lambda now, before: (now / before - 1.0) * 100.0
        ),
    )


@overload
def rocp(column: str | pl.Expr, window: int = 10) -> pl.Expr: ...


@overload
def rocp(column: pl.Series, window: int = 10) -> pl.Series: ...


def rocp(column: IntoColumn, window: int = 10) -> pl.Expr | pl.Series:
    """Rate of change as a fraction: ``(P_t - P_{t-window}) / P_{t-window}``.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null, and a zero reference value reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(
        column,
        lambda values: _change_expr(
            values, window, lambda now, before: (now - before) / before
        ),
    )


@overload
def rocr(column: str | pl.Expr, window: int = 10) -> pl.Expr: ...


@overload
def rocr(column: pl.Series, window: int = 10) -> pl.Series: ...


def rocr(column: IntoColumn, window: int = 10) -> pl.Expr | pl.Series:
    """Rate of change as a ratio: ``P_t / P_{t-window}``.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null, and a zero reference value reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(
        column,
        lambda values: _change_expr(values, window, lambda now, before: now / before),
    )


@overload
def rocr100(column: str | pl.Expr, window: int = 10) -> pl.Expr: ...


@overload
def rocr100(column: pl.Series, window: int = 10) -> pl.Series: ...


def rocr100(column: IntoColumn, window: int = 10) -> pl.Expr | pl.Series:
    """Rate of change as a ratio on a 100 scale: ``P_t / P_{t-window} * 100``.

    Args:
        column: Column name, expression, or series holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null, and a zero reference value reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
    """
    validate_window(window)
    return apply_to_column(
        column,
        lambda values: _change_expr(
            values, window, lambda now, before: now / before * 100.0
        ),
    )
