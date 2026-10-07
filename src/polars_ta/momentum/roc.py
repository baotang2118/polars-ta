"""Momentum and the rate-of-change family."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window


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


def mom(column: IntoColumn, window: int = 10) -> pl.Expr:
    """Momentum: the absolute change over ``window`` periods.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Expr`` yielding ``P_t - P_{t-window}``. The first ``window``
        rows are null.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _mom_expr(to_expr(column), window)


def roc(column: IntoColumn, window: int = 10) -> pl.Expr:
    """Rate of change as a percentage: ``(P_t / P_{t-window} - 1) * 100``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Expr``. The first ``window`` rows are null, and a zero
        reference value reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _change_expr(
        to_expr(column), window, lambda now, before: (now / before - 1.0) * 100.0
    )


def rocp(column: IntoColumn, window: int = 10) -> pl.Expr:
    """Rate of change as a fraction: ``(P_t - P_{t-window}) / P_{t-window}``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Expr``. The first ``window`` rows are null, and a zero
        reference value reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _change_expr(
        to_expr(column), window, lambda now, before: (now - before) / before
    )


def rocr(column: IntoColumn, window: int = 10) -> pl.Expr:
    """Rate of change as a ratio: ``P_t / P_{t-window}``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Expr``. The first ``window`` rows are null, and a zero
        reference value reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _change_expr(to_expr(column), window, lambda now, before: now / before)


def rocr100(column: IntoColumn, window: int = 10) -> pl.Expr:
    """Rate of change as a ratio on a 100 scale: ``P_t / P_{t-window} * 100``.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods to look back.

    Returns:
        A ``pl.Expr``. The first ``window`` rows are null, and a zero
        reference value reports ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _change_expr(
        to_expr(column), window, lambda now, before: now / before * 100.0
    )
