"""Return measures, which restate price on a percentage scale."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column
from polars_ta.momentum.roc import _change_expr


def _daily_return_expr(values: pl.Expr) -> pl.Expr:
    return _change_expr(values, 1, lambda now, before: (now / before - 1.0) * 100.0)


def _daily_log_return_expr(values: pl.Expr) -> pl.Expr:
    return _change_expr(values, 1, lambda now, before: (now / before).log() * 100.0)


def _cumulative_return_expr(values: pl.Expr) -> pl.Expr:
    base = values.drop_nulls().first()
    return (
        pl.when(values.is_null() | base.is_null())
        .then(None)
        .when(base != 0.0)
        .then((values / base - 1.0) * 100.0)
        .otherwise(0.0)
    )


@overload
def daily_return(column: str | pl.Expr) -> pl.Expr: ...


@overload
def daily_return(column: pl.Series) -> pl.Series: ...


def daily_return(column: IntoColumn) -> pl.Expr | pl.Series:
    """Daily Return: the percentage change from the previous row.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A percentage: a ``pl.Series`` when ``column`` is a series, otherwise a
        ``pl.Expr``. The first row is null, and a zero reference price reports
        ``0.0`` rather than dividing.
    """
    return apply_to_column(column, _daily_return_expr)


@overload
def daily_log_return(column: str | pl.Expr) -> pl.Expr: ...


@overload
def daily_log_return(column: pl.Series) -> pl.Series: ...


def daily_log_return(column: IntoColumn) -> pl.Expr | pl.Series:
    """Daily Log Return: the natural log of the ratio to the previous row.

    Log returns add across time, which simple returns do not, so a sum over a
    period is the period's return rather than an approximation of it.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A percentage: a ``pl.Series`` when ``column`` is a series, otherwise a
        ``pl.Expr``. The first row is null, and a zero reference price reports
        ``0.0`` rather than dividing. Values are assumed positive; a sign
        change yields ``NaN``, as the logarithm is undefined there.
    """
    return apply_to_column(column, _daily_log_return_expr)


@overload
def cumulative_return(column: str | pl.Expr) -> pl.Expr: ...


@overload
def cumulative_return(column: pl.Series) -> pl.Series: ...


def cumulative_return(column: IntoColumn) -> pl.Expr | pl.Series:
    """Cumulative Return: the percentage change from the first known value.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A percentage: a ``pl.Series`` when ``column`` is a series, otherwise a
        ``pl.Expr``. There is no warm-up, so the first known row is ``0.0``.
        A zero base price reports ``0.0`` rather than dividing.
    """
    return apply_to_column(column, _cumulative_return_expr)
