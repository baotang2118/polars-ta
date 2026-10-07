"""Price transforms: single-value summaries of an OHLC bar."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns


def _avgprice_expr(
    open_: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr
) -> pl.Expr:
    return (open_ + high + low + close) / 4.0


def _medprice_expr(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    return (high + low) / 2.0


def _typprice_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    return (high + low + close) / 3.0


def _wclprice_expr(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    return (high + low + 2.0 * close) / 4.0


@overload
def avgprice(
    open_: str | pl.Expr,
    high: str | pl.Expr,
    low: str | pl.Expr,
    close: str | pl.Expr,
) -> pl.Expr: ...


@overload
def avgprice(
    open_: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series
) -> pl.Series: ...


def avgprice(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr | pl.Series:
    """Average Price: the mean of the four OHLC values.

    Args:
        open_: Column name, expression, or series of opening prices.
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.

    Returns:
        A price: a ``pl.Series`` when every input is a series, otherwise a
        ``pl.Expr``. There is no warm-up, and any null input nulls the row.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((open_, high, low, close), _avgprice_expr)


@overload
def medprice(high: str | pl.Expr, low: str | pl.Expr) -> pl.Expr: ...


@overload
def medprice(high: pl.Series, low: pl.Series) -> pl.Series: ...


def medprice(high: IntoColumn, low: IntoColumn) -> pl.Expr | pl.Series:
    """Median Price: the midpoint of the bar's range.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.

    Returns:
        A price: a ``pl.Series`` when every input is a series, otherwise a
        ``pl.Expr``. There is no warm-up, and any null input nulls the row.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((high, low), _medprice_expr)


@overload
def typprice(
    high: str | pl.Expr, low: str | pl.Expr, close: str | pl.Expr
) -> pl.Expr: ...


@overload
def typprice(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series: ...


def typprice(
    high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr | pl.Series:
    """Typical Price: the mean of high, low, and close.

    This is the price series that :func:`~polars_ta.momentum.cci`,
    :func:`~polars_ta.momentum.mfi`, and :func:`~polars_ta.volume.vwap` consume.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.

    Returns:
        A price: a ``pl.Series`` when every input is a series, otherwise a
        ``pl.Expr``. There is no warm-up, and any null input nulls the row.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((high, low, close), _typprice_expr)


@overload
def wclprice(
    high: str | pl.Expr, low: str | pl.Expr, close: str | pl.Expr
) -> pl.Expr: ...


@overload
def wclprice(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series: ...


def wclprice(
    high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr | pl.Series:
    """Weighted Close Price: :func:`typprice` with the close counted twice.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        close: Column name, expression, or series of closing prices.

    Returns:
        A price: a ``pl.Series`` when every input is a series, otherwise a
        ``pl.Expr``. There is no warm-up, and any null input nulls the row.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((high, low, close), _wclprice_expr)
