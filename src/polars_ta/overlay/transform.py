"""Price transforms: single-value summaries of an OHLC bar."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs


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


def avgprice(
    open_: IntoColumn, high: IntoColumn, low: IntoColumn, close: IntoColumn
) -> pl.Expr:
    """Average Price: the mean of the four OHLC values.

    Args:
        open_: Column name or expression of opening prices.
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding a price. There is no warm-up, and any null
        input nulls the row.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    return _avgprice_expr(*to_exprs(open_, high, low, close))


def medprice(high: IntoColumn, low: IntoColumn) -> pl.Expr:
    """Median Price: the midpoint of the bar's range.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.

    Returns:
        A ``pl.Expr`` yielding a price. There is no warm-up, and any null
        input nulls the row.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    return _medprice_expr(*to_exprs(high, low))


def typprice(high: IntoColumn, low: IntoColumn, close: IntoColumn) -> pl.Expr:
    """Typical Price: the mean of high, low, and close.

    This is the price series that :func:`~polars_ta.momentum.cci`,
    :func:`~polars_ta.momentum.mfi`, and :func:`~polars_ta.volume.vwap` consume.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding a price. There is no warm-up, and any null
        input nulls the row.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    return _typprice_expr(*to_exprs(high, low, close))


def wclprice(high: IntoColumn, low: IntoColumn, close: IntoColumn) -> pl.Expr:
    """Weighted Close Price: :func:`typprice` with the close counted twice.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.

    Returns:
        A ``pl.Expr`` yielding a price. There is no warm-up, and any null
        input nulls the row.

    Raises:
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    return _wclprice_expr(*to_exprs(high, low, close))
