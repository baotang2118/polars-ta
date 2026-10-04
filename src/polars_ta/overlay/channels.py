"""Channel overlays built from rolling price extremes."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_columns, validate_window

DONCHIAN_FIELDS = ("lower", "middle", "upper")


def _donchian_expr(high: pl.Expr, low: pl.Expr, window: int) -> pl.Expr:
    upper = high.rolling_max(window_size=window, min_samples=window)
    lower = low.rolling_min(window_size=window, min_samples=window)
    return pl.struct(lower=lower, middle=(upper + lower) / 2.0, upper=upper)


@overload
def donchian(high: str | pl.Expr, low: str | pl.Expr, window: int = 20) -> pl.Expr: ...


@overload
def donchian(high: pl.Series, low: pl.Series, window: int = 20) -> pl.Series: ...


def donchian(
    high: IntoColumn, low: IntoColumn, window: int = 20
) -> pl.Expr | pl.Series:
    """Donchian Channels: the highest high and lowest low of the last ``window`` bars.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        window: Number of bars spanned by the channel.

    Returns:
        A struct with fields ``lower``, ``middle``, and ``upper``: a
        ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``window - 1`` rows are null, as is any row whose window
        contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low), lambda h, low_: _donchian_expr(h, low_, window)
    )
