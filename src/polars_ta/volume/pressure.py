"""Volume indicators that weight each bar's price move by the volume behind it."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import (
    IntoColumn,
    apply_to_columns,
    validate_positive,
    validate_window,
)
from polars_ta.overlay.ma import ema

# Ease of Movement divides by raw volume, so the ratio needs rescaling to be readable.
_EOM_SCALE = 100_000_000.0


def _fi_expr(close: pl.Expr, volume: pl.Expr, window: int) -> pl.Expr:
    return ema(close.diff() * volume, window, mode="recursive")


def _eom_expr(high: pl.Expr, low: pl.Expr, volume: pl.Expr, window: int) -> pl.Expr:
    # (high + low) - previous (high + low) is twice the move of the bar's midpoint.
    distance = (high + low) - (high.shift(1) + low.shift(1))
    span = high - low
    raw = (
        pl.when(distance.is_null() | span.is_null() | volume.is_null())
        .then(None)
        .when(volume != 0.0)
        .then(distance * span / (2.0 * volume) * _EOM_SCALE)
        .otherwise(0.0)
    )
    return raw.rolling_mean(window_size=window, min_samples=window)


def _vpt_expr(close: pl.Expr, volume: pl.Expr) -> pl.Expr:
    previous = close.shift(1)
    step = (
        pl.when(previous.is_null() | close.is_null() | volume.is_null())
        .then(None)
        .when(previous != 0.0)
        .then((close - previous) / previous * volume)
        .otherwise(0.0)
    )
    # The first bar has no return to contribute, so it seeds the total at zero.
    return (pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(step)).cum_sum()


def _nvi_expr(close: pl.Expr, volume: pl.Expr, start_value: float) -> pl.Expr:
    previous_close = close.shift(1)
    previous_volume = volume.shift(1)
    known = (
        close.is_not_null()
        & previous_close.is_not_null()
        & volume.is_not_null()
        & previous_volume.is_not_null()
    )
    growth = (
        pl.when(previous_close != 0.0)
        .then((close - previous_close) / previous_close)
        .otherwise(0.0)
    )
    # Bars that did not shrink in volume, and bars we cannot judge, carry the level on.
    factor = (
        pl.when(known & (volume < previous_volume)).then(1.0 + growth).otherwise(1.0)
    )
    return start_value * factor.cum_prod()


@overload
def fi(close: str | pl.Expr, volume: str | pl.Expr, window: int = 13) -> pl.Expr: ...


@overload
def fi(close: pl.Series, volume: pl.Series, window: int = 13) -> pl.Series: ...


def fi(close: IntoColumn, volume: IntoColumn, window: int = 13) -> pl.Expr | pl.Series:
    """Force Index: the close-to-close move multiplied by the volume behind it.

    Price change alone says how far a move went; multiplying by volume says how
    much conviction carried it. The result is smoothed, since the raw product
    is far too noisy to read.

    Args:
        close: Column name, expression, or series of closing prices.
        volume: Column name, expression, or series of traded volume.
        window: Number of periods in the exponential average.

    Returns:
        A value in price-times-volume units: a ``pl.Series`` when every input
        is a series, otherwise a ``pl.Expr``. The first ``window`` rows are
        null, one for the difference and ``window - 1`` for the average.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns((close, volume), lambda c, v: _fi_expr(c, v, window))


@overload
def eom(
    high: str | pl.Expr,
    low: str | pl.Expr,
    volume: str | pl.Expr,
    window: int = 14,
) -> pl.Expr: ...


@overload
def eom(
    high: pl.Series, low: pl.Series, volume: pl.Series, window: int = 14
) -> pl.Series: ...


def eom(
    high: IntoColumn, low: IntoColumn, volume: IntoColumn, window: int = 14
) -> pl.Expr | pl.Series:
    """Ease of Movement: how far the bar's midpoint travelled per unit of volume.

    A large move on light volume reads high, a small move on heavy volume reads
    near zero, so the sign gives direction and the size gives how little effort
    that direction cost.

    Args:
        high: Column name, expression, or series of high prices.
        low: Column name, expression, or series of low prices.
        volume: Column name, expression, or series of traded volume.
        window: Number of periods in the simple average; ``1`` leaves the
            reading unsmoothed.

    Returns:
        A ``pl.Series`` when every input is a series, otherwise a ``pl.Expr``.
        The first ``window`` rows are null. Readings are scaled by ``1e8``, as
        the usual box-ratio convention does, and a bar with no volume reports
        ``0.0`` rather than dividing.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_window(window)
    return apply_to_columns(
        (high, low, volume), lambda h, low_, v: _eom_expr(h, low_, v, window)
    )


@overload
def vpt(close: str | pl.Expr, volume: str | pl.Expr) -> pl.Expr: ...


@overload
def vpt(close: pl.Series, volume: pl.Series) -> pl.Series: ...


def vpt(close: IntoColumn, volume: IntoColumn) -> pl.Expr | pl.Series:
    """Volume-Price Trend: a running total of volume scaled by each bar's return.

    Where :func:`~polars_ta.volume.obv` adds the whole of a bar's volume on any
    up close, this weights it by how far price actually moved, so a large move
    counts for more than a marginal one.

    Args:
        close: Column name, expression, or series of closing prices.
        volume: Column name, expression, or series of traded volume.

    Returns:
        A running total: a ``pl.Series`` when every input is a series,
        otherwise a ``pl.Expr``. There is no warm-up; the first bar has no
        return and so seeds the total at ``0.0``.

    Raises:
        TypeError: If series inputs are mixed with names or expressions.
    """
    return apply_to_columns((close, volume), _vpt_expr)


@overload
def nvi(
    close: str | pl.Expr, volume: str | pl.Expr, *, start_value: float = 1000.0
) -> pl.Expr: ...


@overload
def nvi(
    close: pl.Series, volume: pl.Series, *, start_value: float = 1000.0
) -> pl.Series: ...


def nvi(
    close: IntoColumn, volume: IntoColumn, *, start_value: float = 1000.0
) -> pl.Expr | pl.Series:
    """Negative Volume Index: an index that compounds only on quiet days.

    The premise is that informed money moves on light volume, so the index
    takes the day's return when volume fell and holds flat otherwise.

    Args:
        close: Column name, expression, or series of closing prices.
        volume: Column name, expression, or series of traded volume.
        start_value: Level the index starts from, conventionally ``1000``.

    Returns:
        An index level: a ``pl.Series`` when every input is a series, otherwise
        a ``pl.Expr``. There is no warm-up; the first bar is ``start_value``.
        A bar whose return cannot be computed carries the level forward rather
        than nulling the rest of the index.

    Raises:
        ValueError: If ``start_value`` is not a positive finite number.
        TypeError: If series inputs are mixed with names or expressions.
    """
    validate_positive("start_value", start_value)
    return apply_to_columns(
        (close, volume), lambda c, v: _nvi_expr(c, v, float(start_value))
    )
