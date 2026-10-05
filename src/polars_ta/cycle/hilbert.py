"""Hilbert transform cycle indicators."""

from __future__ import annotations

from typing import overload

import polars as pl

from polars_ta._common import IntoColumn, apply_to_column
from polars_ta._hilbert import (
    LONG_LOOKBACK,
    SHORT_LOOKBACK,
    HilbertSeries,
    hilbert_transform,
    mask_lookback,
)

PHASOR_FIELDS = ("in_phase", "quadrature")
SINE_FIELDS = ("sine", "lead_sine")

_FLOAT = pl.Float64
_PHASOR_DTYPE = pl.Struct({"in_phase": pl.Float64, "quadrature": pl.Float64})
_SINE_DTYPE = pl.Struct({"sine": pl.Float64, "lead_sine": pl.Float64})
_MODE_DTYPE = pl.Int8


def _transform(column: pl.Series) -> HilbertSeries:
    return hilbert_transform(column.to_list(), 0.5, 0.05)


def _single(field: str, lookback: int, dtype: pl.DataType) -> object:
    def scan(column: pl.Series) -> pl.Series:
        values = getattr(_transform(column), field)
        return pl.Series(values=mask_lookback(values, lookback), dtype=dtype)

    return scan


def _pair(first: str, second: str, lookback: int, dtype: pl.DataType) -> object:
    def scan(column: pl.Series) -> pl.Series:
        series = _transform(column)
        left = mask_lookback(getattr(series, first), lookback)
        right = mask_lookback(getattr(series, second), lookback)
        return pl.Series(
            values=[{first: a, second: b} for a, b in zip(left, right)], dtype=dtype
        )

    return scan


@overload
def ht_dcperiod(column: str | pl.Expr) -> pl.Expr: ...


@overload
def ht_dcperiod(column: pl.Series) -> pl.Series: ...


def ht_dcperiod(column: IntoColumn) -> pl.Expr | pl.Series:
    """Hilbert Transform Dominant Cycle Period: the length of the current cycle.

    The measured period is clamped to ``[6, 50]`` bars and smoothed, so it
    drifts rather than jumping when the market changes rhythm.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A period in bars: a ``pl.Series`` when ``column`` is a series,
        otherwise a ``pl.Expr``. The first 32 rows are null, and a null input
        ends the recursion, leaving every later row null.
    """
    return apply_to_column(
        column,
        lambda values: values.map_batches(
            _single("smooth_period", SHORT_LOOKBACK, _FLOAT), return_dtype=_FLOAT
        ),
    )


@overload
def ht_dcphase(column: str | pl.Expr) -> pl.Expr: ...


@overload
def ht_dcphase(column: pl.Series) -> pl.Series: ...


def ht_dcphase(column: IntoColumn) -> pl.Expr | pl.Series:
    """Hilbert Transform Dominant Cycle Phase: where in its cycle price sits.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A phase in degrees, wrapped to end at ``315``: a ``pl.Series`` when
        ``column`` is a series, otherwise a ``pl.Expr``. The first 63 rows are
        null.
    """
    return apply_to_column(
        column,
        lambda values: values.map_batches(
            _single("dc_phase", LONG_LOOKBACK, _FLOAT), return_dtype=_FLOAT
        ),
    )


@overload
def ht_phasor(column: str | pl.Expr) -> pl.Expr: ...


@overload
def ht_phasor(column: pl.Series) -> pl.Series: ...


def ht_phasor(column: IntoColumn) -> pl.Expr | pl.Series:
    """Hilbert Transform Phasor Components: the real and imaginary cycle parts.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A struct with fields ``in_phase`` and ``quadrature``: a ``pl.Series``
        when ``column`` is a series, otherwise a ``pl.Expr``. The first 32 rows
        are null.
    """
    return apply_to_column(
        column,
        lambda values: values.map_batches(
            _pair("in_phase", "quadrature", SHORT_LOOKBACK, _PHASOR_DTYPE),
            return_dtype=_PHASOR_DTYPE,
        ),
    )


@overload
def ht_sine(column: str | pl.Expr) -> pl.Expr: ...


@overload
def ht_sine(column: pl.Series) -> pl.Series: ...


def ht_sine(column: IntoColumn) -> pl.Expr | pl.Series:
    """Hilbert Transform SineWave: the cycle phase as a sine and a lead sine.

    The two lines cross a quarter-cycle apart, which marks cycle turns well
    ahead of a moving-average crossover — but only while the market is in a
    cycle rather than a trend, which is what :func:`ht_trendmode` reports.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A struct with fields ``sine`` and ``lead_sine``, each in ``[-1, 1]``: a
        ``pl.Series`` when ``column`` is a series, otherwise a ``pl.Expr``. The
        first 63 rows are null.
    """
    return apply_to_column(
        column,
        lambda values: values.map_batches(
            _pair("sine", "lead_sine", LONG_LOOKBACK, _SINE_DTYPE),
            return_dtype=_SINE_DTYPE,
        ),
    )


@overload
def ht_trendmode(column: str | pl.Expr) -> pl.Expr: ...


@overload
def ht_trendmode(column: pl.Series) -> pl.Series: ...


def ht_trendmode(column: IntoColumn) -> pl.Expr | pl.Series:
    """Hilbert Transform Trend versus Cycle Mode.

    Reports ``1`` when the market is trending and ``0`` when it is cycling,
    which decides whether a trend-following or a cycle-following indicator
    should be trusted.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        An ``Int8`` of ``0`` or ``1``: a ``pl.Series`` when ``column`` is a
        series, otherwise a ``pl.Expr``. The first 63 rows are null.
    """
    return apply_to_column(
        column,
        lambda values: values.map_batches(
            _single("trend_mode", LONG_LOOKBACK, _MODE_DTYPE),
            return_dtype=_MODE_DTYPE,
        ),
    )


@overload
def ht_trendline(column: str | pl.Expr) -> pl.Expr: ...


@overload
def ht_trendline(column: pl.Series) -> pl.Series: ...


def ht_trendline(column: IntoColumn) -> pl.Expr | pl.Series:
    """Hilbert Transform Instantaneous Trendline.

    Averages price over exactly one dominant cycle, which cancels the cycle
    and leaves the trend. Because the averaging length adapts, it keeps that
    property as the cycle stretches or compresses.

    Args:
        column: Column name, expression, or series holding the input values.

    Returns:
        A level in price units: a ``pl.Series`` when ``column`` is a series,
        otherwise a ``pl.Expr``. The first 63 rows are null.
    """
    return apply_to_column(
        column,
        lambda values: values.map_batches(
            _single("trendline", LONG_LOOKBACK, _FLOAT), return_dtype=_FLOAT
        ),
    )
