"""Band overlays: envelopes plotted around the price series."""

from __future__ import annotations

import polars as pl

from polars_ta._common import (
    IntoColumn,
    to_expr,
    validate_positive,
    validate_window,
)

BBANDS_FIELDS = ("lower", "middle", "upper")


def _bbands_expr(values: pl.Expr, window: int, num_std: float, ddof: int) -> pl.Expr:
    middle = values.rolling_mean(window_size=window, min_samples=window)
    deviation = (
        values.rolling_std(window_size=window, min_samples=window, ddof=ddof) * num_std
    )
    return pl.struct(
        lower=middle - deviation,
        middle=middle,
        upper=middle + deviation,
    )


def bbands(
    column: IntoColumn,
    window: int = 20,
    *,
    num_std: float = 2.0,
    ddof: int = 0,
) -> pl.Expr:
    """Bollinger Bands: an SMA with standard-deviation envelopes above and below.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods for the moving average and deviation.
        num_std: Width of the envelope in standard deviations.
        ddof: Delta degrees of freedom for the deviation. ``0`` is the
            population deviation used by TA-Lib; ``1`` is the sample deviation.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``lower``, ``middle``, and
        ``upper``. The first ``window - 1`` rows are null, as is any row whose
        window contains a null input.

    Raises:
        ValueError: If ``window``, ``num_std``, or ``ddof`` is invalid.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    validate_positive("num_std", num_std)
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise ValueError(f"ddof must be an int, got {type(ddof).__name__}")
    if not 0 <= ddof < window:
        raise ValueError(f"ddof must satisfy 0 <= ddof < window, got {ddof}")
    return _bbands_expr(to_expr(column), window, num_std, ddof)
