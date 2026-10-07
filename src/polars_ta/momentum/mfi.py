"""Money Flow Index."""

from __future__ import annotations

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_window


def _mfi_expr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    window: int,
) -> pl.Expr:
    typical = (high + low + close) / 3.0
    flow = typical * volume
    change = typical.diff()
    known = change.is_not_null() & flow.is_not_null()
    positive = pl.when(~known).then(None).when(change > 0.0).then(flow).otherwise(0.0)
    negative = pl.when(~known).then(None).when(change < 0.0).then(flow).otherwise(0.0)
    positive_sum = positive.rolling_sum(window_size=window, min_samples=window)
    negative_sum = negative.rolling_sum(window_size=window, min_samples=window)
    total = positive_sum + negative_sum
    return (
        pl.when(total.is_null())
        .then(None)
        .when(total > 0.0)
        .then(100.0 * positive_sum / total)
        .otherwise(0.0)
    )


def mfi(
    high: IntoColumn,
    low: IntoColumn,
    close: IntoColumn,
    volume: IntoColumn,
    window: int = 14,
) -> pl.Expr:
    """Money Flow Index: a volume-weighted relative strength index.

    Each bar's typical price ``(high + low + close) / 3`` is multiplied by
    volume to give its money flow, which counts as positive or negative
    depending on whether the typical price rose or fell. The index is the
    positive share of the total flow over ``window`` bars, matching TA-Lib.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        close: Column name or expression of closing prices.
        volume: Column name or expression of traded volume.
        window: Number of bars summed for each side of the flow.

    Returns:
        A ``pl.Expr`` yielding a value in ``[0, 100]``. The first ``window``
        rows are null. A window with no money flow at all reports ``0.0``.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    high_expr, low_expr, close_expr, volume_expr = to_exprs(high, low, close, volume)
    return _mfi_expr(high_expr, low_expr, close_expr, volume_expr, window)
