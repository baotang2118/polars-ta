"""Shared argument validation and input dispatch for indicator functions."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

IntoColumn = str | pl.Expr | pl.Series
"""Accepted indicator input: a column name, an expression, or an eager series."""

_TEMP_NAME = "__polars_ta_input__"


def validate_window(window: int) -> None:
    """Raise ``ValueError`` unless ``window`` is an integer of at least 1."""
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError(f"window must be an int, got {type(window).__name__}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")


def validate_alpha(alpha: float) -> None:
    """Raise ``ValueError`` unless ``alpha`` lies in the interval (0, 1]."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a float, got {type(alpha).__name__}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}")


def apply_to_column(
    column: IntoColumn,
    builder: Callable[[pl.Expr], pl.Expr],
) -> pl.Expr | pl.Series:
    """Build an indicator expression, evaluating it eagerly for series input."""
    if isinstance(column, pl.Series):
        frame = column.rename(_TEMP_NAME).to_frame()
        result = frame.select(builder(pl.col(_TEMP_NAME))).to_series()
        return result.rename(column.name)
    if isinstance(column, pl.Expr):
        return builder(column)
    if isinstance(column, str):
        return builder(pl.col(column))
    raise TypeError(
        f"column must be a str, pl.Expr, or pl.Series, got {type(column).__name__}"
    )
