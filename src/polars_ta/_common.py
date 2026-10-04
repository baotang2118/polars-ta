"""Shared argument validation and input dispatch for indicator functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

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


def validate_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is a positive, finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float, got {type(value).__name__}")
    if not value > 0.0 or value == float("inf"):
        raise ValueError(f"{name} must be a positive finite number, got {value}")


def _to_expr(column: IntoColumn) -> pl.Expr:
    if isinstance(column, pl.Expr):
        return column
    if isinstance(column, str):
        return pl.col(column)
    raise TypeError(
        f"column must be a str, pl.Expr, or pl.Series, got {type(column).__name__}"
    )


def apply_to_column(
    column: IntoColumn,
    builder: Callable[[pl.Expr], pl.Expr],
) -> pl.Expr | pl.Series:
    """Build an indicator expression, evaluating it eagerly for series input."""
    if isinstance(column, pl.Series):
        frame = column.rename(_TEMP_NAME).to_frame()
        result = frame.select(builder(pl.col(_TEMP_NAME))).to_series()
        return result.rename(column.name)
    return builder(_to_expr(column))


def apply_to_columns(
    columns: Sequence[IntoColumn],
    builder: Callable[..., pl.Expr],
) -> pl.Expr | pl.Series:
    """Build a multi-input indicator expression, eager when every input is a series."""
    series_count = sum(isinstance(column, pl.Series) for column in columns)
    if series_count == 0:
        return builder(*(_to_expr(column) for column in columns))
    if series_count != len(columns):
        raise TypeError(
            "mixing pl.Series inputs with str or pl.Expr inputs is not supported"
        )
    names = [f"{_TEMP_NAME}{index}" for index in range(len(columns))]
    frame = pl.DataFrame(
        {name: column for name, column in zip(names, columns)}, strict=False
    )
    result = frame.select(builder(*(pl.col(name) for name in names))).to_series()
    return result.rename(columns[0].name)
