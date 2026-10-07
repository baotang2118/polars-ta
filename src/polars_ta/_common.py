"""Shared argument validation and input conversion for indicator functions."""

from __future__ import annotations

import polars as pl

IntoColumn = str | pl.Expr
"""Accepted indicator input: a column name or an expression."""


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


def to_expr(column: IntoColumn) -> pl.Expr:
    """Normalise a column name into an expression, passing expressions through."""
    if isinstance(column, pl.Expr):
        return column
    if isinstance(column, str):
        return pl.col(column)
    if isinstance(column, pl.Series):
        raise TypeError(
            "pl.Series input is not supported; pass a column name or pl.Expr and "
            "evaluate the result on a frame, e.g. series.to_frame().select(...)"
        )
    raise TypeError(f"column must be a str or pl.Expr, got {type(column).__name__}")


def to_exprs(*columns: IntoColumn) -> tuple[pl.Expr, ...]:
    """Normalise several indicator inputs into expressions."""
    return tuple(to_expr(column) for column in columns)
