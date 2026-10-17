"""Mesa Sine Wave."""

from __future__ import annotations

import math

import polars as pl

from polars_ta._common import IntoColumn, to_expr, validate_window

MSW_FIELDS = ("sine", "lead")

# Below this the real part is treated as zero and the phase pinned to a quarter turn.
_REAL_EPSILON = 0.001
_LEAD_OFFSET = math.pi / 4.0


def _projection(values: pl.Expr, window: int, weights: list[float]) -> pl.Expr:
    """Project the window onto one cycle of a sinusoid, oldest bar first."""
    # Weighted rolling sums panic on nulls, so fill them and mask the result instead.
    filled = values.cast(pl.Float64).fill_null(0.0)
    nulls_in_window = (
        values.is_null()
        .cast(pl.UInt32)
        .rolling_sum(window_size=window, min_samples=window)
    )
    return (
        pl.when(nulls_in_window > 0)
        .then(None)
        .otherwise(
            filled.rolling_sum(window_size=window, weights=weights, min_samples=window)
        )
    )


def _msw_expr(values: pl.Expr, window: int) -> pl.Expr:
    # Weight index 0 is the oldest bar in the window, so it carries the largest age.
    ages = [window - 1 - position for position in range(window)]
    turn = 2.0 * math.pi / window
    real = _projection(values, window, [math.cos(turn * age) for age in ages])
    imaginary = _projection(values, window, [math.sin(turn * age) for age in ages])

    phase = (
        pl.when(real.abs() > _REAL_EPSILON)
        .then((imaginary / real).arctan())
        .when(imaginary < 0.0)
        .then(-math.pi / 2.0)
        .otherwise(math.pi / 2.0)
    )
    # arctan only spans half a turn; a negative real part places the phase in the
    # other half. The further quarter turn is Ehlers' own offset.
    phase = phase + pl.when(real < 0.0).then(math.pi).otherwise(0.0) + math.pi / 2.0
    phase = phase % (2.0 * math.pi)
    # The when-chain above treats a null projection as a false branch, so an
    # incomplete window has to be masked back out explicitly.
    known = real.is_not_null() & imaginary.is_not_null()
    return pl.struct(
        sine=pl.when(known).then(phase.sin()),
        lead=pl.when(known).then((phase + _LEAD_OFFSET).sin()),
    ).alias("msw")


def msw(column: IntoColumn, window: int = 5) -> pl.Expr:
    """Mesa Sine Wave: the phase of a fixed-length cycle, read as a sine.

    Each window is projected onto one full cycle of a sinusoid, which gives a
    phase angle; the two outputs are that phase and the same phase an eighth of
    a cycle ahead. Their crossings call cycle turns early, but only while the
    market is actually cycling — in a trend the two lines run apart and stay
    there, which is itself the signal to stop using them.

    Unlike :func:`ht_sine`, the cycle length is fixed by ``window`` rather than
    measured bar by bar.

    Args:
        column: Column name or expression holding the input values.
        window: Number of periods in one cycle.

    Returns:
        A ``pl.Expr`` yielding a struct with fields ``sine`` and ``lead``, both
        in ``[-1, 1]``. The first ``window - 1`` rows are null, as is any row
        whose window contains a null input.

    Raises:
        ValueError: If ``window`` is not an integer of at least 1.
        TypeError: If ``column`` is not a ``str`` or ``pl.Expr``.
    """
    validate_window(window)
    return _msw_expr(to_expr(column), window)
