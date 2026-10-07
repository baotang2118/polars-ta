"""Parabolic SAR, in its plain and extended forms."""

from __future__ import annotations

from typing import NamedTuple

import polars as pl

from polars_ta._common import IntoColumn, to_exprs, validate_positive

_RETURN_DTYPE = pl.Float64


class _Settings(NamedTuple):
    """Acceleration and reversal parameters for one direction pair."""

    init_long: float
    step_long: float
    max_long: float
    init_short: float
    step_short: float
    max_short: float
    offset_on_reverse: float
    start_value: float
    signed: bool


def _parabolic(
    highs: list[float | None], lows: list[float | None], settings: _Settings
) -> list[float | None]:
    """Walk the stop forward bar by bar, flipping side when price touches it."""
    size = len(highs)
    result: list[float | None] = [None] * size
    start = 0
    while start < size and (highs[start] is None or lows[start] is None):
        start += 1
    first = start + 1
    if first >= size or highs[first] is None or lows[first] is None:
        return result

    if settings.start_value > 0.0:
        is_long = True
    elif settings.start_value < 0.0:
        is_long = False
    else:
        # The sign of the first -DM decides which side the stop starts on.
        up = highs[first] - highs[first - 1]
        down = lows[first - 1] - lows[first]
        is_long = not (down > up and down > 0.0)

    if settings.start_value == 0.0:
        extreme = highs[first] if is_long else lows[first]
        stop = lows[first - 1] if is_long else highs[first - 1]
    else:
        extreme = highs[first] if settings.start_value > 0.0 else lows[first]
        stop = abs(settings.start_value)

    af_long, af_short = settings.init_long, settings.init_short
    # TA-Lib reuses the first bar as its own predecessor on the opening step.
    previous_high, previous_low = highs[first], lows[first]

    for index in range(first, size):
        high, low = highs[index], lows[index]
        if high is None or low is None:
            break
        if index > first:
            previous_high, previous_low = highs[index - 1], lows[index - 1]

        if is_long:
            if low <= stop:
                is_long = False
                stop = max(extreme, previous_high, high)
                stop += stop * settings.offset_on_reverse
                result[index] = -stop if settings.signed else stop
                af_short = settings.init_short
                extreme = low
                stop = max(stop + af_short * (extreme - stop), previous_high, high)
            else:
                result[index] = stop
                if high > extreme:
                    extreme = high
                    af_long = min(af_long + settings.step_long, settings.max_long)
                stop = min(stop + af_long * (extreme - stop), previous_low, low)
        elif high >= stop:
            is_long = True
            stop = min(extreme, previous_low, low)
            stop -= stop * settings.offset_on_reverse
            result[index] = stop
            af_long = settings.init_long
            extreme = high
            stop = min(stop + af_long * (extreme - stop), previous_low, low)
        else:
            result[index] = -stop if settings.signed else stop
            if low < extreme:
                extreme = low
                af_short = min(af_short + settings.step_short, settings.max_short)
            stop = max(stop + af_short * (extreme - stop), previous_high, high)

    return result


def _sar_expr(high: pl.Expr, low: pl.Expr, settings: _Settings) -> pl.Expr:
    def scan(bars: pl.Series) -> pl.Series:
        return pl.Series(
            values=_parabolic(
                bars.struct.field("high").to_list(),
                bars.struct.field("low").to_list(),
                settings,
            ),
            dtype=_RETURN_DTYPE,
        )

    return pl.struct(high=high, low=low).map_batches(scan, return_dtype=_RETURN_DTYPE)


def sar(
    high: IntoColumn,
    low: IntoColumn,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pl.Expr:
    """Parabolic SAR: a trailing stop that accelerates toward price.

    The stop moves only in the direction of the trade and speeds up each time
    a new extreme is set, so it eventually catches price and flips sides. The
    first bar's direction is taken from the sign of the first ``-DM``.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        acceleration: Step added to the acceleration factor at each new extreme.
        maximum: Ceiling on the acceleration factor.

    Returns:
        A ``pl.Expr`` yielding a stop level in price units. The first row is
        null, and a null high or low ends the scan, leaving every later row
        null.

    Raises:
        ValueError: If ``acceleration`` or ``maximum`` is not positive and finite.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    validate_positive("acceleration", acceleration)
    validate_positive("maximum", maximum)
    settings = _Settings(
        init_long=acceleration,
        step_long=acceleration,
        max_long=maximum,
        init_short=acceleration,
        step_short=acceleration,
        max_short=maximum,
        offset_on_reverse=0.0,
        start_value=0.0,
        signed=False,
    )
    return _sar_expr(*to_exprs(high, low), settings)


def sarext(
    high: IntoColumn,
    low: IntoColumn,
    *,
    start_value: float = 0.0,
    offset_on_reverse: float = 0.0,
    acceleration_init_long: float = 0.02,
    acceleration_long: float = 0.02,
    acceleration_max_long: float = 0.2,
    acceleration_init_short: float = 0.02,
    acceleration_short: float = 0.02,
    acceleration_max_short: float = 0.2,
) -> pl.Expr:
    """Extended Parabolic SAR: independent settings for each side, signed output.

    Long and short trades rarely behave symmetrically, so every acceleration
    parameter is separate here. Short readings are returned negated, which is
    how TA-Lib reports the current side alongside the level.

    Args:
        high: Column name or expression of high prices.
        low: Column name or expression of low prices.
        start_value: Initial stop level. Positive forces a long start, negative
            a short one; ``0`` picks the side from the first ``-DM``.
        offset_on_reverse: Fraction by which the stop is widened on a reversal.
        acceleration_init_long: Acceleration factor at the start of a long.
        acceleration_long: Step added at each new high.
        acceleration_max_long: Ceiling on the long acceleration factor.
        acceleration_init_short: Acceleration factor at the start of a short.
        acceleration_short: Step added at each new low.
        acceleration_max_short: Ceiling on the short acceleration factor.

    Returns:
        A ``pl.Expr`` yielding a signed stop level: positive while long,
        negative while short. The first row is null.

    Raises:
        ValueError: If an acceleration argument is not positive and finite, or
            if ``offset_on_reverse`` is negative.
        TypeError: If an input is not a ``str`` or ``pl.Expr``.
    """
    accelerations = {
        "acceleration_init_long": acceleration_init_long,
        "acceleration_long": acceleration_long,
        "acceleration_max_long": acceleration_max_long,
        "acceleration_init_short": acceleration_init_short,
        "acceleration_short": acceleration_short,
        "acceleration_max_short": acceleration_max_short,
    }
    for name, value in accelerations.items():
        validate_positive(name, value)
    if isinstance(offset_on_reverse, bool) or not isinstance(
        offset_on_reverse, (int, float)
    ):
        raise ValueError(
            f"offset_on_reverse must be a float, got {type(offset_on_reverse).__name__}"
        )
    if offset_on_reverse < 0.0:
        raise ValueError(f"offset_on_reverse must be >= 0, got {offset_on_reverse}")
    settings = _Settings(
        init_long=acceleration_init_long,
        step_long=acceleration_long,
        max_long=acceleration_max_long,
        init_short=acceleration_init_short,
        step_short=acceleration_short,
        max_short=acceleration_max_short,
        offset_on_reverse=float(offset_on_reverse),
        start_value=float(start_value),
        signed=True,
    )
    return _sar_expr(*to_exprs(high, low), settings)
