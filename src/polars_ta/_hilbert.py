"""Ehlers' Hilbert transform, the shared engine behind the cycle indicators.

TA-Lib derives ``HT_DCPERIOD``, ``HT_DCPHASE``, ``HT_PHASOR``, ``HT_SINE``,
``HT_TRENDMODE``, ``HT_TRENDLINE``, and ``MAMA`` from one recursion. Running it
once here keeps the arithmetic in a single place; each public indicator then
selects the series it needs and applies its own lookback.
"""

from __future__ import annotations

import math
from typing import NamedTuple

_A = 0.0962
_B = 0.5769
_RAD2DEG = 180.0 / math.pi
_DEG2RAD = math.pi / 180.0
_TWO_PI = 2.0 * math.pi

WMA_PRIMING = 12
"""Bars consumed priming the four-period price smoother before the recursion."""

SHORT_LOOKBACK = 32
"""Leading nulls for the indicators that need only the dominant cycle period."""

LONG_LOOKBACK = 63
"""Leading nulls for the indicators that also need the dominant cycle phase."""

_SMOOTH_BUFFER = 50


class HilbertSeries(NamedTuple):
    """Every per-bar series the transform produces, before any lookback mask."""

    smooth_period: list[float | None]
    in_phase: list[float | None]
    quadrature: list[float | None]
    dc_phase: list[float | None]
    sine: list[float | None]
    lead_sine: list[float | None]
    trend_mode: list[int | None]
    trendline: list[float | None]
    mama: list[float | None]
    fama: list[float | None]


def _empty(size: int) -> HilbertSeries:
    return HilbertSeries(*([None] * size for _ in range(len(HilbertSeries._fields))))


class _Quadrature:
    """One Hilbert transform channel, with separate odd- and even-bar buffers."""

    __slots__ = ("buffers", "previous", "previous_input")

    def __init__(self) -> None:
        self.buffers = {"odd": [0.0, 0.0, 0.0], "even": [0.0, 0.0, 0.0]}
        self.previous = {"odd": 0.0, "even": 0.0}
        self.previous_input = {"odd": 0.0, "even": 0.0}

    def step(self, value: float, parity: str, scale: float, slot: int) -> float:
        weighted = _A * value
        buffer = self.buffers[parity]
        result = weighted - buffer[slot] - self.previous[parity]
        buffer[slot] = weighted
        self.previous[parity] = _B * self.previous_input[parity]
        result += self.previous[parity]
        self.previous_input[parity] = value
        return result * scale


def hilbert_transform(
    values: list[float | None], fast_limit: float, slow_limit: float
) -> HilbertSeries:
    """Run the transform over ``values``, stopping at the first null."""
    size = len(values)
    series = _empty(size)
    usable = size
    for index, value in enumerate(values):
        if value is None:
            usable = index
            break
    if usable <= WMA_PRIMING:
        return series

    trailing_index = 0
    today = 0
    first = values[today]
    today += 1
    wma_sub = first
    wma_sum = first
    second = values[today]
    today += 1
    wma_sub += second
    wma_sum += second * 2.0
    third = values[today]
    today += 1
    wma_sub += third
    wma_sum += third * 3.0
    trailing_value = 0.0
    for _ in range(9):
        price = values[today]
        today += 1
        wma_sub += price - trailing_value
        wma_sum += price * 4.0
        trailing_value = values[trailing_index]
        trailing_index += 1
        wma_sum -= wma_sub

    channels = {name: _Quadrature() for name in ("detrender", "q1", "ji", "jq")}
    slot = 0
    period = 0.0
    smooth_period = 0.0
    previous_q2 = previous_i2 = 0.0
    real = imaginary = 0.0
    odd_previous2 = odd_previous3 = 0.0
    even_previous2 = even_previous3 = 0.0
    smooth_price = [0.0] * _SMOOTH_BUFFER
    smooth_slot = 0
    dc_phase = 0.0
    sine = lead_sine = 0.0
    trend1 = trend2 = trend3 = 0.0
    days_in_trend = 0
    trendline = 0.0
    previous_mama = previous_fama = 0.0
    previous_phase = 0.0

    while today < usable:
        scale = 0.075 * period + 0.54
        price = values[today]
        wma_sub += price - trailing_value
        wma_sum += price * 4.0
        trailing_value = values[trailing_index]
        trailing_index += 1
        smoothed = wma_sum * 0.1
        wma_sum -= wma_sub
        smooth_price[smooth_slot] = smoothed

        even = today % 2 == 0
        parity = "even" if even else "odd"
        detrender = channels["detrender"].step(smoothed, parity, scale, slot)
        quadrature = channels["q1"].step(detrender, parity, scale, slot)
        in_phase = even_previous3 if even else odd_previous3
        rotated_i = channels["ji"].step(in_phase, parity, scale, slot)
        rotated_q = channels["jq"].step(quadrature, parity, scale, slot)
        if even:
            slot = (slot + 1) % 3
        smoothed_q2 = 0.2 * (quadrature + rotated_i) + 0.8 * previous_q2
        smoothed_i2 = 0.2 * (in_phase - rotated_q) + 0.8 * previous_i2
        if even:
            odd_previous3, odd_previous2 = odd_previous2, detrender
        else:
            even_previous3, even_previous2 = even_previous2, detrender

        phase = (
            math.atan(quadrature / in_phase) * _RAD2DEG if in_phase != 0.0 else 0.0
        )

        real = 0.2 * (smoothed_i2 * previous_i2 + smoothed_q2 * previous_q2) + 0.8 * real
        imaginary = (
            0.2 * (smoothed_i2 * previous_q2 - smoothed_q2 * previous_i2)
            + 0.8 * imaginary
        )
        previous_q2, previous_i2 = smoothed_q2, smoothed_i2
        earlier_period = period
        if imaginary != 0.0 and real != 0.0:
            period = 360.0 / (math.atan(imaginary / real) * _RAD2DEG)
        period = min(period, 1.5 * earlier_period)
        period = max(period, 0.67 * earlier_period)
        period = min(max(period, 6.0), 50.0)
        period = 0.2 * period + 0.8 * earlier_period
        smooth_period = 0.33 * period + 0.67 * smooth_period

        # MAMA adapts its smoothing factor to how fast the phase is turning.
        turn = max(previous_phase - phase, 1.0)
        previous_phase = phase
        alpha = fast_limit if turn <= 1.0 else max(fast_limit / turn, slow_limit)
        mama = alpha * price + (1.0 - alpha) * previous_mama
        previous_mama = mama
        half = alpha * 0.5
        fama = half * mama + (1.0 - half) * previous_fama
        previous_fama = fama

        earlier_phase = dc_phase
        cycle_length = int(smooth_period + 0.5)
        real_part = imaginary_part = 0.0
        cursor = smooth_slot
        for step in range(cycle_length):
            angle = step * _TWO_PI / cycle_length
            sampled = smooth_price[cursor]
            real_part += math.sin(angle) * sampled
            imaginary_part += math.cos(angle) * sampled
            cursor = _SMOOTH_BUFFER - 1 if cursor == 0 else cursor - 1
        magnitude = abs(imaginary_part)
        if magnitude > 0.0:
            dc_phase = math.atan(real_part / imaginary_part) * _RAD2DEG
        elif real_part < 0.0:
            dc_phase -= 90.0
        elif real_part > 0.0:
            dc_phase += 90.0
        dc_phase += 90.0
        # Compensate for the one-bar lag of the price smoother.
        dc_phase += 360.0 / smooth_period
        if imaginary_part < 0.0:
            dc_phase += 180.0
        if dc_phase > 315.0:
            dc_phase -= 360.0

        earlier_sine, earlier_lead = sine, lead_sine
        sine = math.sin(dc_phase * _DEG2RAD)
        lead_sine = math.sin((dc_phase + 45.0) * _DEG2RAD)

        # The trendline averages the raw price, not the smoothed price.
        total = 0.0
        for step in range(min(cycle_length, _SMOOTH_BUFFER)):
            if today - step >= 0:
                total += values[today - step]
        if cycle_length > 0:
            total /= cycle_length
        trendline = (4.0 * total + 3.0 * trend1 + 2.0 * trend2 + trend3) / 10.0
        trend3, trend2, trend1 = trend2, trend1, total

        mode = 1
        crossed = (sine > lead_sine and earlier_sine <= earlier_lead) or (
            sine < lead_sine and earlier_sine >= earlier_lead
        )
        if crossed:
            days_in_trend = 0
            mode = 0
        days_in_trend += 1
        if days_in_trend < 0.5 * smooth_period:
            mode = 0
        turned = dc_phase - earlier_phase
        if smooth_period != 0.0 and (
            0.67 * 360.0 / smooth_period < turned < 1.5 * 360.0 / smooth_period
        ):
            mode = 0
        if trendline != 0.0 and abs((smoothed - trendline) / trendline) >= 0.015:
            mode = 1

        series.smooth_period[today] = smooth_period
        series.in_phase[today] = in_phase
        series.quadrature[today] = quadrature
        series.dc_phase[today] = dc_phase
        series.sine[today] = sine
        series.lead_sine[today] = lead_sine
        series.trend_mode[today] = mode
        series.trendline[today] = trendline
        series.mama[today] = mama
        series.fama[today] = fama

        smooth_slot = (smooth_slot + 1) % _SMOOTH_BUFFER
        today += 1

    return series


def mask_lookback(values: list, lookback: int) -> list:
    """Blank the leading rows TA-Lib does not emit for a given indicator."""
    return [None] * min(lookback, len(values)) + values[lookback:]
