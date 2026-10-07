from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, constant, frame, ramp_up

from polars_ta import mass, vortex

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
VORTEX_PLUS_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    0.95, 0.9696969696969697, 0.9771428571428571, 1.027515923566879,
    1.0630806845965772, 1.045107398568019, 1.028117359413203, 1.0343221377788674,
    0.9895138226882746, 1.053079044117647, 1.0648212226066895, 1.028099173553719,
]
VORTEX_MINUS_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    0.95, 0.9212121212121213, 0.9607453416149069, 0.9087898089171975,
    0.7951100244498777, 0.8164677804295943, 0.8789731051344745, 0.828879627359647,
    0.8217349857006674, 0.7392003676470588, 0.7344867358708189, 0.766469893742621,
]
MASS_9_25: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None,
    25.081966775478254, 25.05878511176363, 25.070964465811787, 25.104295617612685,
    25.034083634080908, 25.016975913643396, 25.03462764531113, 25.073058837956353,
    25.007612010927936, 24.994344748903107, 25.015315383174944, 25.056718582897954,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def field(name: str, expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    return column(expr.struct.field(name), bars)


class TestVortex(IndicatorAssertions):
    def test_known_values(self) -> None:
        indicator = vortex("high", "low", "close", 14)
        plus = field("plus", indicator)
        minus = field("minus", indicator)
        self.assert_values_equal(plus[: len(VORTEX_PLUS_14)], VORTEX_PLUS_14)
        self.assert_values_equal(minus[: len(VORTEX_MINUS_14)], VORTEX_MINUS_14)

    def test_warm_up_is_the_window(self) -> None:
        for window in (5, 14, 21):
            with self.subTest(window=window):
                indicator = vortex("high", "low", "close", window)
                for name in ("plus", "minus"):
                    result = field(name, indicator)
                    self.assertEqual(result[:window], [None] * window)
                    self.assertIsNotNone(result[window])

    def test_a_rising_series_favours_the_plus_line(self) -> None:
        rising = ramp_up(20)
        bars = pl.DataFrame(
            {
                "high": rising,
                "low": [value - 1.0 for value in rising],
                "close": rising,
            }
        )
        indicator = vortex("high", "low", "close", 5)
        plus = field("plus", indicator, bars)
        minus = field("minus", indicator, bars)
        self.assertGreater(plus[-1], minus[-1])

    def test_a_flat_series_reports_zero(self) -> None:
        flat = constant(12, 4.0)
        bars = pl.DataFrame({"high": flat, "low": flat, "close": flat})
        indicator = vortex("high", "low", "close", 4)
        self.assert_values_equal(field("plus", indicator, bars), [None] * 4 + [0.0] * 8)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            vortex("high", "low", "close", 0)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            vortex(cast(Any, pl.Series("high", HIGH[:5])), "low", "close")


class TestMass(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = column(mass("high", "low", 9, 25))
        self.assert_values_equal(result[: len(MASS_9_25)], MASS_9_25)

    def test_warm_up_covers_both_averages_and_the_sum(self) -> None:
        for fast, slow in ((3, 4), (5, 10), (9, 25)):
            with self.subTest(fast=fast, slow=slow):
                result = column(mass("high", "low", fast, slow))
                warm_up = 2 * (fast - 1) + slow - 1
                self.assertEqual(result[:warm_up], [None] * warm_up)
                self.assertIsNotNone(result[warm_up])

    def test_a_constant_range_sums_to_the_slow_period(self) -> None:
        middle = ramp_up(40)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in middle],
                "low": [value - 1.0 for value in middle],
            }
        )
        self.assertAlmostEqual(column(mass("high", "low", 3, 4), bars)[-1], 4.0)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            mass("high", "low", 0, 25)
