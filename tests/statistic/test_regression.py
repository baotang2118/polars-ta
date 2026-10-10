import math
from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, constant, ramp_down, ramp_up
from pytest import mark, raises

from polars_ta import (
    linearreg,
    linearreg_angle,
    linearreg_intercept,
    linearreg_slope,
    tsf,
)

LENGTH: int = 28
BARS: pl.DataFrame = pl.DataFrame({"close": CLOSE[:LENGTH]})

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
LINEARREG_5: list[float | None] = [
    None, None, None, None, 9.8, 9.6, 10.200000000000001, 11.600000000000001,
    11.800000000000002, 10.8, 11.200000000000001, 12.200000000000001, 12.0, 11.0,
    9.200000000000001, 9.6, 10.798, 12.130000000000003,
]
SLOPE_5: list[float | None] = [
    None, None, None, None, 0.0, -0.2, 0.0, 0.6, 0.6, 0.0, 0.0, 0.3, 0.3, -0.1, -0.9,
    -0.6, 0.16599999999999965, 0.699000000000001,
]
INTERCEPT_5: list[float | None] = [
    None, None, None, None, 9.8, 10.4, 10.200000000000001, 9.200000000000001,
    9.400000000000002, 10.8, 11.200000000000001, 11.000000000000002, 10.8, 11.4,
    12.8, 12.0, 10.134000000000002, 9.333999999999998,
]
TSF_5: list[float | None] = [
    None, None, None, None, 9.8, 9.4, 10.200000000000001, 12.200000000000001,
    12.400000000000002, 10.8, 11.200000000000001, 12.500000000000002, 12.3, 10.9, 8.3,
    9.0, 10.964, 12.829000000000002,
]
LINEARREG_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    11.571428571428571, 10.914285714285713, 10.914285714285715, 11.113428571428567,
    11.417428571428571, 12.064857142857145, 12.778285714285714,
]
# fmt: on

FUNCTIONS = (linearreg, linearreg_angle, linearreg_intercept, linearreg_slope, tsf)


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr.alias("out")).to_series().to_list()


class TestLinearreg:
    def test_known_values(self) -> None:
        result = column(linearreg("close", 5))
        assert_values_equal(result[: len(LINEARREG_5)], LINEARREG_5)

    def test_known_values_over_a_longer_window(self) -> None:
        result = column(linearreg("close", 14))
        assert_values_equal(result[: len(LINEARREG_14)], LINEARREG_14)

    def test_a_straight_line_is_reproduced_exactly(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(10)})
        assert_values_equal(column(linearreg("close", 4), bars)[3:], ramp_up(10)[3:])

    def test_a_flat_series_returns_its_own_level(self) -> None:
        bars = pl.DataFrame({"close": constant(8, 7.0)})
        assert_values_equal(column(linearreg("close", 4), bars)[3:], [7.0] * 5)


class TestLinearregSlope:
    def test_known_values(self) -> None:
        result = column(linearreg_slope("close", 5))
        assert_values_equal(result[: len(SLOPE_5)], SLOPE_5)

    def test_a_rising_ramp_climbs_one_unit_a_bar(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(10)})
        assert_values_equal(column(linearreg_slope("close", 4), bars)[3:], [1.0] * 7)

    def test_a_falling_ramp_mirrors_it(self) -> None:
        bars = pl.DataFrame({"close": ramp_down(10)})
        assert_values_equal(column(linearreg_slope("close", 4), bars)[3:], [-1.0] * 7)

    def test_a_flat_series_has_no_slope(self) -> None:
        bars = pl.DataFrame({"close": constant(8, 7.0)})
        assert_values_equal(column(linearreg_slope("close", 4), bars)[3:], [0.0] * 5)


class TestLinearregIntercept:
    def test_known_values(self) -> None:
        result = column(linearreg_intercept("close", 5))
        assert_values_equal(result[: len(INTERCEPT_5)], INTERCEPT_5)

    def test_it_sits_a_window_behind_the_fitted_value(self) -> None:
        window = 5
        intercept = column(linearreg_intercept("close", window))
        fitted = column(linearreg("close", window))
        slope = column(linearreg_slope("close", window))
        expected = [
            None if a is None else a - b * (window - 1)
            for a, b in zip(fitted, cast(list, slope))
        ]
        assert_values_equal(intercept, expected)


class TestLinearregAngle:
    def test_a_unit_slope_is_forty_five_degrees(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(10)})
        assert_values_equal(column(linearreg_angle("close", 4), bars)[3:], [45.0] * 7)

    def test_it_is_the_arctangent_of_the_slope(self) -> None:
        angle = column(linearreg_angle("close", 5))
        slope = column(linearreg_slope("close", 5))
        expected = [
            None if value is None else math.degrees(math.atan(value)) for value in slope
        ]
        assert_values_equal(angle, expected)

    def test_a_flat_series_is_level(self) -> None:
        bars = pl.DataFrame({"close": constant(8, 7.0)})
        assert_values_equal(column(linearreg_angle("close", 4), bars)[3:], [0.0] * 5)


class TestTsf:
    def test_known_values(self) -> None:
        assert_values_equal(column(tsf("close", 5))[: len(TSF_5)], TSF_5)

    def test_it_leads_the_fitted_line_by_one_slope_step(self) -> None:
        forecast = column(tsf("close", 5))
        fitted = column(linearreg("close", 5))
        slope = column(linearreg_slope("close", 5))
        expected = [
            None if a is None else a + b for a, b in zip(fitted, cast(list, slope))
        ]
        assert_values_equal(forecast, expected)

    def test_a_straight_line_is_extended_by_one_bar(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(10)})
        expected = [value + 1.0 for value in ramp_up(10)[3:]]
        assert_values_equal(column(tsf("close", 4), bars)[3:], expected)


class TestSharedBehaviour:
    @mark.parametrize("function", FUNCTIONS)
    @mark.parametrize("window", (2, 5, 14))
    def test_warm_up_is_window_minus_one_nulls(self, function, window: int) -> None:
        result = column(function("close", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    @mark.parametrize("function", FUNCTIONS)
    def test_a_window_too_short_to_fit_a_line_raises(self, function) -> None:
        with raises(ValueError):
            function("close", 1)

    @mark.parametrize("function", FUNCTIONS)
    def test_invalid_window_raises(self, function) -> None:
        with raises(ValueError):
            function("close", 0)

    @mark.parametrize("function", FUNCTIONS)
    def test_series_input_is_rejected(self, function) -> None:
        with raises(TypeError):
            function(cast(Any, pl.Series("close", CLOSE[:LENGTH])))
