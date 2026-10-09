from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import HIGH, LOW, OPEN, constant, frame, ramp_down, ramp_up
from pytest import raises

from polars_ta import aroon, aroonosc, bop

LENGTH: int = 60

# Frozen expectations: each table covers the warm-up plus the first live bars.
# fmt: off
AROON_DOWN: dict[int, list[float | None]] = {
    3: [
        None, None, None, 0.0, 100.0, 66.66666666666667, 33.333333333333336, 0.0, 0.0,
        100.0, 66.66666666666667, 33.333333333333336, 0.0, 100.0, 100.0,
    ],
    5: [
        None, None, None, None, None, 80.0, 60.0, 40.0, 20.0, 0.0, 80.0, 60.0, 40.0,
        100.0, 100.0, 80.0, 60.0,
    ],
    14: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, 100.0, 92.85714285714286, 85.71428571428572, 78.57142857142857,
        71.42857142857143, 64.28571428571429, 57.142857142857146, 50.0,
        42.85714285714286, 35.714285714285715, 28.571428571428573, 21.42857142857143,
    ],
}
AROON_UP: dict[int, list[float | None]] = {
    3: [
        None, None, None, 66.66666666666667, 33.333333333333336, 0.0, 100.0, 100.0,
        66.66666666666667, 33.333333333333336, 0.0, 100.0, 66.66666666666667,
        33.333333333333336, 0.0,
    ],
    5: [
        None, None, None, None, None, 40.0, 100.0, 100.0, 80.0, 60.0, 40.0, 100.0,
        80.0, 60.0, 40.0, 20.0, 0.0,
    ],
    14: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, 78.57142857142857, 71.42857142857143, 64.28571428571429,
        57.142857142857146, 100.0, 100.0, 92.85714285714286, 100.0, 92.85714285714286,
        100.0, 100.0, 92.85714285714286,
    ],
}
# The canonical bars repeat a four-bar spread cycle, so BOP repeats with it.
BOP: list[float | None] = [
    -0.25, 0.0, 0.25, -0.25, 0.0, 0.25, -0.25, 0.0, 0.25, -0.25, 0.0, 0.25, -0.25, 0.0,
    0.25, -0.25,
]
# fmt: on


def unnest(expr: pl.Expr, bars: pl.DataFrame) -> dict[str, list]:
    result = bars.select(expr.alias("out")).unnest("out")
    return {name: result[name].to_list() for name in result.columns}


class TestAroon(IndicatorAssertions):
    def test_known_values(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        for window in (3, 5, 14):
            with self.subTest(window=window):
                fields = unnest(aroon("high", "low", window), bars)
                down, up = AROON_DOWN[window], AROON_UP[window]
                self.assert_values_equal(fields["down"][: len(down)], down)
                self.assert_values_equal(fields["up"][: len(up)], up)

    def test_warm_up_is_window_nulls(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        for window in (3, 5, 14):
            with self.subTest(window=window):
                fields = unnest(aroon("high", "low", window), bars)
                for name in ("down", "up"):
                    self.assertEqual(fields[name][:window], [None] * window)
                    self.assertIsNotNone(fields[name][window])

    def test_rising_series_pins_up_high_and_down_low(self) -> None:
        rising = ramp_up(30)
        bars = frame(high=rising, low=rising, close=rising)
        fields = unnest(aroon("high", "low", 5), bars)
        self.assert_values_equal(fields["up"][5:], [100.0] * 25)
        self.assert_values_equal(fields["down"][5:], [0.0] * 25)

    def test_falling_series_pins_down_high_and_up_low(self) -> None:
        falling = ramp_down(30, 40.0)
        bars = frame(high=falling, low=falling, close=falling)
        fields = unnest(aroon("high", "low", 5), bars)
        self.assert_values_equal(fields["down"][5:], [100.0] * 25)
        self.assert_values_equal(fields["up"][5:], [0.0] * 25)

    def test_flat_series_ties_resolve_to_the_newest_bar(self) -> None:
        flat = constant(20, 3.0)
        bars = frame(high=flat, low=flat, close=flat)
        fields = unnest(aroon("high", "low", 5), bars)
        self.assert_values_equal(fields["up"][5:], [100.0] * 15)
        self.assert_values_equal(fields["down"][5:], [100.0] * 15)

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        fields = unnest(aroon("high", "low"), bars)
        for name in ("down", "up"):
            for value in fields[name]:
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_oscillator_is_up_minus_down(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        fields = unnest(aroon("high", "low", 7), bars)
        expected = [
            None if u is None else u - d for u, d in zip(fields["up"], fields["down"])
        ]
        result = bars.select(aroonosc("high", "low", 7)).to_series().to_list()
        self.assert_values_equal(result, expected)

    def test_null_blanks_the_whole_window(self) -> None:
        highs: list[float | None] = list(HIGH[:LENGTH])
        highs[10] = None
        bars = pl.DataFrame({"high": highs, "low": LOW[:LENGTH]})
        fields = unnest(aroon("high", "low", 4), bars)
        self.assertEqual(fields["up"][10:15], [None] * 5)
        self.assertIsNotNone(fields["up"][15])

    def test_default_window_is_fourteen(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        self.assert_values_equal(
            bars.select(aroonosc("high", "low")).to_series().to_list(),
            bars.select(aroonosc("high", "low", 14)).to_series().to_list(),
        )

    def test_invalid_window_raises(self) -> None:
        for function in (aroon, aroonosc):
            with (
                self.subTest(function=function.__name__),
                raises(ValueError),
            ):
                function("high", "low", 0)


class TestBop(IndicatorAssertions):
    def test_known_values(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assert_values_equal(result[: len(BOP)], BOP)

    def test_has_no_warm_up(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assertIsNotNone(result[0])

    def test_output_stays_within_minus_one_and_one(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        for value in bars.select(bop("open", "high", "low", "close")).to_series():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_bar_closing_at_its_high_reports_one(self) -> None:
        bars = pl.DataFrame(
            {"open": [1.0], "high": [3.0], "low": [1.0], "close": [3.0]}
        )
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assert_values_equal(result, [1.0])

    def test_bar_with_no_range_reports_zero(self) -> None:
        flat = constant(5, 2.0)
        bars = frame(high=flat, low=flat, close=flat, open_=flat)
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assert_values_equal(result, [0.0] * 5)

    def test_null_propagates(self) -> None:
        bars = pl.DataFrame(
            {
                "open": [1.0, None],
                "high": [3.0, 3.0],
                "low": [1.0, 1.0],
                "close": [3.0, 2.0],
            }
        )
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assertIsNone(result[1])

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            bop(cast(Any, pl.Series("open", OPEN[:5])), "high", "low", "close")
