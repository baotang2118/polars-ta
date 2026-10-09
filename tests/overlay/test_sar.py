from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import HIGH, LOW, ramp_down, ramp_up
from pytest import raises

from polars_ta import sar, sarext

LENGTH: int = 80
BARS: pl.DataFrame = pl.DataFrame({"high": HIGH[:LENGTH], "low": LOW[:LENGTH]})

# Frozen expectations: the warm-up plus the first live bars, including a reversal.
# fmt: off
SAR_SLOW: list[float | None] = [
    None, 8.0, 8.065, 8.2424, 12.5, 12.41, 8.0, 8.09, 8.3164, 8.533744,
    8.742394240000001, 8.75, 9.11,
]
SAR_FAST: list[float | None] = [
    None, 8.0, 8.1625, 12.5, 12.5, 12.05, 8.0, 8.225, 8.7775, 13.75, 8.75, 8.75, 9.35,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestSar(IndicatorAssertions):
    def test_known_values(self) -> None:
        for (acceleration, maximum), expected in (
            ((0.02, 0.2), SAR_SLOW),
            ((0.05, 0.3), SAR_FAST),
        ):
            with self.subTest(acceleration=acceleration, maximum=maximum):
                result = column(sar("high", "low", acceleration, maximum))
                self.assert_values_equal(result[: len(expected)], expected)

    def test_warm_up_is_one_row(self) -> None:
        result = column(sar("high", "low"))
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[1])

    def test_stays_below_price_in_a_sustained_rise(self) -> None:
        rising = ramp_up(40)
        bars = pl.DataFrame({"high": rising, "low": [value - 1.0 for value in rising]})
        result = column(sar("high", "low"), bars)
        for index in range(2, 40):
            self.assertLessEqual(result[index], rising[index])

    def test_stays_above_price_in_a_sustained_fall(self) -> None:
        falling = ramp_down(40, 60.0)
        bars = pl.DataFrame(
            {"high": [value + 1.0 for value in falling], "low": falling}
        )
        result = column(sar("high", "low"), bars)
        for index in range(2, 40):
            self.assertGreaterEqual(result[index], falling[index])

    def test_null_ends_the_scan(self) -> None:
        highs: list[float | None] = list(HIGH[:20])
        lows = list(LOW[:20])
        highs[10] = None
        result = column(sar("high", "low"), pl.DataFrame({"high": highs, "low": lows}))
        self.assertIsNotNone(result[9])
        self.assertEqual(result[10:], [None] * 10)

    def test_too_short_an_input_is_all_null(self) -> None:
        bars = pl.DataFrame({"high": [2.0], "low": [1.0]})
        self.assertEqual(column(sar("high", "low"), bars), [None])

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            sar(
                cast(Any, pl.Series("high", HIGH[:LENGTH])),
                cast(Any, pl.Series("low", LOW[:LENGTH])),
            )

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(BARS).with_columns(sar("high", "low").alias("sar")).collect()
        )
        self.assert_values_equal(collected["sar"].to_list(), column(sar("high", "low")))

    def test_invalid_arguments_raise(self) -> None:
        for acceleration in (0.0, -0.1, "fast"):
            with self.subTest(acceleration=acceleration), raises(ValueError):
                sar("high", "low", cast(Any, acceleration))


class TestSarext(IndicatorAssertions):
    def test_matches_plain_sar_up_to_the_sign(self) -> None:
        plain = column(sar("high", "low"))
        extended = column(sarext("high", "low"))
        self.assert_values_equal(
            [None if value is None else abs(value) for value in extended], plain
        )

    def test_short_readings_are_negative(self) -> None:
        extended = column(sarext("high", "low"))
        self.assertTrue(any(value < 0.0 for value in extended[1:]))
        self.assertTrue(any(value > 0.0 for value in extended[1:]))

    def test_positive_start_value_begins_long(self) -> None:
        rising = ramp_up(30)
        bars = pl.DataFrame({"high": rising, "low": [value - 1.0 for value in rising]})
        result = column(sarext("high", "low", start_value=0.5), bars)
        self.assertGreater(result[1], 0.0)

    def test_negative_start_value_begins_short(self) -> None:
        rising = ramp_up(30)
        bars = pl.DataFrame({"high": rising, "low": [value - 1.0 for value in rising]})
        result = column(sarext("high", "low", start_value=-100.0), bars)
        self.assertLess(result[1], 0.0)

    def test_offset_on_reverse_widens_the_stop(self) -> None:
        plain = column(sarext("high", "low"))
        offset = column(sarext("high", "low", offset_on_reverse=0.05))
        self.assertNotEqual(plain, offset)

    def test_asymmetric_acceleration_changes_the_result(self) -> None:
        symmetric = column(sarext("high", "low"))
        skewed = column(
            sarext(
                "high",
                "low",
                acceleration_long=0.05,
                acceleration_max_long=0.5,
            )
        )
        self.assertNotEqual(symmetric, skewed)

    def test_invalid_arguments_raise(self) -> None:
        with raises(ValueError):
            sarext("high", "low", acceleration_long=0.0)
        with raises(ValueError):
            sarext("high", "low", offset_on_reverse=-0.1)
