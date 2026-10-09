from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import (
    CLOSE,
    HIGH,
    LOW,
    constant,
    frame,
    frame_from,
    ramp_down,
    ramp_up,
    with_null,
)
from pytest import raises

from polars_ta import stoch

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
STOCH_K: list[float | None] = [
    None, None, None, None, None, None, None, None, 62.80193236714976,
    48.91304347826087, 47.391304347826086, 53.61111111111111, 57.77777777777777,
    43.05555555555555, 26.340996168582375, 29.93295019157088, 47.305966064586755,
    66.88967749834356, 78.23483039359404, 79.24887441998375,
]
STOCH_D: list[float | None] = [
    None, None, None, None, None, None, None, None, 55.823939881910896,
    57.31347289318304, 53.03542673107891, 49.97181964573269, 52.92673107890499,
    51.481481481481474, 42.39144316730523, 33.10983397190294, 34.526637474913336,
    48.04286458483373, 64.14349131884144, 74.79112743730711,
]
# A null close at index 6 blanks every window that overlaps it.
STOCH_K_NULL_CLOSE: list[float | None] = [
    None, None, None, None, 31.69934640522876, 37.77777777777778, None, None, None,
    30.147058823529413, 46.71052631578947, 69.62719298245614, 45.94298245614035,
    20.94298245614035, 27.083333333333336, 50.0,
]
# fmt: on


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None) -> pl.DataFrame:
    source = data if data is not None else frame()
    return source.select(expr.alias("s")).unnest("s")


class TestStoch(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = evaluate(stoch("high", "low", "close", 5, 3, 3))
        self.assert_values_equal(result["k"].to_list()[: len(STOCH_K)], STOCH_K)
        self.assert_values_equal(result["d"].to_list()[: len(STOCH_D)], STOCH_D)

    def test_field_names_and_order(self) -> None:
        self.assertEqual(evaluate(stoch("high", "low", "close")).columns, ["k", "d"])

    def test_warm_up_matches_the_talib_lookback(self) -> None:
        for fastk, slowk, slowd in ((5, 3, 3), (3, 2, 2), (4, 1, 1)):
            with self.subTest(periods=(fastk, slowk, slowd)):
                result = evaluate(stoch("high", "low", "close", fastk, slowk, slowd))
                lookback = (fastk - 1) + (slowk - 1) + (slowd - 1)
                self.assertEqual(result["k"].to_list()[:lookback], [None] * lookback)
                self.assertIsNotNone(result["k"][lookback])
                self.assertIsNotNone(result["d"][lookback])

    def test_both_lines_start_on_the_same_row(self) -> None:
        result = evaluate(stoch("high", "low", "close", 5, 3, 3))
        self.assertEqual(result["k"].null_count(), result["d"].null_count())

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        result = evaluate(stoch("high", "low", "close", 5, 3, 3))
        for field in ("k", "d"):
            for value in result[field].to_list():
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_close_at_the_window_high_is_one_hundred(self) -> None:
        rising = ramp_up(10)
        result = evaluate(
            stoch("high", "low", "close", 3, 1, 1), frame_from(rising, 0.0)
        )
        self.assert_values_equal(result["k"].to_list()[2:], [100.0] * 8)

    def test_close_at_the_window_low_is_zero(self) -> None:
        falling = ramp_down(10)
        result = evaluate(
            stoch("high", "low", "close", 3, 1, 1), frame_from(falling, 0.0)
        )
        self.assert_values_equal(result["k"].to_list()[2:], [0.0] * 8)

    def test_flat_range_reports_zero(self) -> None:
        data = frame_from(constant(8), 0.0)
        result = evaluate(stoch("high", "low", "close", 3, 2, 2), data)
        self.assert_values_equal(result["k"].to_list()[4:], [0.0] * 4)

    def test_null_input_propagates(self) -> None:
        close = with_null(CLOSE, 6)
        result = evaluate(stoch("high", "low", "close", 3, 2, 2), frame(close=close))
        self.assert_values_equal(
            result["k"].to_list()[: len(STOCH_K_NULL_CLOSE)], STOCH_K_NULL_CLOSE
        )

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        result = evaluate(stoch("high", "low", "close", len(HIGH), 3, 3))
        self.assertEqual(result["k"].to_list(), [None] * len(HIGH))

    def test_names_and_expressions_agree(self) -> None:
        from_names = evaluate(stoch("high", "low", "close", 5, 3, 3))["k"].to_list()
        from_exprs = evaluate(
            stoch(pl.col("high"), pl.col("low"), pl.col("close"), 5, 3, 3)
        )["k"].to_list()
        self.assert_values_equal(from_exprs, from_names)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            stoch(
                cast(Any, pl.Series("high", HIGH)),
                cast(Any, pl.Series("low", LOW)),
                cast(Any, pl.Series("close", CLOSE)),
                5,
                3,
                3,
            )

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(stoch("high", "low", "close", 5, 3, 3).alias("s"))
            .collect()
        )
        self.assertEqual(collected.columns[-1], "s")

    def test_invalid_periods_raise(self) -> None:
        for periods in ((0, 3, 3), (5, 0, 3), (5, 3, -1), (5, 2.5, 3)):
            with self.subTest(periods=periods), raises(ValueError):
                stoch("high", "low", "close", *cast(Any, periods))
