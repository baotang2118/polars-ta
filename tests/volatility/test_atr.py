from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, frame_from, with_null
from pytest import raises

from polars_ta import atr, true_range

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
TRUE_RANGE: list[float | None] = [
    None, 2.5, 3.0, 3.5, 2.0, 2.5, 3.0, 3.5, 2.0, 2.5, 3.5, 3.5, 3.0,
]
TRUE_RANGE_NULL_HIGH: list[float | None] = [
    None, 2.5, 3.0, 3.5, None, 2.5, 3.0, 3.5, 2.0, 2.5, 3.5, 3.5, 3.0, 2.5, 3.0, 3.75,
]
ATR_5: list[float | None] = [
    None, None, None, None, None, 2.7, 2.7600000000000002, 2.9080000000000004,
    2.7264000000000004, 2.6811200000000004, 2.8448960000000003, 2.9759168000000003,
    2.9807334400000003, 2.884586752, 2.9076694016, 3.0761355212800003,
    2.8609084170240004,
]
# A null high at index 2 delays the seed by one bar.
ATR_3_NULL_HIGH: list[float | None] = [
    None, None, None, None, None, 2.6666666666666665, 2.7777777777777772,
    3.0185185185185177, 2.6790123456790114, 2.6193415637860076, 2.9128943758573382,
    3.1085962505715585, 3.072397500381039, 2.881598333587359, 2.921065555724906,
    3.1973770371499377, 2.7982513580999586,
]
# fmt: on


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None):
    return (data if data is not None else frame()).select(expr).to_series().to_list()


class TestTrueRange(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = evaluate(true_range("high", "low", "close"))
        self.assert_values_equal(result[: len(TRUE_RANGE)], TRUE_RANGE)

    def test_first_row_is_null(self) -> None:
        self.assertIsNone(evaluate(true_range("high", "low", "close"))[0])

    def test_is_never_negative(self) -> None:
        for value in evaluate(true_range("high", "low", "close"))[1:]:
            self.assertGreaterEqual(value, 0.0)

    def test_gap_up_widens_the_range_beyond_the_bar(self) -> None:
        data = frame([10.0, 20.0], [9.0, 19.0], [9.5, 19.5])
        # The bar spans 1.0, but the gap from the previous close is 10.5.
        self.assertAlmostEqual(
            evaluate(true_range("high", "low", "close"), data)[1], 10.5
        )

    def test_null_input_propagates(self) -> None:
        high = with_null(HIGH, 4)
        result = evaluate(true_range("high", "low", "close"), frame(high=high))
        self.assert_values_equal(
            result[: len(TRUE_RANGE_NULL_HIGH)], TRUE_RANGE_NULL_HIGH
        )

    def test_names_and_expressions_agree(self) -> None:
        from_names = evaluate(true_range("high", "low", "close"))
        from_exprs = evaluate(
            true_range(pl.col("high"), pl.col("low"), pl.col("close"))
        )
        self.assert_values_equal(from_exprs, from_names)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            true_range(
                cast(Any, pl.Series("high", HIGH)),
                cast(Any, pl.Series("low", LOW)),
                cast(Any, pl.Series("close", CLOSE)),
            )


class TestAtr(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = evaluate(atr("high", "low", "close", 5))
        self.assert_values_equal(result[: len(ATR_5)], ATR_5)

    def test_warm_up_is_window_nulls(self) -> None:
        for window in (3, 5, 7):
            with self.subTest(window=window):
                result = evaluate(atr("high", "low", "close", window))
                self.assertEqual(result[:window], [None] * window)
                self.assertIsNotNone(result[window])

    def test_is_never_negative(self) -> None:
        for value in evaluate(atr("high", "low", "close", 5)):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)

    def test_constant_range_gives_that_range(self) -> None:
        data = frame_from(constant(8), 1.0)
        self.assert_values_equal(
            evaluate(atr("high", "low", "close", 3), data)[3:], [2.0] * 5
        )

    def test_seed_is_the_mean_of_the_first_true_ranges(self) -> None:
        # The first five true ranges are 2.5, 3.0, 3.5, 2.0 and 2.5.
        result = evaluate(atr("high", "low", "close", 5))
        self.assertAlmostEqual(result[5], 2.7, places=10)

    def test_null_input_delays_the_seed(self) -> None:
        high = with_null(HIGH, 2)
        result = evaluate(atr("high", "low", "close", 3), frame(high=high))
        self.assert_values_equal(result[: len(ATR_3_NULL_HIGH)], ATR_3_NULL_HIGH)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        self.assertEqual(
            evaluate(atr("high", "low", "close", len(HIGH))), [None] * len(HIGH)
        )

    def test_default_window_is_fourteen(self) -> None:
        self.assert_values_equal(
            evaluate(atr("high", "low", "close")),
            evaluate(atr("high", "low", "close", 14)),
        )

    def test_names_and_expressions_agree(self) -> None:
        from_names = evaluate(atr("high", "low", "close", 5))
        from_exprs = evaluate(atr(pl.col("high"), pl.col("low"), pl.col("close"), 5))
        self.assert_values_equal(from_exprs, from_names)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            atr(
                cast(Any, pl.Series("high", HIGH)),
                cast(Any, pl.Series("low", LOW)),
                cast(Any, pl.Series("close", CLOSE)),
                5,
            )

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(atr("high", "low", "close", 5).alias("atr"))
            .collect()
        )
        self.assert_values_equal(collected["atr"].to_list()[: len(ATR_5)], ATR_5)

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), raises(ValueError):
                atr("high", "low", "close", cast(Any, window))
