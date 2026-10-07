import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, ramp_up, with_null

from polars_ta import midpoint, midprice

VALUES: list[float] = CLOSE[:40]


def evaluate(expr: pl.Expr, values: list[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def reference_midpoint(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(window - 1, len(values)):
        span = values[index - window + 1 : index + 1]
        result[index] = (max(span) + min(span)) / 2.0
    return result


class TestMidpoint(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        for window in (2, 5, 14):
            with self.subTest(window=window):
                self.assert_values_equal(
                    evaluate(midpoint("close", window)),
                    reference_midpoint(VALUES, window),
                )

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        for window in (2, 5, 14):
            with self.subTest(window=window):
                result = evaluate(midpoint("close", window))
                self.assertEqual(result[: window - 1], [None] * (window - 1))
                self.assertIsNotNone(result[window - 1])

    def test_window_of_one_returns_the_input(self) -> None:
        self.assert_values_equal(evaluate(midpoint("close", 1)), VALUES)

    def test_flat_series_equals_its_level(self) -> None:
        self.assert_values_equal(
            evaluate(midpoint("close", 3), constant(6))[2:], [5.0] * 4
        )

    def test_rising_series_is_the_window_centre(self) -> None:
        rising = ramp_up(10, 0.0)
        self.assert_values_equal(
            evaluate(midpoint("close", 3), rising)[2:],
            [index + 1.0 for index in range(8)],
        )

    def test_null_blanks_the_whole_window(self) -> None:
        result = evaluate(midpoint("close", 3), with_null(VALUES, 5))
        self.assertEqual(result[5:8], [None] * 3)
        self.assertIsNotNone(result[8])

    def test_default_window_is_fourteen(self) -> None:
        self.assert_values_equal(
            evaluate(midpoint("close")), evaluate(midpoint("close", 14))
        )

    def test_name_and_expression_agree(self) -> None:
        from_name = evaluate(midpoint("close", 5))
        self.assert_values_equal(evaluate(midpoint(pl.col("close"), 5)), from_name)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            midpoint(pl.Series("close", VALUES), 5)

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                midpoint("close", window)


class TestMidprice(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        bars = frame(high=HIGH[:40], low=LOW[:40])
        for window in (2, 5, 14):
            with self.subTest(window=window):
                result = (
                    bars.select(midprice("high", "low", window)).to_series().to_list()
                )
                expected: list[float | None] = [None] * 40
                for index in range(window - 1, 40):
                    start = index - window + 1
                    expected[index] = (
                        max(HIGH[start : index + 1]) + min(LOW[start : index + 1])
                    ) / 2.0
                self.assert_values_equal(result, expected)

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        bars = frame(high=HIGH[:40], low=LOW[:40])
        for window in (3, 7):
            with self.subTest(window=window):
                result = (
                    bars.select(midprice("high", "low", window)).to_series().to_list()
                )
                self.assertEqual(result[: window - 1], [None] * (window - 1))
                self.assertIsNotNone(result[window - 1])

    def test_equals_midpoint_when_bars_have_no_spread(self) -> None:
        bars = frame(high=VALUES, low=VALUES, close=VALUES)
        self.assert_values_equal(
            bars.select(midprice("high", "low", 5)).to_series().to_list(),
            evaluate(midpoint("close", 5)),
        )

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            midprice(pl.Series("high", HIGH[:40]), pl.Series("low", LOW[:40]), 5)

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                midprice("high", "low", window)
