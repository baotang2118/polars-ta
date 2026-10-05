import polars as pl
from _assertions import IndicatorAssertions
from _data import HAND_CHECKED, WILDER_CLOSE, constant, ramp_down, ramp_up

from polars_ta import rsi

VALUES: list[float] = HAND_CHECKED[:10]


def evaluate(expr: pl.Expr, values: list[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def reference_rsi(values: list[float], window: int) -> list[float | None]:
    """Wilder's RSI, seeded with the mean of the first ``window`` changes."""
    result: list[float | None] = [None] * len(values)
    if len(values) <= window:
        return result
    gains = [max(values[i] - values[i - 1], 0.0) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], 0.0) for i in range(1, len(values))]
    average_gain = sum(gains[:window]) / window
    average_loss = sum(losses[:window]) / window
    for index in range(window, len(values)):
        if index > window:
            change = index - 1
            average_gain = (average_gain * (window - 1) + gains[change]) / window
            average_loss = (average_loss * (window - 1) + losses[change]) / window
        total = average_gain + average_loss
        result[index] = 50.0 if total == 0.0 else 100.0 * average_gain / total
    return result


class TestRsi(IndicatorAssertions):
    def test_matches_wilders_published_value(self) -> None:
        result = evaluate(rsi("close", 14), WILDER_CLOSE)
        self.assertAlmostEqual(result[14], 70.4641, places=4)

    def test_matches_reference_recursion(self) -> None:
        self.assert_values_equal(
            evaluate(rsi("close", 14), WILDER_CLOSE), reference_rsi(WILDER_CLOSE, 14)
        )

    def test_matches_reference_on_a_short_window(self) -> None:
        self.assert_values_equal(evaluate(rsi("close", 3)), reference_rsi(VALUES, 3))

    def test_warm_up_is_window_nulls(self) -> None:
        for window in (2, 3, 5):
            with self.subTest(window=window):
                result = evaluate(rsi("close", window))
                self.assertEqual(result[:window], [None] * window)
                self.assertIsNotNone(result[window])

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        for value in evaluate(rsi("close", 3)):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)

    def test_monotonic_rise_reaches_one_hundred(self) -> None:
        rising = ramp_up(10, 0.0)
        self.assert_values_equal(evaluate(rsi("close", 3), rising)[3:], [100.0] * 7)

    def test_monotonic_fall_reaches_zero(self) -> None:
        falling = ramp_down(10)
        self.assert_values_equal(evaluate(rsi("close", 3), falling)[3:], [0.0] * 7)

    def test_flat_series_reports_the_neutral_fifty(self) -> None:
        self.assert_values_equal(evaluate(rsi("close", 3), constant(8))[3:], [50.0] * 5)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        self.assertEqual(evaluate(rsi("close", len(VALUES))), [None] * len(VALUES))

    def test_null_delays_the_seed(self) -> None:
        values = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = evaluate(rsi("close", 2), values)
        self.assertEqual(result[:5], [None] * 5)
        self.assertIsNotNone(result[5])

    def test_default_window_is_fourteen(self) -> None:
        self.assert_values_equal(
            evaluate(rsi("close"), WILDER_CLOSE),
            evaluate(rsi("close", 14), WILDER_CLOSE),
        )

    def test_name_expression_and_series_agree(self) -> None:
        from_name = evaluate(rsi("close", 3))
        self.assert_values_equal(evaluate(rsi(pl.col("close"), 3)), from_name)
        self.assert_values_equal(
            rsi(pl.Series("close", VALUES), 3).to_list(), from_name
        )

    def test_series_input_keeps_its_name(self) -> None:
        result = rsi(pl.Series("close", VALUES), 3)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(rsi("close", 3).alias("rsi"))
            .collect()
        )
        self.assert_values_equal(collected["rsi"].to_list(), reference_rsi(VALUES, 3))

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                rsi("close", window)
