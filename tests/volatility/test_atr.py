import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, frame_from, with_null

from polars_ta import atr, true_range


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None):
    return (data if data is not None else frame()).select(expr).to_series().to_list()


def reference_true_range(high, low, close) -> list[float | None]:
    result: list[float | None] = [None]
    for index in range(1, len(close)):
        previous = close[index - 1]
        if None in (high[index], low[index], previous):
            result.append(None)
            continue
        result.append(
            max(
                high[index] - low[index],
                abs(high[index] - previous),
                abs(low[index] - previous),
            )
        )
    return result


def reference_atr(high, low, close, window: int) -> list[float | None]:
    ranges = reference_true_range(high, low, close)
    result: list[float | None] = [None] * len(ranges)
    seed_index = None
    for index in range(window - 1, len(ranges)):
        chunk = ranges[index + 1 - window : index + 1]
        if all(v is not None for v in chunk):
            seed_index = index
            previous = sum(chunk) / window
            break
    if seed_index is None:
        return result
    result[seed_index] = previous
    for index in range(seed_index + 1, len(ranges)):
        previous = (previous * (window - 1) + ranges[index]) / window
        result[index] = previous
    return result


class TestTrueRange(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            evaluate(true_range("high", "low", "close")),
            reference_true_range(HIGH, LOW, CLOSE),
        )

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
        self.assert_values_equal(result, reference_true_range(high, LOW, CLOSE))

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(true_range("high", "low", "close"))
        from_series = true_range(
            pl.Series("high", HIGH), pl.Series("low", LOW), pl.Series("close", CLOSE)
        )
        self.assert_values_equal(from_series.to_list(), from_names)


class TestAtr(IndicatorAssertions):
    def test_matches_wilder_smoothed_reference(self) -> None:
        self.assert_values_equal(
            evaluate(atr("high", "low", "close", 5)), reference_atr(HIGH, LOW, CLOSE, 5)
        )

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
        ranges = reference_true_range(HIGH, LOW, CLOSE)
        result = evaluate(atr("high", "low", "close", 5))
        self.assertAlmostEqual(result[5], sum(ranges[1:6]) / 5, places=10)

    def test_null_input_delays_the_seed(self) -> None:
        high = with_null(HIGH, 2)
        result = evaluate(atr("high", "low", "close", 3), frame(high=high))
        self.assert_values_equal(result, reference_atr(high, LOW, CLOSE, 3))

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        self.assertEqual(
            evaluate(atr("high", "low", "close", len(HIGH))), [None] * len(HIGH)
        )

    def test_default_window_is_fourteen(self) -> None:
        self.assert_values_equal(
            evaluate(atr("high", "low", "close")),
            evaluate(atr("high", "low", "close", 14)),
        )

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(atr("high", "low", "close", 5))
        from_exprs = evaluate(atr(pl.col("high"), pl.col("low"), pl.col("close"), 5))
        from_series = atr(
            pl.Series("high", HIGH), pl.Series("low", LOW), pl.Series("close", CLOSE), 5
        )
        self.assert_values_equal(from_exprs, from_names)
        self.assert_values_equal(from_series.to_list(), from_names)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(atr("high", "low", "close", 5).alias("atr"))
            .collect()
        )
        self.assert_values_equal(
            collected["atr"].to_list(), reference_atr(HIGH, LOW, CLOSE, 5)
        )

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                atr("high", "low", "close", window)
