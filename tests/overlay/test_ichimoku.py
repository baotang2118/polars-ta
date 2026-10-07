import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, frame_from, with_null

from polars_ta import ichimoku


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None) -> pl.DataFrame:
    source = data if data is not None else frame()
    return source.select(expr.alias("i")).unnest("i")


def reference_midpoint(high, low, window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(high)):
        window_high = high[index + 1 - window : index + 1]
        window_low = low[index + 1 - window : index + 1]
        if index + 1 < window or any(v is None for v in window_high + window_low):
            result.append(None)
        else:
            result.append((max(window_high) + min(window_low)) / 2.0)
    return result


class TestIchimoku(IndicatorAssertions):
    def test_field_names_and_order(self) -> None:
        self.assertEqual(
            evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6)).columns,
            ["conversion", "base", "span_a", "span_b", "lagging"],
        )

    def test_conversion_and_base_are_rolling_midpoints(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        self.assert_values_equal(
            result["conversion"].to_list(), reference_midpoint(HIGH, LOW, 3)
        )
        self.assert_values_equal(
            result["base"].to_list(), reference_midpoint(HIGH, LOW, 6)
        )

    def test_span_a_is_the_displaced_average_of_the_two_lines(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        conversion = reference_midpoint(HIGH, LOW, 3)
        base = reference_midpoint(HIGH, LOW, 6)
        expected = [
            None
            if conversion[i] is None or base[i] is None
            else (conversion[i] + base[i]) / 2.0
            for i in range(len(HIGH))
        ]
        shifted = [None] * 6 + expected[: len(expected) - 6]
        self.assert_values_equal(result["span_a"].to_list(), shifted)

    def test_span_b_is_the_displaced_long_midpoint(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        expected = reference_midpoint(HIGH, LOW, 12)
        shifted = [None] * 6 + expected[: len(expected) - 6]
        self.assert_values_equal(result["span_b"].to_list(), shifted)

    def test_lagging_span_is_the_close_pulled_backward(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        expected = CLOSE[6:] + [None] * 6
        self.assert_values_equal(result["lagging"].to_list(), expected)

    def test_lagging_span_tail_is_null(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        self.assertEqual(result["lagging"].to_list()[-6:], [None] * 6)

    def test_each_field_carries_its_own_warm_up(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        self.assertEqual(result["conversion"].to_list()[:2], [None, None])
        self.assertIsNotNone(result["conversion"][2])
        self.assertEqual(result["base"].to_list()[:5], [None] * 5)
        self.assertIsNotNone(result["base"][5])
        self.assertEqual(result["span_b"].to_list()[: 11 + 6], [None] * 17)

    def test_defaults_are_the_classic_nine_twentysix_fiftytwo(self) -> None:
        result = evaluate(ichimoku("high", "low", "close"))
        self.assertEqual(result["conversion"].to_list()[:8], [None] * 8)
        self.assertIsNotNone(result["conversion"][8])
        self.assertEqual(result["lagging"].to_list()[-26:], [None] * 26)

    def test_constant_input_gives_constant_lines(self) -> None:
        data = frame_from(constant(20), 0.0)
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6), data)
        for value in result["conversion"].to_list()[2:]:
            self.assertAlmostEqual(value, 5.0, places=10)

    def test_null_blanks_every_overlapping_window(self) -> None:
        high = with_null(HIGH, 10)
        result = evaluate(
            ichimoku("high", "low", "close", 3, 6, 12, 6), frame(high=high)
        )
        self.assert_values_equal(
            result["conversion"].to_list(), reference_midpoint(high, LOW, 3)
        )

    def test_names_and_expressions_agree(self) -> None:
        from_names = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))[
            "conversion"
        ].to_list()
        from_exprs = evaluate(
            ichimoku(pl.col("high"), pl.col("low"), pl.col("close"), 3, 6, 12, 6)
        )["conversion"].to_list()
        self.assert_values_equal(from_exprs, from_names)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ichimoku(
                pl.Series("high", HIGH),
                pl.Series("low", LOW),
                pl.Series("close", CLOSE),
                3,
                6,
                12,
                6,
            )

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(ichimoku("high", "low", "close", 3, 6, 12, 6).alias("i"))
            .collect()
        )
        self.assertEqual(collected.columns[-1], "i")

    def test_invalid_periods_raise(self) -> None:
        for periods in (
            (0, 26, 52, 26),
            (9, -1, 52, 26),
            (9, 26, 0, 26),
            (9, 26, 52, 0),
        ):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                ichimoku("high", "low", "close", *periods)
