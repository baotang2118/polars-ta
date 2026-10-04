import polars as pl
from _assertions import IndicatorAssertions

from polars_ta import donchian

HIGH: list[float] = [10.0, 11, 12, 11, 10, 11, 12, 13, 12, 11, 13, 14, 12, 11, 10, 12]
LOW: list[float] = [8.0, 9, 10, 9, 8, 9, 10, 11, 10, 9, 11, 12, 10, 9, 8, 10]


def frame(high=None, low=None) -> pl.DataFrame:
    return pl.DataFrame(
        {"high": HIGH if high is None else high, "low": LOW if low is None else low}
    )


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None) -> pl.DataFrame:
    source = data if data is not None else frame()
    return source.select(expr.alias("d")).unnest("d")


def reference_donchian(high, low, window: int):
    lower: list[float | None] = []
    middle: list[float | None] = []
    upper: list[float | None] = []
    for index in range(len(high)):
        window_high = high[index + 1 - window : index + 1]
        window_low = low[index + 1 - window : index + 1]
        short = index + 1 < window
        # Each edge depends only on its own input, so they can blank separately.
        top = None if short or any(v is None for v in window_high) else max(window_high)
        bottom = (
            None if short or any(v is None for v in window_low) else min(window_low)
        )
        upper.append(top)
        lower.append(bottom)
        middle.append(None if top is None or bottom is None else (top + bottom) / 2.0)
    return lower, middle, upper


class TestDonchian(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        result = evaluate(donchian("high", "low", 5))
        lower, middle, upper = reference_donchian(HIGH, LOW, 5)
        self.assert_values_equal(result["lower"].to_list(), lower)
        self.assert_values_equal(result["middle"].to_list(), middle)
        self.assert_values_equal(result["upper"].to_list(), upper)

    def test_field_names_and_order(self) -> None:
        self.assertEqual(
            evaluate(donchian("high", "low", 5)).columns, ["lower", "middle", "upper"]
        )

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        for window in (2, 5, 10):
            with self.subTest(window=window):
                result = evaluate(donchian("high", "low", window))
                self.assertEqual(
                    result["upper"].to_list()[: window - 1], [None] * (window - 1)
                )
                self.assertIsNotNone(result["upper"][window - 1])

    def test_middle_is_halfway_between_the_edges(self) -> None:
        for row in (
            evaluate(donchian("high", "low", 5)).drop_nulls().iter_rows(named=True)
        ):
            self.assertAlmostEqual(
                row["middle"], (row["upper"] + row["lower"]) / 2.0, places=10
            )

    def test_channel_contains_the_prices(self) -> None:
        result = evaluate(donchian("high", "low", 5))
        for index in range(4, len(HIGH)):
            self.assertGreaterEqual(result["upper"][index], HIGH[index])
            self.assertLessEqual(result["lower"][index], LOW[index])

    def test_window_of_one_tracks_each_bar(self) -> None:
        result = evaluate(donchian("high", "low", 1))
        self.assert_values_equal(result["upper"].to_list(), HIGH)
        self.assert_values_equal(result["lower"].to_list(), LOW)

    def test_constant_input_collapses_the_channel(self) -> None:
        result = evaluate(donchian("high", "low", 3), frame([5.0] * 6, [5.0] * 6))
        self.assert_values_equal(result["upper"].to_list()[2:], [5.0] * 4)
        self.assert_values_equal(result["lower"].to_list()[2:], [5.0] * 4)

    def test_null_blanks_every_overlapping_window(self) -> None:
        high = list(HIGH)
        high[6] = None
        result = evaluate(donchian("high", "low", 3), frame(high=high))
        lower, middle, upper = reference_donchian(high, LOW, 3)
        self.assert_values_equal(result["upper"].to_list(), upper)
        self.assert_values_equal(result["lower"].to_list(), lower)
        self.assert_values_equal(result["middle"].to_list(), middle)
        self.assertEqual(result["upper"].to_list()[6:9], [None, None, None])

    def test_window_longer_than_input_is_all_null(self) -> None:
        result = evaluate(donchian("high", "low", len(HIGH) + 1))
        self.assertEqual(result["upper"].to_list(), [None] * len(HIGH))

    def test_default_window_is_twenty(self) -> None:
        values = [float(index) for index in range(25)]
        result = evaluate(donchian("high", "low"), frame(values, values))
        self.assertEqual(result["upper"].to_list()[:19], [None] * 19)

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(donchian("high", "low", 5))["upper"].to_list()
        from_exprs = evaluate(donchian(pl.col("high"), pl.col("low"), 5))[
            "upper"
        ].to_list()
        from_series = donchian(pl.Series("high", HIGH), pl.Series("low", LOW), 5)
        self.assert_values_equal(from_exprs, from_names)
        self.assert_values_equal(
            from_series.struct.field("upper").to_list(), from_names
        )

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(donchian("high", "low", 5).alias("d"))
            .collect()
        )
        self.assertEqual(collected.columns, ["high", "low", "d"])

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                donchian("high", "low", window)
