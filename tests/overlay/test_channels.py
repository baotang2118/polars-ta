import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, frame_from, ramp_up, with_null

from polars_ta import atr, donchian, ema, keltner


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
        result = evaluate(donchian("high", "low", 3), frame_from(constant(6), 0.0))
        self.assert_values_equal(result["upper"].to_list()[2:], [5.0] * 4)
        self.assert_values_equal(result["lower"].to_list()[2:], [5.0] * 4)

    def test_null_blanks_every_overlapping_window(self) -> None:
        high = with_null(HIGH, 6)
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
        values = ramp_up(25, 0.0)
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
        self.assertEqual(collected.columns[-1], "d")

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                donchian("high", "low", window)


class TestKeltner(IndicatorAssertions):
    def test_matches_an_ema_with_an_atr_envelope(self) -> None:
        bars = frame()
        result = evaluate(keltner("high", "low", "close", 10, atr_window=5))
        middle = bars.select(ema("close", 10)).to_series().to_list()
        ranges = bars.select(atr("high", "low", "close", 5)).to_series().to_list()
        self.assert_values_equal(result["middle"].to_list(), middle)
        for index, (centre, width) in enumerate(zip(middle, ranges)):
            with self.subTest(index=index):
                if centre is None or width is None:
                    self.assertIsNone(result["upper"][index])
                    self.assertIsNone(result["lower"][index])
                else:
                    self.assertAlmostEqual(
                        result["upper"][index], centre + 2.0 * width, places=10
                    )
                    self.assertAlmostEqual(
                        result["lower"][index], centre - 2.0 * width, places=10
                    )

    def test_field_names_and_order(self) -> None:
        self.assertEqual(
            evaluate(keltner("high", "low", "close", 5)).columns,
            ["lower", "middle", "upper"],
        )

    def test_centre_line_warms_up_before_the_edges(self) -> None:
        for window, atr_window in ((10, 5), (6, 14), (20, 10)):
            with self.subTest(window=window, atr_window=atr_window):
                result = evaluate(
                    keltner("high", "low", "close", window, atr_window=atr_window)
                )
                centre = window - 1
                edge = max(window - 1, atr_window)
                self.assertEqual(result["middle"].to_list()[:centre], [None] * centre)
                self.assertIsNotNone(result["middle"][centre])
                self.assertEqual(result["upper"].to_list()[:edge], [None] * edge)
                self.assertIsNotNone(result["upper"][edge])

    def test_middle_is_halfway_between_the_edges(self) -> None:
        rows = (
            evaluate(keltner("high", "low", "close", 10, atr_window=5))
            .drop_nulls()
            .iter_rows(named=True)
        )
        for row in rows:
            self.assertAlmostEqual(
                row["middle"], (row["upper"] + row["lower"]) / 2.0, places=10
            )

    def test_edges_are_ordered(self) -> None:
        result = evaluate(keltner("high", "low", "close", 10, atr_window=5))
        for row in result.drop_nulls().iter_rows(named=True):
            self.assertLessEqual(row["lower"], row["middle"])
            self.assertLessEqual(row["middle"], row["upper"])

    def test_a_larger_multiplier_widens_the_channel(self) -> None:
        narrow = evaluate(keltner("high", "low", "close", 10, multiplier=1.0))
        wide = evaluate(keltner("high", "low", "close", 10, multiplier=3.0))
        for index in range(10, len(CLOSE)):
            with self.subTest(index=index):
                self.assertGreater(wide["upper"][index], narrow["upper"][index])
                self.assertLess(wide["lower"][index], narrow["lower"][index])
                self.assertAlmostEqual(
                    wide["middle"][index], narrow["middle"][index], places=10
                )

    def test_flat_market_collapses_the_channel(self) -> None:
        data = frame_from(constant(30), 0.0)
        result = evaluate(keltner("high", "low", "close", 5, atr_window=3), data)
        self.assert_values_equal(result["upper"].to_list()[5:], [5.0] * 25)
        self.assert_values_equal(result["lower"].to_list()[5:], [5.0] * 25)

    def test_null_propagates_to_every_field_it_feeds(self) -> None:
        result = evaluate(
            keltner("high", "low", "close", 5, atr_window=3),
            frame(close=with_null(CLOSE, 8)),
        )
        self.assertIsNone(result["middle"][8])
        self.assertIsNone(result["upper"][8])

    def test_a_null_high_spares_the_centre_line(self) -> None:
        result = evaluate(
            keltner("high", "low", "close", 5, atr_window=3),
            frame(high=with_null(HIGH, 8)),
        )
        self.assertIsNotNone(result["middle"][8])
        self.assertIsNone(result["upper"][8])

    def test_defaults_are_twenty_and_ten(self) -> None:
        self.assert_values_equal(
            evaluate(keltner("high", "low", "close"))["upper"].to_list(),
            evaluate(
                keltner("high", "low", "close", 20, atr_window=10, multiplier=2.0)
            )["upper"].to_list(),
        )

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(keltner("high", "low", "close", 5))["upper"].to_list()
        from_exprs = evaluate(
            keltner(pl.col("high"), pl.col("low"), pl.col("close"), 5)
        )["upper"].to_list()
        from_series = keltner(
            pl.Series("high", HIGH), pl.Series("low", LOW), pl.Series("close", CLOSE), 5
        )
        self.assert_values_equal(from_exprs, from_names)
        self.assert_values_equal(
            from_series.struct.field("upper").to_list(), from_names
        )

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(keltner("high", "low", "close", 5).alias("k"))
            .collect()
        )
        self.assertEqual(collected.columns[-1], "k")

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            keltner(pl.Series("high", HIGH), "low", "close")

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            keltner("high", "low", "close", 0)
        with self.assertRaises(ValueError):
            keltner("high", "low", "close", 20, atr_window=0)
        with self.assertRaises(ValueError):
            keltner("high", "low", "close", 20, multiplier=0.0)
        with self.assertRaises(ValueError):
            keltner("high", "low", "close", 20, mode="mesa")
