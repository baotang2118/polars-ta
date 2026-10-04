import polars as pl
from _assertions import IndicatorAssertions

from polars_ta import ema, macd

VALUES: list[float] = [
    float(value)
    for value in [
        1,
        3,
        2,
        6,
        5,
        9,
        4,
        8,
        7,
        11,
        6,
        10,
        9,
        13,
        8,
        12,
        11,
        15,
        10,
        14,
        13,
        17,
        12,
        16,
        15,
        19,
        14,
        18,
        17,
        21,
        16,
        20,
        19,
        23,
        18,
        22,
        21,
        25,
        20,
        24,
    ]
]


def evaluate(expr: pl.Expr, values: list[float] | None = None) -> pl.DataFrame:
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr.alias("m")).unnest("m")


def reference_macd(values, fast: int, slow: int, signal: int):
    frame = pl.DataFrame({"close": values})
    fast_line = frame.select(ema("close", fast)).to_series().to_list()
    slow_line = frame.select(ema("close", slow)).to_series().to_list()
    macd_line = [
        None if f is None or s is None else f - s for f, s in zip(fast_line, slow_line)
    ]
    signal_line = (
        pl.DataFrame({"macd": macd_line})
        .select(ema("macd", signal))
        .to_series()
        .to_list()
    )
    aligned = [None if s is None else macd_line[i] for i, s in enumerate(signal_line)]
    histogram = [
        None if a is None or s is None else a - s for a, s in zip(aligned, signal_line)
    ]
    return aligned, signal_line, histogram


class TestMacd(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        macd_line, signal, histogram = reference_macd(VALUES, 3, 6, 4)
        self.assert_values_equal(result["macd"].to_list(), macd_line)
        self.assert_values_equal(result["signal"].to_list(), signal)
        self.assert_values_equal(result["histogram"].to_list(), histogram)

    def test_field_names_and_order(self) -> None:
        self.assertEqual(
            evaluate(macd("close", 3, 6, 4)).columns, ["macd", "signal", "histogram"]
        )

    def test_warm_up_matches_the_talib_lookback(self) -> None:
        for fast, slow, signal in ((3, 6, 4), (12, 26, 9), (2, 5, 3)):
            with self.subTest(periods=(fast, slow, signal)):
                result = evaluate(macd("close", fast, slow, signal))
                lookback = (slow - 1) + (signal - 1)
                self.assertEqual(result["macd"].to_list()[:lookback], [None] * lookback)
                self.assertIsNotNone(result["macd"][lookback])

    def test_all_three_fields_start_on_the_same_row(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        self.assertEqual(result["macd"].null_count(), result["signal"].null_count())
        self.assertEqual(result["macd"].null_count(), result["histogram"].null_count())

    def test_histogram_is_macd_minus_signal(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        for row in result.drop_nulls().iter_rows(named=True):
            self.assertAlmostEqual(
                row["histogram"], row["macd"] - row["signal"], places=10
            )

    def test_macd_line_is_the_gap_between_the_two_emas(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        gap = (
            pl.DataFrame({"close": VALUES})
            .select((ema("close", 3) - ema("close", 6)).alias("gap"))
            .to_series()
            .to_list()
        )
        for index, value in enumerate(result["macd"].to_list()):
            if value is not None:
                self.assertAlmostEqual(value, gap[index], places=10)

    def test_rising_input_gives_a_positive_macd_line(self) -> None:
        rising = [float(index) for index in range(40)]
        for value in evaluate(macd("close", 3, 6, 4), rising)["macd"].to_list():
            if value is not None:
                self.assertGreater(value, 0.0)

    def test_constant_input_gives_a_zero_macd_line(self) -> None:
        result = evaluate(macd("close", 3, 6, 4), [7.0] * 30)
        for value in result["macd"].to_list():
            if value is not None:
                self.assertAlmostEqual(value, 0.0, places=10)

    def test_every_mode_is_supported(self) -> None:
        for mode in ("talib", "adjust", "recursive"):
            with self.subTest(mode=mode):
                result = evaluate(macd("close", 3, 6, 4, mode=mode))
                self.assertIsNotNone(result["macd"][-1])

    def test_default_periods_are_twelve_twenty_six_and_nine(self) -> None:
        result = evaluate(macd("close"))
        self.assertEqual(result["macd"].null_count(), 25 + 8)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        result = evaluate(macd("close", 3, 6, 4), VALUES[:5])
        self.assertEqual(result["macd"].to_list(), [None] * 5)

    def test_series_input_returns_a_struct_series(self) -> None:
        result = macd(pl.Series("close", VALUES), 3, 6, 4)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")
        self.assertEqual(result.struct.fields, ["macd", "signal", "histogram"])

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(macd("close", 3, 6, 4).alias("m"))
            .collect()
        )
        self.assertEqual(collected.columns, ["close", "m"])

    def test_invalid_arguments_raise(self) -> None:
        for periods in ((0, 26, 9), (12, -1, 9), (12, 26, 0), (12, 26, 2.5)):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                macd("close", *periods)
        with self.assertRaises(ValueError):
            macd("close", 12, 26, 9, mode="convergence")
