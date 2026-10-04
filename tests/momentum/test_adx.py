import polars as pl
from _assertions import IndicatorAssertions

from polars_ta import adx

HIGH: list[float] = [
    10.0,
    11,
    12,
    11,
    10,
    11,
    12,
    13,
    12,
    11,
    13,
    14,
    12,
    11,
    10,
    12,
    14,
    15,
]
LOW: list[float] = [8.0, 9, 10, 9, 8, 9, 10, 11, 10, 9, 11, 12, 10, 9, 8, 10, 12, 13]
CLOSE: list[float] = [
    9.0,
    10,
    11,
    10,
    9,
    10,
    11,
    12,
    11,
    10,
    12,
    13,
    11,
    10,
    9,
    11,
    13,
    14,
]


def frame(high=None, low=None, close=None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "high": HIGH if high is None else high,
            "low": LOW if low is None else low,
            "close": CLOSE if close is None else close,
        }
    )


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None) -> pl.DataFrame:
    source = data if data is not None else frame()
    return source.select(expr.alias("a")).unnest("a")


class TestAdx(IndicatorAssertions):
    def test_field_names_and_order(self) -> None:
        self.assertEqual(
            evaluate(adx("high", "low", "close", 5)).columns,
            ["adx", "plus_di", "minus_di"],
        )

    def test_directional_indicators_warm_up_in_window_rows(self) -> None:
        for window in (3, 5, 7):
            with self.subTest(window=window):
                result = evaluate(adx("high", "low", "close", window))
                for field in ("plus_di", "minus_di"):
                    self.assertEqual(result[field].to_list()[:window], [None] * window)
                    self.assertIsNotNone(result[field][window])

    def test_adx_warms_up_in_twice_window_minus_one_rows(self) -> None:
        for window in (3, 5, 7):
            with self.subTest(window=window):
                result = evaluate(adx("high", "low", "close", window))
                lookback = 2 * window - 1
                self.assertEqual(result["adx"].to_list()[:lookback], [None] * lookback)
                self.assertIsNotNone(result["adx"][lookback])

    def test_adx_lags_the_indicators_by_window_minus_one(self) -> None:
        result = evaluate(adx("high", "low", "close", 5))
        self.assertEqual(result["adx"].null_count() - result["plus_di"].null_count(), 4)

    def test_all_fields_stay_within_zero_and_one_hundred(self) -> None:
        result = evaluate(adx("high", "low", "close", 5))
        for field in ("adx", "plus_di", "minus_di"):
            for value in result[field].to_list():
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_steady_rise_favours_the_plus_indicator(self) -> None:
        rising = [float(index) for index in range(1, 20)]
        data = frame(rising, [v - 1 for v in rising], rising)
        result = evaluate(adx("high", "low", "close", 5), data)
        self.assertAlmostEqual(result["plus_di"][-1], 100.0, places=6)
        self.assertAlmostEqual(result["minus_di"][-1], 0.0, places=6)

    def test_steady_fall_favours_the_minus_indicator(self) -> None:
        falling = [float(20 - index) for index in range(19)]
        data = frame([v + 1 for v in falling], falling, falling)
        result = evaluate(adx("high", "low", "close", 5), data)
        self.assertAlmostEqual(result["minus_di"][-1], 100.0, places=6)
        self.assertAlmostEqual(result["plus_di"][-1], 0.0, places=6)

    def test_sustained_trend_drives_adx_toward_one_hundred(self) -> None:
        rising = [float(index) for index in range(1, 40)]
        data = frame(rising, [v - 1 for v in rising], rising)
        result = evaluate(adx("high", "low", "close", 5), data)
        self.assertGreater(result["adx"][-1], 90.0)

    def test_flat_market_reports_zero(self) -> None:
        data = frame([5.0] * 20, [5.0] * 20, [5.0] * 20)
        result = evaluate(adx("high", "low", "close", 3), data)
        self.assertAlmostEqual(result["plus_di"][-1], 0.0, places=10)
        self.assertAlmostEqual(result["adx"][-1], 0.0, places=10)

    def test_null_input_propagates(self) -> None:
        high = list(HIGH)
        high[4] = None
        result = evaluate(adx("high", "low", "close", 3), frame(high=high))
        self.assertGreater(result["plus_di"].null_count(), 3)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        result = evaluate(adx("high", "low", "close", len(HIGH)))
        self.assertEqual(result["adx"].to_list(), [None] * len(HIGH))

    def test_default_window_is_fourteen(self) -> None:
        result = evaluate(adx("high", "low", "close"))
        self.assertEqual(result["plus_di"].null_count(), 14)
        self.assertEqual(result["adx"].to_list(), [None] * len(HIGH))

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(adx("high", "low", "close", 5))["adx"].to_list()
        from_exprs = evaluate(adx(pl.col("high"), pl.col("low"), pl.col("close"), 5))[
            "adx"
        ].to_list()
        from_series = adx(
            pl.Series("high", HIGH), pl.Series("low", LOW), pl.Series("close", CLOSE), 5
        )
        self.assert_values_equal(from_exprs, from_names)
        self.assert_values_equal(from_series.struct.field("adx").to_list(), from_names)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(adx("high", "low", "close", 5).alias("a"))
            .collect()
        )
        self.assertEqual(collected.columns, ["high", "low", "close", "a"])

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                adx("high", "low", "close", window)
