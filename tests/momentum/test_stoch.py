import polars as pl
from _assertions import IndicatorAssertions

from polars_ta import stoch

HIGH: list[float] = [10.0, 11, 12, 11, 10, 11, 12, 13, 12, 11, 13, 14, 12, 11, 10, 12]
LOW: list[float] = [8.0, 9, 10, 9, 8, 9, 10, 11, 10, 9, 11, 12, 10, 9, 8, 10]
CLOSE: list[float] = [9.0, 10, 11, 10, 9, 10, 11, 12, 11, 10, 12, 13, 11, 10, 9, 11]


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
    return source.select(expr.alias("s")).unnest("s")


def reference_stoch(high, low, close, fastk: int, slowk: int, slowd: int):
    fast_k: list[float | None] = []
    for index in range(len(close)):
        window_high = high[index + 1 - fastk : index + 1]
        window_low = low[index + 1 - fastk : index + 1]
        if (
            index + 1 < fastk
            or close[index] is None
            or any(v is None for v in window_high)
            or any(v is None for v in window_low)
        ):
            fast_k.append(None)
            continue
        highest, lowest = max(window_high), min(window_low)
        span = highest - lowest
        fast_k.append(0.0 if span <= 0.0 else 100.0 * (close[index] - lowest) / span)

    def smooth(values, period):
        out: list[float | None] = []
        for index in range(len(values)):
            chunk = values[index + 1 - period : index + 1]
            if index + 1 < period or any(v is None for v in chunk):
                out.append(None)
            else:
                out.append(sum(chunk) / period)
        return out

    k = smooth(fast_k, slowk)
    d = smooth(k, slowd)
    aligned_k = [None if d[i] is None else k[i] for i in range(len(k))]
    return aligned_k, d


class TestStoch(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        result = evaluate(stoch("high", "low", "close", 5, 3, 3))
        k, d = reference_stoch(HIGH, LOW, CLOSE, 5, 3, 3)
        self.assert_values_equal(result["k"].to_list(), k)
        self.assert_values_equal(result["d"].to_list(), d)

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
        rising = [float(index) for index in range(1, 11)]
        result = evaluate(
            stoch("high", "low", "close", 3, 1, 1), frame(rising, rising, rising)
        )
        self.assert_values_equal(result["k"].to_list()[2:], [100.0] * 8)

    def test_close_at_the_window_low_is_zero(self) -> None:
        falling = [float(10 - index) for index in range(10)]
        result = evaluate(
            stoch("high", "low", "close", 3, 1, 1), frame(falling, falling, falling)
        )
        self.assert_values_equal(result["k"].to_list()[2:], [0.0] * 8)

    def test_flat_range_reports_zero(self) -> None:
        data = frame([5.0] * 8, [5.0] * 8, [5.0] * 8)
        result = evaluate(stoch("high", "low", "close", 3, 2, 2), data)
        self.assert_values_equal(result["k"].to_list()[4:], [0.0] * 4)

    def test_null_input_propagates(self) -> None:
        close = list(CLOSE)
        close[6] = None
        result = evaluate(stoch("high", "low", "close", 3, 2, 2), frame(close=close))
        k, _ = reference_stoch(HIGH, LOW, close, 3, 2, 2)
        self.assert_values_equal(result["k"].to_list(), k)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        result = evaluate(stoch("high", "low", "close", len(HIGH), 3, 3))
        self.assertEqual(result["k"].to_list(), [None] * len(HIGH))

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(stoch("high", "low", "close", 5, 3, 3))["k"].to_list()
        from_exprs = evaluate(
            stoch(pl.col("high"), pl.col("low"), pl.col("close"), 5, 3, 3)
        )["k"].to_list()
        from_series = stoch(
            pl.Series("high", HIGH),
            pl.Series("low", LOW),
            pl.Series("close", CLOSE),
            5,
            3,
            3,
        )
        self.assert_values_equal(from_exprs, from_names)
        self.assert_values_equal(from_series.struct.field("k").to_list(), from_names)
        self.assertEqual(from_series.name, "high")

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(stoch("high", "low", "close", 5, 3, 3).alias("s"))
            .collect()
        )
        self.assertEqual(collected.columns, ["high", "low", "close", "s"])

    def test_invalid_periods_raise(self) -> None:
        for periods in ((0, 3, 3), (5, 0, 3), (5, 3, -1), (5, 2.5, 3)):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                stoch("high", "low", "close", *periods)
