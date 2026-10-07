import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HAND_CHECKED, HIGH, LOW, constant, frame, ramp_up

from polars_ta import atr, ema, macd, macdext, macdfix, natr, sma, trix, ultosc

LENGTH: int = 90
BARS: pl.DataFrame = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestNatr(IndicatorAssertions):
    def test_is_atr_as_a_percentage_of_close(self) -> None:
        average = column(atr("high", "low", "close", 14))
        expected = [
            None if value is None else 100.0 * value / close
            for value, close in zip(average, CLOSE[:LENGTH])
        ]
        self.assert_values_equal(column(natr("high", "low", "close", 14)), expected)

    def test_warm_up_matches_atr(self) -> None:
        for window in (3, 7, 14):
            with self.subTest(window=window):
                result = column(natr("high", "low", "close", window))
                self.assertEqual(result[:window], [None] * window)
                self.assertIsNotNone(result[window])

    def test_zero_close_reports_zero(self) -> None:
        bars = pl.DataFrame(
            {
                "high": [1.0] * 8,
                "low": [-1.0] * 8,
                "close": [0.0] * 8,
            }
        )
        self.assert_values_equal(
            column(natr("high", "low", "close", 3), bars)[3:], [0.0] * 5
        )

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            natr("high", "low", "close", 0)


class TestTrix(IndicatorAssertions):
    def test_is_the_one_period_roc_of_a_triple_ema(self) -> None:
        values = CLOSE[:LENGTH]
        bars = pl.DataFrame({"close": values})
        stage = bars.select(ema("close", 5)).to_series().to_list()
        for _ in range(2):
            stage = (
                pl.DataFrame({"close": stage})
                .select(ema("close", 5))
                .to_series()
                .to_list()
            )
        expected = [
            None
            if index == 0 or stage[index] is None or stage[index - 1] is None
            else (stage[index] / stage[index - 1] - 1.0) * 100.0
            for index in range(LENGTH)
        ]
        self.assert_values_equal(column(trix("close", 5), bars), expected)

    def test_warm_up_is_three_passes_plus_one(self) -> None:
        bars = pl.DataFrame({"close": CLOSE[:LENGTH]})
        for window in (3, 5, 8):
            with self.subTest(window=window):
                result = column(trix("close", window), bars)
                lookback = 3 * (window - 1) + 1
                self.assertEqual(result[:lookback], [None] * lookback)
                self.assertIsNotNone(result[lookback])

    def test_flat_series_has_no_slope(self) -> None:
        bars = pl.DataFrame({"close": constant(40, 6.0)})
        self.assert_values_equal(column(trix("close", 3), bars)[7:], [0.0] * 33)

    def test_rising_series_is_positive(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(40)})
        for value in column(trix("close", 3), bars)[7:]:
            self.assertGreater(value, 0.0)

    def test_default_window_is_thirty(self) -> None:
        bars = pl.DataFrame({"close": CLOSE[:LENGTH]})
        self.assert_values_equal(
            column(trix("close"), bars), column(trix("close", 30), bars)
        )

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            trix("close", 0)


class TestUltosc(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        high, low, close = HIGH[:LENGTH], LOW[:LENGTH], CLOSE[:LENGTH]
        periods = (7, 14, 28)
        buying: list[float | None] = [None] * LENGTH
        ranges: list[float | None] = [None] * LENGTH
        for index in range(1, LENGTH):
            floor = min(low[index], close[index - 1])
            ceiling = max(high[index], close[index - 1])
            buying[index] = close[index] - floor
            ranges[index] = ceiling - floor
        expected: list[float | None] = [None] * LENGTH
        for index in range(max(periods), LENGTH):
            total = 0.0
            for weight, window in zip((4.0, 2.0, 1.0), periods):
                start = index - window + 1
                pressure = sum(buying[start : index + 1])
                span = sum(ranges[start : index + 1])
                if span > 0.0:
                    total += weight * pressure / span
            expected[index] = 100.0 * total / 7.0
        self.assert_values_equal(column(ultosc("high", "low", "close")), expected)

    def test_warm_up_is_the_longest_period(self) -> None:
        for periods in ((7, 14, 28), (3, 5, 9), (9, 5, 3)):
            with self.subTest(periods=periods):
                result = column(ultosc("high", "low", "close", *periods))
                lookback = max(periods)
                self.assertEqual(result[:lookback], [None] * lookback)
                self.assertIsNotNone(result[lookback])

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        for value in column(ultosc("high", "low", "close")):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)

    def test_flat_market_reports_zero(self) -> None:
        flat = constant(20, 5.0)
        bars = frame(high=flat, low=flat, close=flat)
        self.assert_values_equal(
            column(ultosc("high", "low", "close", 2, 3, 5), bars)[5:], [0.0] * 15
        )

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            ultosc("high", "low", "close", 0, 14, 28)


class TestMacdVariants(IndicatorAssertions):
    def test_macdfix_is_macd_at_twelve_and_twenty_six(self) -> None:
        bars = pl.DataFrame({"close": CLOSE[:LENGTH]})
        fixed = bars.select(macdfix("close", 9).alias("m")).unnest("m")
        plain = bars.select(macd("close", 12, 26, 9).alias("m")).unnest("m")
        for name in ("macd", "signal", "histogram"):
            with self.subTest(name=name):
                self.assert_values_equal(fixed[name].to_list(), plain[name].to_list())

    def test_macdext_defaults_to_simple_averages(self) -> None:
        bars = pl.DataFrame({"close": CLOSE[:LENGTH]})
        fields = bars.select(macdext("close", 5, 12, 4).alias("m")).unnest("m")
        fast = bars.select(sma("close", 5)).to_series().to_list()
        slow = bars.select(sma("close", 12)).to_series().to_list()
        line = [None if f is None or s is None else f - s for f, s in zip(fast, slow)]
        signal = (
            pl.DataFrame({"line": line}).select(sma("line", 4)).to_series().to_list()
        )
        expected = [None if s is None else line[i] for i, s in enumerate(signal)]
        self.assert_values_equal(fields["macd"].to_list(), expected)
        self.assert_values_equal(fields["signal"].to_list(), signal)

    def test_macdext_honours_each_ma_type(self) -> None:
        bars = pl.DataFrame({"close": CLOSE[:LENGTH]})
        fields = bars.select(
            macdext(
                "close",
                5,
                12,
                4,
                fast_ma_type="ema",
                slow_ma_type="ema",
                signal_ma_type="ema",
            ).alias("m")
        ).unnest("m")
        plain = bars.select(macd("close", 5, 12, 4).alias("m")).unnest("m")
        self.assert_values_equal(fields["signal"].to_list(), plain["signal"].to_list())

    def test_macdext_fields_start_together(self) -> None:
        bars = pl.DataFrame({"close": CLOSE[:LENGTH]})
        fields = bars.select(macdext("close", 5, 12, 4).alias("m")).unnest("m")
        lookback = (12 - 1) + (4 - 1)
        for name in ("macd", "signal", "histogram"):
            with self.subTest(name=name):
                self.assertEqual(fields[name].to_list()[:lookback], [None] * lookback)
                self.assertIsNotNone(fields[name][lookback])

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            macdext("close", 0)
        with self.assertRaises(ValueError):
            macdext("close", 5, 12, 4, fast_ma_type="mesa")
        with self.assertRaises(ValueError):
            macdfix(pl.Series("close", HAND_CHECKED), 0)
