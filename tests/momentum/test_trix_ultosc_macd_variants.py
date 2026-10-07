import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HAND_CHECKED, HIGH, LOW, constant, frame, ramp_up

from polars_ta import macd, macdext, macdfix, natr, trix, ultosc

LENGTH: int = 90
BARS: pl.DataFrame = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
NATR_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    31.746031746031747, 26.55380333951763, 25.199810096778595, 22.95029561554003,
    19.947232064990676, 19.82452894771489, 20.394825525049015, 17.930445963779242,
    20.124070060982966, 17.84491751426511, 15.796635486676347, 16.1219986400884,
]
TRIX_5: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    1.2208338068720792, 0.011412687066858496, -0.288504617521379, -0.12243202378512397,
    0.4877800212819583, 1.7825470238750585, 2.9353134649920065, 3.3306883727149073,
    3.930231069621426, 3.707671947508051, 3.8507511773261394, 4.523079995944235,
]
ULTOSC: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    51.241416971345984, 50.65143600609738, 49.90560751303091, 48.15966989955722,
    47.906309442897616, 47.853558805239096, 48.13567960045787, 48.47393653415503,
    49.901807923036436, 51.14914811947017, 50.68076696821248, 50.623954615347245,
]
MACDEXT_MACD: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    0.3333333333333339, 0.05000000000000071, -0.4781666666666684, -0.4063333333333343,
    0.19966666666666733, 1.125166666666665, 1.4810000000000016, 1.9371666666666663,
    2.165166666666668, 2.315999999999999, 2.509999999999998, 2.6963333333333352,
]
MACDEXT_SIGNAL: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    0.5499999999999998, 0.3291666666666666, 0.0679583333333329, -0.12529166666666702,
    -0.15870833333333367, 0.11008333333333242, 0.5998749999999999, 1.18575,
    1.6771250000000002, 1.9748333333333337, 2.2320833333333328, 2.421875,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestNatr(IndicatorAssertions):
    def test_is_atr_as_a_percentage_of_close(self) -> None:
        result = column(natr("high", "low", "close", 14))
        self.assert_values_equal(result[: len(NATR_14)], NATR_14)

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
        bars = pl.DataFrame({"close": CLOSE[:LENGTH]})
        result = column(trix("close", 5), bars)
        self.assert_values_equal(result[: len(TRIX_5)], TRIX_5)

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
    def test_known_values(self) -> None:
        result = column(ultosc("high", "low", "close"))
        self.assert_values_equal(result[: len(ULTOSC)], ULTOSC)

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
        size = len(MACDEXT_MACD)
        self.assert_values_equal(fields["macd"].to_list()[:size], MACDEXT_MACD)
        self.assert_values_equal(fields["signal"].to_list()[:size], MACDEXT_SIGNAL)

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
            macdfix("close", 0)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            macdfix(pl.Series("close", HAND_CHECKED))
