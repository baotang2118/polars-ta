import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, ramp_down, ramp_up

from polars_ta import cmo, rsi, stochf, stochrsi, willr

VALUES: list[float] = CLOSE[:60]


def evaluate(expr: pl.Expr, values: list[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def unnest(expr: pl.Expr, bars: pl.DataFrame) -> dict[str, list]:
    result = bars.select(expr.alias("out")).unnest("out")
    return {name: result[name].to_list() for name in result.columns}


def reference_fast_k(
    high: list[float], low: list[float], close: list[float], window: int
) -> list[float | None]:
    result: list[float | None] = [None] * len(close)
    for index in range(window - 1, len(close)):
        start = index - window + 1
        highest = max(high[start : index + 1])
        lowest = min(low[start : index + 1])
        span = highest - lowest
        result[index] = 0.0 if span == 0.0 else 100.0 * (close[index] - lowest) / span
    return result


class TestCmo(IndicatorAssertions):
    def test_is_rsi_rescaled_around_zero(self) -> None:
        strength = evaluate(rsi("close", 14))
        expected = [None if v is None else 2.0 * v - 100.0 for v in strength]
        self.assert_values_equal(evaluate(cmo("close", 14)), expected)

    def test_warm_up_is_window_nulls(self) -> None:
        for window in (2, 5, 14):
            with self.subTest(window=window):
                result = evaluate(cmo("close", window))
                self.assertEqual(result[:window], [None] * window)
                self.assertIsNotNone(result[window])

    def test_monotonic_rise_reaches_one_hundred(self) -> None:
        self.assert_values_equal(
            evaluate(cmo("close", 3), ramp_up(10, 0.0))[3:], [100.0] * 7
        )

    def test_monotonic_fall_reaches_minus_one_hundred(self) -> None:
        self.assert_values_equal(
            evaluate(cmo("close", 3), ramp_down(10))[3:], [-100.0] * 7
        )

    def test_flat_series_reports_zero_not_fifty(self) -> None:
        self.assert_values_equal(evaluate(cmo("close", 3), constant(8))[3:], [0.0] * 5)

    def test_default_window_is_fourteen(self) -> None:
        self.assert_values_equal(evaluate(cmo("close")), evaluate(cmo("close", 14)))

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            cmo("close", 0)


class TestStochf(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        fields = unnest(stochf("high", "low", "close", 5, 3), bars)
        raw = reference_fast_k(HIGH[:60], LOW[:60], VALUES, 5)
        expected_d: list[float | None] = [None] * 60
        for index in range(6, 60):
            expected_d[index] = sum(raw[index - 2 : index + 1]) / 3.0
        self.assert_values_equal(fields["fast_d"], expected_d)
        self.assert_values_equal(fields["fast_k"], [None] * 6 + raw[6:])

    def test_warm_up_sums_both_periods(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        for fastk, fastd in ((5, 3), (3, 4), (8, 2)):
            with self.subTest(fastk=fastk, fastd=fastd):
                fields = unnest(stochf("high", "low", "close", fastk, fastd), bars)
                lookback = (fastk - 1) + (fastd - 1)
                for name in ("fast_k", "fast_d"):
                    self.assertEqual(fields[name][:lookback], [None] * lookback)
                    self.assertIsNotNone(fields[name][lookback])

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        fields = unnest(stochf("high", "low", "close", 5, 3), bars)
        for name in ("fast_k", "fast_d"):
            for value in fields[name]:
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_flat_range_reports_zero(self) -> None:
        flat = constant(20, 4.0)
        bars = frame(high=flat, low=flat, close=flat)
        fields = unnest(stochf("high", "low", "close", 5, 3), bars)
        self.assert_values_equal(fields["fast_k"][6:], [0.0] * 14)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            stochf("high", "low", "close", 0, 3)


class TestStochrsi(IndicatorAssertions):
    def test_is_the_fast_stochastic_of_the_rsi(self) -> None:
        strength = evaluate(rsi("close", 14))
        known = [value for value in strength if value is not None]
        inner = pl.DataFrame({"close": known}).select(
            stochf("close", "close", "close", 5, 3).alias("out")
        )
        expected = inner.unnest("out")["fast_k"].to_list()
        fields = unnest(stochrsi("close", 14, 5, 3), pl.DataFrame({"close": VALUES}))
        self.assert_values_equal(fields["fast_k"][14:], expected)

    def test_warm_up_sums_every_period(self) -> None:
        bars = pl.DataFrame({"close": VALUES})
        for window, fastk, fastd in ((14, 5, 3), (5, 4, 2)):
            with self.subTest(window=window, fastk=fastk, fastd=fastd):
                fields = unnest(stochrsi("close", window, fastk, fastd), bars)
                lookback = window + (fastk - 1) + (fastd - 1)
                for name in ("fast_k", "fast_d"):
                    self.assertEqual(fields[name][:lookback], [None] * lookback)
                    self.assertIsNotNone(fields[name][lookback])

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        fields = unnest(stochrsi("close"), pl.DataFrame({"close": VALUES}))
        for name in ("fast_k", "fast_d"):
            for value in fields[name]:
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertAlmostEqual(min(value, 100.0), value, places=10)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            stochrsi("close", 0)


class TestWillr(IndicatorAssertions):
    def test_mirrors_raw_fast_k(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        result = bars.select(willr("high", "low", "close", 14)).to_series().to_list()
        raw = reference_fast_k(HIGH[:60], LOW[:60], VALUES, 14)
        expected = [None if v is None else v - 100.0 for v in raw]
        self.assert_values_equal(result, expected)

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        for window in (3, 7, 14):
            with self.subTest(window=window):
                result = (
                    bars.select(willr("high", "low", "close", window))
                    .to_series()
                    .to_list()
                )
                self.assertEqual(result[: window - 1], [None] * (window - 1))
                self.assertIsNotNone(result[window - 1])

    def test_output_stays_within_minus_one_hundred_and_zero(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        for value in bars.select(willr("high", "low", "close")).to_series():
            if value is not None:
                self.assertGreaterEqual(value, -100.0)
                self.assertLessEqual(value, 0.0)

    def test_close_at_the_high_reports_zero(self) -> None:
        rising = ramp_up(20)
        bars = frame(high=rising, low=rising, close=rising)
        result = bars.select(willr("high", "low", "close", 5)).to_series().to_list()
        self.assert_values_equal(result[4:], [0.0] * 16)

    def test_flat_range_reports_zero(self) -> None:
        flat = constant(20, 4.0)
        bars = frame(high=flat, low=flat, close=flat)
        result = bars.select(willr("high", "low", "close", 5)).to_series().to_list()
        self.assert_values_equal(result[4:], [0.0] * 16)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            willr("high", "low", "close", 0)
