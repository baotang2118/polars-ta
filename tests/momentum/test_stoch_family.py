from collections.abc import Sequence

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, ramp_down, ramp_up

from polars_ta import cmo, rsi, stochf, stochrsi, willr

VALUES: list[float] = CLOSE[:60]

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
FAST_K_5: list[float | None] = [
    None, None, None, None, None, None, 66.66666666666667, 69.56521739130434,
    52.17391304347826, 25.0, 65.0, 70.83333333333333, 37.5, 20.833333333333332,
    20.689655172413794, 48.275862068965516, 72.95238095238095, 79.4407894736842,
]
FAST_D_5_3: list[float | None] = [
    None, None, None, None, None, None, 44.44444444444445, 60.225442834138484,
    62.80193236714976, 48.91304347826087, 47.391304347826086, 53.61111111111111,
    57.77777777777777, 43.05555555555555, 26.340996168582375, 29.93295019157088,
    47.305966064586755, 66.88967749834356,
]
WILLR_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    -70.37037037037037, -79.3103448275862, -51.724137931034484, -47.172413793103445,
    -33.37931034482759, -17.688679245283026, -19.358407079646028, -27.5442477876106,
    -12.57545271629779, -29.376257545271642, -15.794223826714813, -7.898894154818322,
]
# fmt: on


def evaluate(expr: pl.Expr, values: Sequence[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def unnest(expr: pl.Expr, bars: pl.DataFrame) -> dict[str, list]:
    result = bars.select(expr.alias("out")).unnest("out")
    return {name: result[name].to_list() for name in result.columns}


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
    def test_known_values(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        fields = unnest(stochf("high", "low", "close", 5, 3), bars)
        self.assert_values_equal(fields["fast_k"][: len(FAST_K_5)], FAST_K_5)
        self.assert_values_equal(fields["fast_d"][: len(FAST_D_5_3)], FAST_D_5_3)

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
    def test_known_values(self) -> None:
        bars = frame(high=HIGH[:60], low=LOW[:60], close=VALUES)
        result = bars.select(willr("high", "low", "close", 14)).to_series().to_list()
        self.assert_values_equal(result[: len(WILLR_14)], WILLR_14)

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
