from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, VOLUME, constant, frame, ramp_down, ramp_up

from polars_ta import ao, ppo, pvo, tsi

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
TSI_13_25: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, 12.599668390650313,
    13.085913291626373, 14.312875394835844, 16.306208397915697, 18.349250186730423,
    21.576021687465865, 22.183717534404472, 23.280754248355496, 24.20699951943854,
    24.982276356800057, 25.454738315750195, 23.993513657573896,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestTsi(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = column(tsi("close", 13, 25))
        self.assert_values_equal(result[: len(TSI_13_25)], TSI_13_25)

    def test_warm_up_covers_both_passes(self) -> None:
        for fast, slow in ((3, 5), (5, 10), (13, 25)):
            with self.subTest(fast=fast, slow=slow):
                result = column(tsi("close", fast, slow))
                warm_up = slow + fast - 1
                self.assertEqual(result[:warm_up], [None] * warm_up)
                self.assertIsNotNone(result[warm_up])

    def test_a_rising_series_pins_at_one_hundred(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(40)})
        self.assertAlmostEqual(column(tsi("close", 5, 10), bars)[-1], 100.0)

    def test_a_falling_series_pins_at_minus_one_hundred(self) -> None:
        bars = pl.DataFrame({"close": ramp_down(40, 100.0)})
        self.assertAlmostEqual(column(tsi("close", 5, 10), bars)[-1], -100.0)

    def test_a_flat_series_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(40, 9.0)})
        self.assertAlmostEqual(column(tsi("close", 5, 10), bars)[-1], 0.0)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            tsi("close", 0)


class TestAo(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        middle = ramp_up(10)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in middle],
                "low": [value - 1.0 for value in middle],
            }
        )
        # Median prices step by one, so a 2-bar average leads a 4-bar one by 1.0.
        self.assert_values_equal(
            column(ao("high", "low", 2, 4), bars), [None] * 3 + [1.0] * 7
        )

    def test_warm_up_is_the_slow_period_minus_one(self) -> None:
        for fast, slow in ((5, 34), (3, 10)):
            with self.subTest(fast=fast, slow=slow):
                result = column(ao("high", "low", fast, slow))
                self.assertEqual(result[: slow - 1], [None] * (slow - 1))
                self.assertIsNotNone(result[slow - 1])

    def test_a_flat_series_reads_zero(self) -> None:
        flat = constant(40, 5.0)
        bars = pl.DataFrame({"high": flat, "low": flat})
        self.assertAlmostEqual(column(ao("high", "low", 5, 34), bars)[-1], 0.0)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            ao("high", "low", 0)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ao(cast(Any, pl.Series("high", HIGH[:5])), "low")


class TestPvo(IndicatorAssertions):
    def test_is_ppo_read_on_volume(self) -> None:
        self.assert_values_equal(
            column(pvo("volume", 5, 10)),
            column(ppo("volume", 5, 10, ma_type="ema")),
        )

    def test_defaults_to_exponential_averages(self) -> None:
        simple = column(ppo("volume", 5, 10, ma_type="sma"))
        exponential = column(pvo("volume", 5, 10))
        self.assertNotAlmostEqual(simple[-1], exponential[-1])

    def test_a_flat_volume_reads_zero(self) -> None:
        bars = BARS.with_columns(pl.lit(100.0).alias("volume"))
        self.assertAlmostEqual(column(pvo("volume", 5, 10), bars)[-1], 0.0)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            pvo("volume", 0)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            pvo(cast(Any, pl.Series("volume", VOLUME[:LENGTH])), 5, 10)


class TestAoAgainstLow(IndicatorAssertions):
    def test_uses_the_median_price(self) -> None:
        median = [(h + low) / 2.0 for h, low in zip(HIGH[:LENGTH], LOW[:LENGTH])]
        bars = pl.DataFrame({"median": median})
        expected = bars.select(
            pl.col("median").rolling_mean(window_size=5, min_samples=5)
            - pl.col("median").rolling_mean(window_size=34, min_samples=34)
        )
        self.assert_values_equal(
            column(ao("high", "low")), expected.to_series().to_list()
        )
