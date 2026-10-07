from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, VOLUME, constant, frame, ramp_up

from polars_ta import ad, adosc, cmf, obv

LENGTH: int = 60
BARS: pl.DataFrame = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars. The canonical bars
# are symmetric about the close, so the accumulation multiplier is zero.
# fmt: off
AD: list[float | None] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]
ADOSC_3_10: list[float | None] = [
    None, None, None, None, None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]
OBV: list[float | None] = [
    150.0, 400.0, 750.0, 300.0, -250.0, 400.0, 1150.0, 2000.0, 1050.0, 900.0, 1150.0,
    1500.0,
]
CMF_20: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, 0.0, 0.0, 2.9882638419817274e-17,
    2.9334333127710535e-17, -9.14469415392702e-19, -8.982841160052205e-19,
    -8.82661783552956e-19, -8.67573547936666e-19, -9.227827737144538e-19,
    2.057223246739539e-17, 2.0180380420397382e-17, 1.9803177048053505e-17,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestAd(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = column(ad("high", "low", "close", "volume"))
        self.assert_values_equal(result[: len(AD)], AD)

    def test_has_no_warm_up(self) -> None:
        self.assertIsNotNone(column(ad("high", "low", "close", "volume"))[0])

    def test_close_at_the_high_accumulates_full_volume(self) -> None:
        rising = ramp_up(10)
        bars = pl.DataFrame(
            {
                "high": rising,
                "low": [value - 1.0 for value in rising],
                "close": rising,
                "volume": [100.0] * 10,
            }
        )
        self.assert_values_equal(
            column(ad("high", "low", "close", "volume"), bars),
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        )

    def test_bar_with_no_range_contributes_nothing(self) -> None:
        flat = constant(6, 3.0)
        bars = pl.DataFrame(
            {"high": flat, "low": flat, "close": flat, "volume": [50.0] * 6}
        )
        self.assert_values_equal(
            column(ad("high", "low", "close", "volume"), bars), [0.0] * 6
        )

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ad(cast(Any, pl.Series("high", HIGH[:5])), "low", "close", "volume")


class TestAdosc(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = column(adosc("high", "low", "close", "volume"))
        self.assert_values_equal(result[: len(ADOSC_3_10)], ADOSC_3_10)

    def test_warm_up_is_slow_period_minus_one(self) -> None:
        for fast, slow in ((3, 10), (2, 6)):
            with self.subTest(fast=fast, slow=slow):
                result = column(adosc("high", "low", "close", "volume", fast, slow))
                self.assertEqual(result[: slow - 1], [None] * (slow - 1))
                self.assertIsNotNone(result[slow - 1])

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            adosc("high", "low", "close", "volume", 0, 10)


class TestObv(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = column(obv("close", "volume"))
        self.assert_values_equal(result[: len(OBV)], OBV)

    def test_first_bar_seeds_with_its_own_volume(self) -> None:
        self.assertAlmostEqual(column(obv("close", "volume"))[0], VOLUME[0], places=10)

    def test_rising_series_accumulates(self) -> None:
        rising = ramp_up(10)
        bars = pl.DataFrame({"close": rising, "volume": [20.0] * 10})
        self.assert_values_equal(
            column(obv("close", "volume"), bars),
            [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
        )

    def test_unchanged_close_contributes_nothing(self) -> None:
        bars = pl.DataFrame({"close": constant(5, 2.0), "volume": [30.0] * 5})
        self.assert_values_equal(column(obv("close", "volume"), bars), [30.0] * 5)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            obv(
                cast(Any, pl.Series("close", CLOSE[:LENGTH])),
                cast(Any, pl.Series("volume", VOLUME[:LENGTH])),
            )


class TestCmf(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = column(cmf("high", "low", "close", "volume", 20))
        self.assert_values_equal(result[: len(CMF_20)], CMF_20)

    def test_warm_up_is_window_minus_one(self) -> None:
        for window in (5, 14, 20):
            with self.subTest(window=window):
                result = column(cmf("high", "low", "close", "volume", window))
                self.assertEqual(result[: window - 1], [None] * (window - 1))
                self.assertIsNotNone(result[window - 1])

    def test_close_at_the_high_reads_one(self) -> None:
        rising = ramp_up(10)
        bars = pl.DataFrame(
            {
                "high": rising,
                "low": [value - 1.0 for value in rising],
                "close": rising,
                "volume": [100.0] * 10,
            }
        )
        self.assert_values_equal(
            column(cmf("high", "low", "close", "volume", 4), bars),
            [None, None, None] + [1.0] * 7,
        )

    def test_window_with_no_volume_reports_zero(self) -> None:
        flat = constant(5, 3.0)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in flat],
                "low": [value - 1.0 for value in flat],
                "close": flat,
                "volume": [0.0] * 5,
            }
        )
        self.assert_values_equal(
            column(cmf("high", "low", "close", "volume", 3), bars),
            [None, None, 0.0, 0.0, 0.0],
        )

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            cmf("high", "low", "close", "volume", 0)
