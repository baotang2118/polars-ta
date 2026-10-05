import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, VOLUME, constant, frame, ramp_up

from polars_ta import ad, adosc, ema, obv

LENGTH = 60
BARS = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def reference_ad(
    high: list[float], low: list[float], close: list[float], volume: list[float]
) -> list[float]:
    total = 0.0
    result = []
    for index in range(len(close)):
        span = high[index] - low[index]
        if span > 0.0:
            multiplier = (
                (close[index] - low[index]) - (high[index] - close[index])
            ) / span
            total += multiplier * volume[index]
        result.append(total)
    return result


def reference_obv(close: list[float], volume: list[float]) -> list[float]:
    total = volume[0]
    result = [total]
    for index in range(1, len(close)):
        if close[index] > close[index - 1]:
            total += volume[index]
        elif close[index] < close[index - 1]:
            total -= volume[index]
        result.append(total)
    return result


class TestAd(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            column(ad("high", "low", "close", "volume")),
            reference_ad(HIGH[:LENGTH], LOW[:LENGTH], CLOSE[:LENGTH], VOLUME[:LENGTH]),
        )

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
            [100.0 * (index + 1) for index in range(10)],
        )

    def test_bar_with_no_range_contributes_nothing(self) -> None:
        flat = constant(6, 3.0)
        bars = pl.DataFrame(
            {"high": flat, "low": flat, "close": flat, "volume": [50.0] * 6}
        )
        self.assert_values_equal(
            column(ad("high", "low", "close", "volume"), bars), [0.0] * 6
        )

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            ad(pl.Series("high", HIGH[:5]), "low", "close", "volume")


class TestAdosc(IndicatorAssertions):
    def test_is_the_difference_of_two_recursive_emas(self) -> None:
        line = column(ad("high", "low", "close", "volume"))
        bars = pl.DataFrame({"line": line})
        fast = bars.select(ema("line", 3, mode="recursive")).to_series().to_list()
        slow = bars.select(ema("line", 10, mode="recursive")).to_series().to_list()
        expected = [
            None if f is None or s is None else f - s for f, s in zip(fast, slow)
        ]
        self.assert_values_equal(
            column(adosc("high", "low", "close", "volume")), expected
        )

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
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            column(obv("close", "volume")),
            reference_obv(CLOSE[:LENGTH], VOLUME[:LENGTH]),
        )

    def test_first_bar_seeds_with_its_own_volume(self) -> None:
        self.assertAlmostEqual(column(obv("close", "volume"))[0], VOLUME[0], places=10)

    def test_rising_series_accumulates(self) -> None:
        rising = ramp_up(10)
        bars = pl.DataFrame({"close": rising, "volume": [20.0] * 10})
        self.assert_values_equal(
            column(obv("close", "volume"), bars),
            [20.0 * (index + 1) for index in range(10)],
        )

    def test_unchanged_close_contributes_nothing(self) -> None:
        bars = pl.DataFrame({"close": constant(5, 2.0), "volume": [30.0] * 5})
        self.assert_values_equal(column(obv("close", "volume"), bars), [30.0] * 5)

    def test_series_input_keeps_its_name(self) -> None:
        result = obv(
            pl.Series("close", CLOSE[:LENGTH]), pl.Series("volume", VOLUME[:LENGTH])
        )
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")
