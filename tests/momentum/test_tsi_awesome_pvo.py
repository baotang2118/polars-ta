import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, VOLUME, constant, frame, ramp_down, ramp_up

from polars_ta import ao, ppo, pvo, tsi

LENGTH = 60
BARS = frame(close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def recursive_ema(values: list, window: int) -> list:
    """``ewm_mean(adjust=False, min_samples=window)`` over a null-free series."""
    alpha = 2.0 / (window + 1.0)
    level = None
    seen = 0
    result: list = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        seen += 1
        level = value if level is None else level + alpha * (value - level)
        result.append(level if seen >= window else None)
    return result


def reference_tsi(values: list[float], fast_period: int, slow_period: int) -> list:
    change: list = [None] + [
        values[index] - values[index - 1] for index in range(1, len(values))
    ]
    magnitude: list = [None if value is None else abs(value) for value in change]

    def smooth(series: list) -> list:
        return recursive_ema(recursive_ema(series, slow_period), fast_period)

    smoothed = smooth(change)
    scaled = smooth(magnitude)
    result: list = []
    for numerator, denominator in zip(smoothed, scaled):
        if numerator is None or denominator is None:
            result.append(None)
        elif denominator == 0.0:
            result.append(0.0)
        else:
            result.append(100.0 * numerator / denominator)
    return result


class TestTsi(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            column(tsi("close", 13, 25)), reference_tsi(CLOSE[:LENGTH], 13, 25)
        )

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

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            ao(pl.Series("high", HIGH[:5]), "low")


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

    def test_series_input_keeps_its_name(self) -> None:
        result = pvo(pl.Series("volume", VOLUME[:LENGTH]), 5, 10)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "volume")


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
