import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, ramp_up

from polars_ta import mass, vortex

LENGTH = 60
BARS = frame(close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def field(name: str, expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    return column(expr.struct.field(name), bars)


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


def reference_vortex(
    high: list[float], low: list[float], close: list[float], window: int
) -> tuple[list, list]:
    ranges: list = [None]
    plus: list = [None]
    minus: list = [None]
    for index in range(1, len(close)):
        ranges.append(
            max(
                high[index] - low[index],
                abs(high[index] - close[index - 1]),
                abs(low[index] - close[index - 1]),
            )
        )
        plus.append(abs(high[index] - low[index - 1]))
        minus.append(abs(low[index] - high[index - 1]))

    def ratio(movement: list) -> list:
        result: list = [None] * window
        for index in range(window, len(close)):
            window_range = sum(ranges[index - window + 1 : index + 1])
            moved = sum(movement[index - window + 1 : index + 1])
            result.append(moved / window_range if window_range != 0.0 else 0.0)
        return result

    return ratio(plus), ratio(minus)


def reference_mass(
    high: list[float], low: list[float], fast_period: int, slow_period: int
) -> list:
    amplitude = [h - low_ for h, low_ in zip(high, low)]
    single = recursive_ema(amplitude, fast_period)
    double = recursive_ema(single, fast_period)
    ratios: list = []
    for numerator, denominator in zip(single, double):
        if numerator is None or denominator is None:
            ratios.append(None)
        else:
            ratios.append(numerator / denominator if denominator != 0.0 else 0.0)
    result: list = []
    for index in range(len(ratios)):
        terms = ratios[index - slow_period + 1 : index + 1]
        if index < slow_period - 1 or any(term is None for term in terms):
            result.append(None)
        else:
            result.append(sum(terms))
    return result


class TestVortex(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        plus, minus = reference_vortex(HIGH[:LENGTH], LOW[:LENGTH], CLOSE[:LENGTH], 14)
        indicator = vortex("high", "low", "close", 14)
        self.assert_values_equal(field("plus", indicator), plus)
        self.assert_values_equal(field("minus", indicator), minus)

    def test_warm_up_is_the_window(self) -> None:
        for window in (5, 14, 21):
            with self.subTest(window=window):
                indicator = vortex("high", "low", "close", window)
                for name in ("plus", "minus"):
                    result = field(name, indicator)
                    self.assertEqual(result[:window], [None] * window)
                    self.assertIsNotNone(result[window])

    def test_a_rising_series_favours_the_plus_line(self) -> None:
        rising = ramp_up(20)
        bars = pl.DataFrame(
            {
                "high": rising,
                "low": [value - 1.0 for value in rising],
                "close": rising,
            }
        )
        indicator = vortex("high", "low", "close", 5)
        plus = field("plus", indicator, bars)
        minus = field("minus", indicator, bars)
        self.assertGreater(plus[-1], minus[-1])

    def test_a_flat_series_reports_zero(self) -> None:
        flat = constant(12, 4.0)
        bars = pl.DataFrame({"high": flat, "low": flat, "close": flat})
        indicator = vortex("high", "low", "close", 4)
        self.assert_values_equal(field("plus", indicator, bars), [None] * 4 + [0.0] * 8)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            vortex("high", "low", "close", 0)

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            vortex(pl.Series("high", HIGH[:5]), "low", "close")


class TestMass(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            column(mass("high", "low", 9, 25)),
            reference_mass(HIGH[:LENGTH], LOW[:LENGTH], 9, 25),
        )

    def test_warm_up_covers_both_averages_and_the_sum(self) -> None:
        for fast, slow in ((3, 4), (5, 10), (9, 25)):
            with self.subTest(fast=fast, slow=slow):
                result = column(mass("high", "low", fast, slow))
                warm_up = 2 * (fast - 1) + slow - 1
                self.assertEqual(result[:warm_up], [None] * warm_up)
                self.assertIsNotNone(result[warm_up])

    def test_a_constant_range_sums_to_the_slow_period(self) -> None:
        middle = ramp_up(40)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in middle],
                "low": [value - 1.0 for value in middle],
            }
        )
        self.assertAlmostEqual(column(mass("high", "low", 3, 4), bars)[-1], 4.0)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            mass("high", "low", 0, 25)
