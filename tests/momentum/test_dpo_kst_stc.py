import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HAND_CHECKED, constant, frame, ramp_up

from polars_ta import dpo, kst, stc

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])


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


def rolling_mean(values: list, window: int) -> list:
    result: list = []
    for index in range(len(values)):
        terms = values[index - window + 1 : index + 1]
        if index < window - 1 or any(term is None for term in terms):
            result.append(None)
        else:
            result.append(sum(terms) / window)
    return result


def reference_kst(
    values: list[float],
    roc_periods: tuple[int, ...],
    sma_periods: tuple[int, ...],
    signal_period: int,
) -> tuple[list, list]:
    line: list = [0.0] * len(values)
    known = [True] * len(values)
    for weight, (roc_period, sma_period) in enumerate(
        zip(roc_periods, sma_periods), start=1
    ):
        change: list = [None] * roc_period
        for index in range(roc_period, len(values)):
            before = values[index - roc_period]
            change.append(values[index] / before - 1.0 if before != 0.0 else 0.0)
        smoothed = rolling_mean(change, sma_period)
        for index, value in enumerate(smoothed):
            if value is None:
                known[index] = False
            else:
                line[index] += weight * value
    scaled: list = [100.0 * value if ok else None for value, ok in zip(line, known)]
    return scaled, rolling_mean(scaled, signal_period)


class TestDpo(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        bars = pl.DataFrame({"close": HAND_CHECKED})
        # window=4 averages four bars and compares against the bar three back.
        averages = rolling_mean(HAND_CHECKED, 4)
        expected = [
            None if average is None or index < 3 else HAND_CHECKED[index - 3] - average
            for index, average in enumerate(averages)
        ]
        self.assert_values_equal(column(dpo("close", 4), bars), expected)

    def test_warm_up_covers_the_average_and_the_shift(self) -> None:
        for window in (4, 10, 20):
            with self.subTest(window=window):
                result = column(dpo("close", window))
                warm_up = max(window - 1, window // 2 + 1)
                self.assertEqual(result[:warm_up], [None] * warm_up)
                self.assertIsNotNone(result[warm_up])

    def test_a_flat_series_detrends_to_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(20, 6.0)})
        self.assert_values_equal(column(dpo("close", 5), bars), [None] * 4 + [0.0] * 16)

    def test_a_rising_series_reads_negative(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(20)})
        self.assertLess(column(dpo("close", 6), bars)[-1], 0.0)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            dpo("close", 0)


class TestKst(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        roc_periods = (3, 5, 7, 9)
        sma_periods = (2, 2, 3, 4)
        line, signal = reference_kst(CLOSE[:LENGTH], roc_periods, sma_periods, 5)
        indicator = kst("close", roc_periods, sma_periods, 5)
        self.assert_values_equal(field("kst", indicator), line)
        self.assert_values_equal(field("signal", indicator), signal)

    def test_warm_up_follows_the_slowest_term(self) -> None:
        indicator = kst("close", (3, 5, 7, 9), (2, 2, 3, 4), 5)
        line = field("kst", indicator)
        warm_up = 9 + 4 - 1
        self.assertEqual(line[:warm_up], [None] * warm_up)
        self.assertIsNotNone(line[warm_up])

    def test_signal_trails_the_line(self) -> None:
        indicator = kst("close", (3, 5, 7, 9), (2, 2, 3, 4), 5)
        line = field("kst", indicator)
        signal = field("signal", indicator)
        self.assertIsNotNone(line[12])
        self.assertIsNone(signal[12])
        self.assertIsNotNone(signal[16])

    def test_a_flat_series_reads_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(40, 5.0)})
        self.assertAlmostEqual(
            field("kst", kst("close", (3, 5, 7, 9), (2, 2, 3, 4), 5), bars)[-1], 0.0
        )

    def test_invalid_periods_raise(self) -> None:
        with self.assertRaises(ValueError):
            kst("close", (10, 15, 20), (10, 10, 10, 15))
        with self.assertRaises(ValueError):
            kst("close", (10, 15, 20, 0))
        with self.assertRaises(ValueError):
            kst("close", signal_period=0)


class TestStc(IndicatorAssertions):
    def test_stays_inside_zero_and_one_hundred(self) -> None:
        result = column(stc("close", 5, 10, 4, smooth_k=3, smooth_d=3))
        emitted = [value for value in result if value is not None]
        self.assertTrue(emitted)
        for value in emitted:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_matches_reference(self) -> None:
        fast, slow, cycle, smooth = 5, 10, 4, 3
        emafast = recursive_ema(CLOSE[:LENGTH], fast)
        emaslow = recursive_ema(CLOSE[:LENGTH], slow)
        macd_line = [
            None if f is None or s is None else f - s for f, s in zip(emafast, emaslow)
        ]

        def rescale(values: list) -> list:
            result: list = []
            for index in range(len(values)):
                terms = values[index - cycle + 1 : index + 1]
                if (
                    index < cycle - 1
                    or values[index] is None
                    or any(term is None for term in terms)
                ):
                    result.append(None)
                    continue
                lowest, highest = min(terms), max(terms)
                span = highest - lowest
                result.append(
                    100.0 * (values[index] - lowest) / span if span != 0.0 else 0.0
                )
            return result

        first = recursive_ema(rescale(macd_line), smooth)
        expected = recursive_ema(rescale(first), smooth)
        self.assert_values_equal(
            column(stc("close", fast, slow, cycle, smooth_k=smooth, smooth_d=smooth)),
            expected,
        )

    def test_a_flat_series_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(60, 8.0)})
        result = column(stc("close", 5, 10, 4, smooth_k=3, smooth_d=3), bars)
        self.assertAlmostEqual(result[-1], 0.0)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            stc("close", 0)
        with self.assertRaises(ValueError):
            stc("close", cycle=0)

    def test_series_input_keeps_its_name(self) -> None:
        result = stc(pl.Series("close", CLOSE[:LENGTH]), 5, 10, 4)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")
