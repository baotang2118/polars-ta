import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, OPEN, constant, frame, ramp_down, ramp_up

from polars_ta import aroon, aroonosc, bop

LENGTH: int = 60


def unnest(expr: pl.Expr, bars: pl.DataFrame) -> dict[str, list]:
    result = bars.select(expr.alias("out")).unnest("out")
    return {name: result[name].to_list() for name in result.columns}


def reference_aroon(
    high: list[float], low: list[float], window: int
) -> tuple[list[float | None], list[float | None]]:
    """Aroon from the most recent extreme in the trailing ``window + 1`` bars."""
    down: list[float | None] = [None] * len(high)
    up: list[float | None] = [None] * len(high)
    factor = 100.0 / window
    for index in range(window, len(high)):
        start = index - window
        span_high = high[start : index + 1]
        span_low = low[start : index + 1]
        since_high = window - max(
            i for i, value in enumerate(span_high) if value == max(span_high)
        )
        since_low = window - max(
            i for i, value in enumerate(span_low) if value == min(span_low)
        )
        up[index] = factor * (window - since_high)
        down[index] = factor * (window - since_low)
    return down, up


class TestAroon(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        for window in (3, 5, 14):
            with self.subTest(window=window):
                fields = unnest(aroon("high", "low", window), bars)
                down, up = reference_aroon(HIGH[:LENGTH], LOW[:LENGTH], window)
                self.assert_values_equal(fields["down"], down)
                self.assert_values_equal(fields["up"], up)

    def test_warm_up_is_window_nulls(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        for window in (3, 5, 14):
            with self.subTest(window=window):
                fields = unnest(aroon("high", "low", window), bars)
                for name in ("down", "up"):
                    self.assertEqual(fields[name][:window], [None] * window)
                    self.assertIsNotNone(fields[name][window])

    def test_rising_series_pins_up_high_and_down_low(self) -> None:
        rising = ramp_up(30)
        bars = frame(high=rising, low=rising, close=rising)
        fields = unnest(aroon("high", "low", 5), bars)
        self.assert_values_equal(fields["up"][5:], [100.0] * 25)
        self.assert_values_equal(fields["down"][5:], [0.0] * 25)

    def test_falling_series_pins_down_high_and_up_low(self) -> None:
        falling = ramp_down(30, 40.0)
        bars = frame(high=falling, low=falling, close=falling)
        fields = unnest(aroon("high", "low", 5), bars)
        self.assert_values_equal(fields["down"][5:], [100.0] * 25)
        self.assert_values_equal(fields["up"][5:], [0.0] * 25)

    def test_flat_series_ties_resolve_to_the_newest_bar(self) -> None:
        flat = constant(20, 3.0)
        bars = frame(high=flat, low=flat, close=flat)
        fields = unnest(aroon("high", "low", 5), bars)
        self.assert_values_equal(fields["up"][5:], [100.0] * 15)
        self.assert_values_equal(fields["down"][5:], [100.0] * 15)

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        fields = unnest(aroon("high", "low"), bars)
        for name in ("down", "up"):
            for value in fields[name]:
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_oscillator_is_up_minus_down(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        fields = unnest(aroon("high", "low", 7), bars)
        expected = [
            None if u is None else u - d for u, d in zip(fields["up"], fields["down"])
        ]
        result = bars.select(aroonosc("high", "low", 7)).to_series().to_list()
        self.assert_values_equal(result, expected)

    def test_null_blanks_the_whole_window(self) -> None:
        highs = list(HIGH[:LENGTH])
        highs[10] = None
        bars = pl.DataFrame({"high": highs, "low": LOW[:LENGTH]})
        fields = unnest(aroon("high", "low", 4), bars)
        self.assertEqual(fields["up"][10:15], [None] * 5)
        self.assertIsNotNone(fields["up"][15])

    def test_default_window_is_fourteen(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        self.assert_values_equal(
            bars.select(aroonosc("high", "low")).to_series().to_list(),
            bars.select(aroonosc("high", "low", 14)).to_series().to_list(),
        )

    def test_invalid_window_raises(self) -> None:
        for function in (aroon, aroonosc):
            with (
                self.subTest(function=function.__name__),
                self.assertRaises(ValueError),
            ):
                function("high", "low", 0)


class TestBop(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        expected = [
            (CLOSE[index] - OPEN[index]) / (HIGH[index] - LOW[index])
            for index in range(LENGTH)
        ]
        self.assert_values_equal(result, expected)

    def test_has_no_warm_up(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assertIsNotNone(result[0])

    def test_output_stays_within_minus_one_and_one(self) -> None:
        bars = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH])
        for value in bars.select(bop("open", "high", "low", "close")).to_series():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_bar_closing_at_its_high_reports_one(self) -> None:
        bars = pl.DataFrame(
            {"open": [1.0], "high": [3.0], "low": [1.0], "close": [3.0]}
        )
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assert_values_equal(result, [1.0])

    def test_bar_with_no_range_reports_zero(self) -> None:
        flat = constant(5, 2.0)
        bars = frame(high=flat, low=flat, close=flat, open_=flat)
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assert_values_equal(result, [0.0] * 5)

    def test_null_propagates(self) -> None:
        bars = pl.DataFrame(
            {
                "open": [1.0, None],
                "high": [3.0, 3.0],
                "low": [1.0, 1.0],
                "close": [3.0, 2.0],
            }
        )
        result = bars.select(bop("open", "high", "low", "close")).to_series().to_list()
        self.assertIsNone(result[1])

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            bop(pl.Series("open", OPEN[:5]), "high", "low", "close")
