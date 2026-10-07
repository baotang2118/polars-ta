import math

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, frame, ramp_down, ramp_up

from polars_ta import ulcer

LENGTH = 60
BARS = frame(close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def reference_ulcer(values: list[float], window: int) -> list:
    drawdowns: list = [None] * (window - 1)
    for index in range(window - 1, len(values)):
        peak = max(values[index - window + 1 : index + 1])
        drawdowns.append(100.0 * (values[index] - peak) / peak)
    result: list = [None] * (2 * window - 2)
    for index in range(2 * window - 2, len(values)):
        squares = [value * value for value in drawdowns[index - window + 1 : index + 1]]
        result.append(math.sqrt(sum(squares) / window))
    return result


class TestUlcer(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            column(ulcer("close", 14)), reference_ulcer(CLOSE[:LENGTH], 14)
        )

    def test_warm_up_is_twice_the_window_minus_two(self) -> None:
        for window in (3, 5, 14):
            with self.subTest(window=window):
                result = column(ulcer("close", window))
                warm_up = 2 * (window - 1)
                self.assertEqual(result[:warm_up], [None] * warm_up)
                self.assertIsNotNone(result[warm_up])

    def test_a_rising_series_never_draws_down(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(12)})
        self.assert_values_equal(
            column(ulcer("close", 3), bars), [None] * 4 + [0.0] * 8
        )

    def test_a_flat_series_never_draws_down(self) -> None:
        bars = pl.DataFrame({"close": constant(8, 7.0)})
        self.assert_values_equal(
            column(ulcer("close", 3), bars), [None] * 4 + [0.0] * 4
        )

    def test_a_falling_series_draws_down(self) -> None:
        bars = pl.DataFrame({"close": ramp_down(10, 100.0)})
        result = column(ulcer("close", 3), bars)
        # Peaks sit two bars back, so each drawdown is -2%, -1%, 0% of 100ish.
        self.assertGreater(result[4], 0.0)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            ulcer("close", 0)

    def test_series_input_keeps_its_name(self) -> None:
        result = ulcer(pl.Series("close", CLOSE[:LENGTH]), 14)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")
