from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, frame, ramp_down, ramp_up

from polars_ta import ulcer

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
ULCER_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None,
    12.09919261141096, 10.761437169239986, 8.916644849689657, 11.167599416747077,
    14.22040644778843, 16.06372685089913, 17.443647758270625, 19.35278139443941,
    20.64532933415509, 23.13322308521081, 24.487138130267855, 25.145959952424935,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestUlcer(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = column(ulcer("close", 14))
        self.assert_values_equal(result[: len(ULCER_14)], ULCER_14)

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

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ulcer(cast(Any, pl.Series("close", CLOSE[:LENGTH])), 14)
