from collections.abc import Sequence
from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, ramp_down, ramp_up
from pytest import raises

from polars_ta import apo, ppo

VALUES: list[float] = CLOSE[:60]

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
APO_5_12: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None,
    0.9333333333333336, 0.5666666666666664, 0.36666666666666536, 0.3333333333333339,
    0.05000000000000071, -0.4781666666666684, -0.4063333333333343, 0.19966666666666733,
    1.125166666666665, 1.4810000000000016, 1.9371666666666663, 2.165166666666668,
]
PPO_5_12: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None,
    8.750000000000002, 5.230769230769228, 3.3846153846153726, 3.1250000000000053,
    0.46511627906977404, -4.369146425036184, -3.64806224749365, 1.7471197316610823,
    9.64911026942041, 12.429710449013863, 15.583562378494332, 17.128353879622924,
]
APO_EMA_5_12: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None,
    1.040420667581163, 0.7534428382165022, 0.3734326350766164, -0.03956966592929945,
    0.08845824970476279, 0.2153735739237348, 0.4554092145639874, 0.9533570521270036,
    1.2410016244893711, 1.206801043540482, 1.5097237392339053, 1.3034378886387117,
]
# fmt: on


def evaluate(expr: pl.Expr, values: Sequence[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


class TestPriceOscillators(IndicatorAssertions):
    def test_apo_is_the_difference_of_two_averages(self) -> None:
        result = evaluate(apo("close", 5, 12))
        self.assert_values_equal(result[: len(APO_5_12)], APO_5_12)

    def test_ppo_scales_apo_by_the_slow_average(self) -> None:
        result = evaluate(ppo("close", 5, 12))
        self.assert_values_equal(result[: len(PPO_5_12)], PPO_5_12)

    def test_warm_up_follows_the_slower_average(self) -> None:
        for function in (apo, ppo):
            with self.subTest(function=function.__name__):
                result = evaluate(function("close", 4, 11))
                self.assertEqual(result[:10], [None] * 10)
                self.assertIsNotNone(result[10])

    def test_periods_are_ordered_before_use(self) -> None:
        self.assert_values_equal(
            evaluate(apo("close", 20, 6)), evaluate(apo("close", 6, 20))
        )

    def test_flat_series_has_no_spread(self) -> None:
        flat = constant(30, 9.0)
        self.assert_values_equal(evaluate(apo("close", 3, 8), flat)[7:], [0.0] * 23)
        self.assert_values_equal(evaluate(ppo("close", 3, 8), flat)[7:], [0.0] * 23)

    def test_rising_series_is_positive(self) -> None:
        for value in evaluate(apo("close", 3, 8), ramp_up(30))[7:]:
            self.assertGreater(value, 0.0)

    def test_falling_series_is_negative(self) -> None:
        for value in evaluate(apo("close", 3, 8), ramp_down(30, 40.0))[7:]:
            self.assertLess(value, 0.0)

    def test_zero_slow_average_reports_zero(self) -> None:
        zeros = constant(10, 0.0)
        self.assert_values_equal(evaluate(ppo("close", 2, 4), zeros)[3:], [0.0] * 7)

    def test_ma_type_is_honoured(self) -> None:
        result = evaluate(apo("close", 5, 12, ma_type="ema"))
        self.assert_values_equal(result[: len(APO_EMA_5_12)], APO_EMA_5_12)

    def test_default_periods_are_twelve_and_twenty_six(self) -> None:
        for function in (apo, ppo):
            with self.subTest(function=function.__name__):
                self.assert_values_equal(
                    evaluate(function("close")), evaluate(function("close", 12, 26))
                )

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            apo(cast(Any, pl.Series("close", VALUES)), 5, 12)

    def test_invalid_arguments_raise(self) -> None:
        with raises(ValueError):
            apo("close", 0, 12)
        with raises(ValueError):
            ppo("close", 5, 12, ma_type=cast(Any, "mesa"))
