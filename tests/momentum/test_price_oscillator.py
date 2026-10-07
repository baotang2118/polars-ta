import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, ramp_down, ramp_up

from polars_ta import apo, ppo, sma

VALUES: list[float] = CLOSE[:60]


def evaluate(expr: pl.Expr, values: list[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def reference_sma(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(window - 1, len(values)):
        result[index] = sum(values[index - window + 1 : index + 1]) / window
    return result


class TestPriceOscillators(IndicatorAssertions):
    def test_apo_is_the_difference_of_two_averages(self) -> None:
        fast = reference_sma(VALUES, 5)
        slow = reference_sma(VALUES, 12)
        expected = [
            None if f is None or s is None else f - s for f, s in zip(fast, slow)
        ]
        self.assert_values_equal(evaluate(apo("close", 5, 12)), expected)

    def test_ppo_scales_apo_by_the_slow_average(self) -> None:
        absolute = evaluate(apo("close", 5, 12))
        slow = evaluate(sma("close", 12))
        expected = [
            None if a is None else a / s * 100.0 for a, s in zip(absolute, slow)
        ]
        self.assert_values_equal(evaluate(ppo("close", 5, 12)), expected)

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
        from polars_ta import ema

        fast = evaluate(ema("close", 5))
        slow = evaluate(ema("close", 12))
        expected = [
            None if f is None or s is None else f - s for f, s in zip(fast, slow)
        ]
        self.assert_values_equal(evaluate(apo("close", 5, 12, ma_type="ema")), expected)

    def test_default_periods_are_twelve_and_twenty_six(self) -> None:
        for function in (apo, ppo):
            with self.subTest(function=function.__name__):
                self.assert_values_equal(
                    evaluate(function("close")), evaluate(function("close", 12, 26))
                )

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            apo(pl.Series("close", VALUES), 5, 12)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            apo("close", 0, 12)
        with self.assertRaises(ValueError):
            ppo("close", 5, 12, ma_type="mesa")
