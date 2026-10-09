from collections.abc import Sequence
from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import HAND_CHECKED
from pytest import raises

from polars_ta import dema, ema, sma, tema, wma

# The literal expectations below were worked out against this exact series.
VALUES: list[float] = HAND_CHECKED[:8]
LONG_VALUES: list[float] = HAND_CHECKED
SHORT_VALUES: list[float] = [1.0, 3.0, 2.0, 6.0, 5.0]
NULL_VALUES: list[float | None] = [1.0, 2.0, None, 4.0, 5.0, 6.0]

SMA_3: list[float | None] = [
    None,
    None,
    2.0,
    3.6666666666666665,
    4.333333333333333,
    6.666666666666667,
    6.0,
    7.0,
]
EMA_TALIB_3: list[float | None] = [None, None, 2.0, 4.0, 4.5, 6.75, 5.375, 6.6875]
EMA_TALIB_3_ALPHA_025: list[float | None] = [
    None,
    None,
    2.0,
    3.0,
    3.5,
    4.875,
    4.65625,
    5.4921875,
]
EMA_RECURSIVE_4: list[float | None] = [
    None,
    None,
    None,
    3.5280000000000005,
    4.1168,
    6.07008,
    5.2420480000000005,
    6.345228800000001,
]
EMA_ADJUST_4: list[float | None] = [
    None,
    None,
    None,
    3.9044117647058822,
    4.3795975017349065,
    6.318206229860366,
    5.364218177987306,
    6.436541826362273,
]
WMA_3: list[float | None] = [
    None,
    None,
    2.1666666666666665,
    4.166666666666667,
    4.833333333333333,
    7.166666666666667,
    5.833333333333333,
    6.833333333333333,
]
WMA_3_SHORT: list[float | None] = [
    None,
    None,
    2.1666666666666665,
    4.166666666666667,
    4.833333333333333,
]
WMA_3_WITH_NULL: list[float | None] = [None, None, None, None, None, 5.333333333333333]
DEMA_3: list[float | None] = [
    None,
    None,
    None,
    None,
    5.5,
    8.375,
    5.5,
    7.40625,
    7.28125,
    10.1796875,
    7.359375,
    9.314453125,
    9.224609375,
    12.14599609375,
    9.33984375,
    11.3033447265625,
]
DEMA_3_ALPHA_025: list[float | None] = [
    None,
    None,
    None,
    None,
    4.166666666666666,
    6.40625,
    5.640625,
    6.857421875,
    7.17578125,
    9.0938720703125,
    8.10443115234375,
    9.166343688964844,
    9.378273010253906,
    11.223841190338135,
    10.18548321723938,
    11.214814156293869,
]
TEMA_3: list[float | None] = [
    None,
    None,
    None,
    None,
    None,
    None,
    5.0,
    7.453125,
    7.1640625,
    10.53125,
    6.85546875,
    9.4052734375,
    9.15771484375,
    12.53955078125,
    8.86669921875,
    11.41510009765625,
]
TEMA_3_ALPHA_025: list[float | None] = [
    None,
    None,
    None,
    None,
    None,
    None,
    6.236111111111111,
    7.589680989583334,
    7.6810302734375,
    9.9493408203125,
    8.219924926757812,
    9.46137809753418,
    9.504980564117432,
    11.762911558151245,
    10.043415188789368,
    11.304559595882893,
]


def evaluate(
    expr: pl.Expr, values: Sequence[float | None] | None = None
) -> list[float | None]:
    """Collect an indicator expression over ``values`` into a Python list."""
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


class TestSma(IndicatorAssertions):
    def test_known_values(self) -> None:
        self.assert_values_equal(evaluate(sma("close", 3)), SMA_3)

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        result = evaluate(sma("close", 4))
        self.assertEqual(result[:3], [None, None, None])
        self.assertIsNotNone(result[3])

    def test_window_of_one_is_identity(self) -> None:
        self.assert_values_equal(evaluate(sma("close", 1)), VALUES)

    def test_null_blanks_every_overlapping_window(self) -> None:
        self.assert_values_equal(
            evaluate(sma("close", 3), NULL_VALUES),
            [None, None, None, None, None, 5.0],
        )

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(sma("close", len(VALUES) + 1)), [None] * len(VALUES))

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1):
            with self.subTest(window=window), raises(ValueError):
                sma("close", window)

    def test_non_integer_window_raises(self) -> None:
        with raises(ValueError):
            sma("close", cast(Any, 2.5))


class TestEma(IndicatorAssertions):
    def test_talib_mode_matches_sma_seeded_recursion(self) -> None:
        self.assert_values_equal(evaluate(ema("close", 3)), EMA_TALIB_3)

    def test_recursive_mode_matches_unadjusted_recursion(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 4, mode="recursive")), EMA_RECURSIVE_4
        )

    def test_adjust_mode_matches_weighted_average(self) -> None:
        self.assert_values_equal(evaluate(ema("close", 4, mode="adjust")), EMA_ADJUST_4)

    def test_explicit_alpha_overrides_window_default(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 3, alpha=0.25)), EMA_TALIB_3_ALPHA_025
        )

    def test_default_alpha_is_two_over_window_plus_one(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 4)), evaluate(ema("close", 4, alpha=0.4))
        )

    def test_every_mode_warms_up_for_window_minus_one_rows(self) -> None:
        for mode in ("talib", "adjust", "recursive"):
            with self.subTest(mode=mode):
                result = evaluate(ema("close", 3, mode=mode))
                self.assertEqual(result[:2], [None, None])
                self.assertIsNotNone(result[2])

    def test_null_in_seed_window_delays_the_seed(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 3, alpha=0.5), NULL_VALUES),
            [None, None, None, None, None, 5.0],
        )

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(ema("close", len(VALUES) + 1)), [None] * len(VALUES))

    def test_invalid_alpha_raises(self) -> None:
        for alpha in (0.0, -0.5, 1.5):
            with self.subTest(alpha=alpha), raises(ValueError):
                ema("close", 3, alpha=alpha)

    def test_alpha_of_one_is_allowed(self) -> None:
        self.assert_values_equal(evaluate(ema("close", 1, alpha=1.0)), VALUES)

    def test_invalid_mode_raises(self) -> None:
        with raises(ValueError):
            ema("close", 3, mode=cast(Any, "exponential"))

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            ema("close", 0)


class TestWma(IndicatorAssertions):
    def test_known_values_over_the_hand_checked_series(self) -> None:
        self.assert_values_equal(evaluate(wma("close", 3)), WMA_3)

    def test_known_values(self) -> None:
        self.assert_values_equal(evaluate(wma("close", 3), SHORT_VALUES), WMA_3_SHORT)

    def test_weights_favour_the_most_recent_value(self) -> None:
        values = [0.0, 0.0, 3.0]
        self.assert_values_equal(evaluate(wma("close", 3), values), [None, None, 1.5])

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        result = evaluate(wma("close", 4))
        self.assertEqual(result[:3], [None, None, None])
        self.assertIsNotNone(result[3])

    def test_window_of_one_is_identity(self) -> None:
        self.assert_values_equal(evaluate(wma("close", 1)), VALUES)

    def test_null_blanks_every_overlapping_window(self) -> None:
        self.assert_values_equal(
            evaluate(wma("close", 3), NULL_VALUES), WMA_3_WITH_NULL
        )

    def test_integer_input_matches_float_input(self) -> None:
        integers = pl.DataFrame({"close": [1, 3, 2, 6, 5]})
        self.assert_values_equal(
            integers.select(wma("close", 3)).to_series().to_list(), WMA_3_SHORT
        )

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(wma("close", len(VALUES) + 1)), [None] * len(VALUES))

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1):
            with self.subTest(window=window), raises(ValueError):
                wma("close", window)


class TestDema(IndicatorAssertions):
    def test_known_values_over_the_long_series(self) -> None:
        self.assert_values_equal(evaluate(dema("close", 3), LONG_VALUES), DEMA_3)

    def test_known_values(self) -> None:
        result = evaluate(dema("close", 3), LONG_VALUES)
        self.assert_values_equal(result[4:6], [5.5, 8.375])

    def test_warm_up_is_twice_window_minus_one(self) -> None:
        for window in (2, 3, 4):
            with self.subTest(window=window):
                result = evaluate(dema("close", window), LONG_VALUES)
                self.assertEqual(result[: 2 * (window - 1)], [None] * 2 * (window - 1))
                self.assertIsNotNone(result[2 * (window - 1)])

    def test_explicit_alpha_overrides_window_default(self) -> None:
        self.assert_values_equal(
            evaluate(dema("close", 3, alpha=0.25), LONG_VALUES), DEMA_3_ALPHA_025
        )

    def test_every_mode_is_supported(self) -> None:
        for mode in ("talib", "adjust", "recursive"):
            with self.subTest(mode=mode):
                result = evaluate(dema("close", 3, mode=mode), LONG_VALUES)
                self.assertEqual(result[:4], [None] * 4)
                self.assertIsNotNone(result[4])

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(dema("close", len(VALUES))), [None] * len(VALUES))

    def test_invalid_arguments_raise(self) -> None:
        with raises(ValueError):
            dema("close", 0)
        with raises(ValueError):
            dema("close", 3, alpha=0.0)
        with raises(ValueError):
            dema("close", 3, mode=cast(Any, "double"))


class TestTema(IndicatorAssertions):
    def test_known_values_over_the_long_series(self) -> None:
        self.assert_values_equal(evaluate(tema("close", 3), LONG_VALUES), TEMA_3)

    def test_known_values(self) -> None:
        result = evaluate(tema("close", 3), LONG_VALUES)
        self.assert_values_equal(result[6:8], [5.0, 7.453125])

    def test_warm_up_is_three_times_window_minus_one(self) -> None:
        for window in (2, 3, 4):
            with self.subTest(window=window):
                result = evaluate(tema("close", window), LONG_VALUES)
                self.assertEqual(result[: 3 * (window - 1)], [None] * 3 * (window - 1))
                self.assertIsNotNone(result[3 * (window - 1)])

    def test_explicit_alpha_overrides_window_default(self) -> None:
        self.assert_values_equal(
            evaluate(tema("close", 3, alpha=0.25), LONG_VALUES), TEMA_3_ALPHA_025
        )

    def test_every_mode_is_supported(self) -> None:
        for mode in ("talib", "adjust", "recursive"):
            with self.subTest(mode=mode):
                result = evaluate(tema("close", 3, mode=mode), LONG_VALUES)
                self.assertEqual(result[:6], [None] * 6)
                self.assertIsNotNone(result[6])

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(tema("close", len(VALUES))), [None] * len(VALUES))

    def test_invalid_arguments_raise(self) -> None:
        with raises(ValueError):
            tema("close", 0)
        with raises(ValueError):
            tema("close", 3, alpha=1.5)
        with raises(ValueError):
            tema("close", 3, mode=cast(Any, "triple"))


class TestInputForms(IndicatorAssertions):
    def test_name_and_expression_agree(self) -> None:
        for indicator in (sma, ema, wma, dema, tema):
            with self.subTest(indicator=indicator.__name__):
                from_name = evaluate(indicator("close", 3), LONG_VALUES)
                from_expr = evaluate(indicator(pl.col("close"), 3), LONG_VALUES)
                self.assert_values_equal(from_expr, from_name)

    def test_series_input_is_rejected(self) -> None:
        for indicator in (sma, ema, wma, dema, tema):
            with (
                self.subTest(indicator=indicator.__name__),
                raises(TypeError),
            ):
                indicator(cast(Any, pl.Series("close", VALUES)), 3)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        frame = pl.LazyFrame({"close": VALUES}).with_columns(
            sma("close", 3).alias("sma"), ema("close", 3).alias("ema")
        )
        collected = frame.collect()
        self.assertEqual(collected.columns, ["close", "sma", "ema"])
        self.assert_values_equal(collected["ema"].to_list(), EMA_TALIB_3)

    def test_integer_input_produces_float_output(self) -> None:
        frame = pl.DataFrame({"close": [1, 2, 3, 4]})
        result = frame.select(sma("close", 2), ema("close", 2).alias("ema"))
        self.assertEqual(result["close"].dtype, pl.Float64)
        self.assertEqual(result["ema"].dtype, pl.Float64)

    def test_unsupported_input_type_raises(self) -> None:
        with raises(TypeError):
            sma(cast(Any, [1.0, 2.0]), 2)
