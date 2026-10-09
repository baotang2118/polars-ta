from collections.abc import Sequence
from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, with_null
from pytest import raises

from polars_ta import kama, ma, mavp, sma, t3, trima
from polars_ta.overlay import MA_TYPES
from polars_ta.overlay.dispatch import MaType

VALUES: list[float] = CLOSE[:80]

# Frozen expectations: each table covers the warm-up plus the first live bars.
# fmt: off
TRIMA: dict[int, list[float | None]] = {
    2: [
        None, 9.5, 10.5, 10.5, 9.5, 9.5, 10.5, 11.5, 11.5, 10.5, 11.0, 12.5, 12.0,
    ],
    3: [
        None, None, 10.0, 10.5, 10.0, 9.5, 10.0, 11.0, 11.5, 11.0, 10.75, 11.75, 12.25,
        11.25,
    ],
    4: [
        None, None, None, 10.166666666666668, 10.166666666666668, 9.833333333333332,
        9.833333333333332, 10.5, 11.166666666666668, 11.166666666666668, 11.0,
        11.333333333333332, 11.833333333333332, 11.666666666666668, 10.666666666666668,
    ],
    5: [
        None, None, None, None, 10.111111111111112, 10.0, 9.888888888888888,
        10.222222222222221, 10.777777777777779, 11.111111111111112, 11.111111111111112,
        11.222222222222221, 11.555555555555555, 11.666666666666666, 11.111111111111112,
        10.444444444444445,
    ],
    10: [
        None, None, None, None, None, None, None, None, None, 10.3, 10.533333333333333,
        10.8, 11.0, 11.133333333333333, 11.2, 11.2, 11.077666666666667, 10.933, 10.971,
        11.235333333333333, 11.634666666666666,
    ],
    11: [
        None, None, None, None, None, None, None, None, None, None, 10.416666666666666,
        10.694444444444445, 10.916666666666666, 11.027777777777779, 11.083333333333334,
        11.166666666666666, 11.148055555555556, 11.027499999999998, 11.00361111111111,
        11.168333333333335, 11.528888888888888, 12.005833333333333,
    ],
}
KAMA: dict[int, list[float | None]] = {
    3: [
        None, None, None, 10.929651469020182, 10.793903322871564, 10.738053390367554,
        10.75648094955034, 11.309156083083522, 11.287407406795122, 11.196840186954548,
        11.200183203866287, 11.440741443701016, 11.425665833953703, 11.235115515529552,
        10.241730841960862,
    ],
    10: [
        None, None, None, None, None, None, None, None, None, None, 10.104643062925156,
        10.25613257200371, 10.25922879438871, 10.25814979836628, 10.252912962577033,
        10.262090542940989, 10.26903359082412, 10.28243303060794, 10.485535167631411,
        10.847611396731011, 10.941885139388805, 11.187020337226286,
    ],
    30: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, 13.922825980348334, 13.926472428919272,
        13.938408612123608, 13.933250566331623, 13.94231381777646, 13.925383858343935,
        13.9209728967971, 13.942244955694049, 13.987601517032497, 14.089179439421317,
        14.221281153661158, 14.361401448821146,
    ],
}
MAVP_CONSTANT_4: list[float | None] = [
    None, None, None, 10.0, 10.0, 10.0, 10.0, 10.5, 11.0, 11.0, 11.25, 11.5, 11.5,
    11.5, 10.75,
]
MAVP_CLAMPED_3: list[float | None] = [
    None, None, 10.0, 10.333333333333334, 10.0, 9.666666666666666, 10.0, 11.0,
    11.333333333333334, 11.0, 11.0, 11.666666666666666, 12.0, 11.333333333333334,
]
MAVP_VARYING: list[float | None] = [
    None, None, None, None, None, 10.0, 10.2, 10.5, 11.333333333333334, 11.0, 11.2,
    11.5, 12.0, 11.5, 11.0, 11.0, 10.443333333333333,
]
# fmt: on


def evaluate(expr: pl.Expr, values: Sequence[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


class TestTrima(IndicatorAssertions):
    def test_known_values(self) -> None:
        for window, expected in TRIMA.items():
            with self.subTest(window=window):
                result = evaluate(trima("close", window))
                self.assert_values_equal(result[: len(expected)], expected)

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        for window in (2, 3, 4, 5, 10, 11):
            with self.subTest(window=window):
                result = evaluate(trima("close", window))
                self.assertEqual(result[: window - 1], [None] * (window - 1))
                self.assertIsNotNone(result[window - 1])

    def test_window_of_one_returns_the_input(self) -> None:
        self.assert_values_equal(evaluate(trima("close", 1)), VALUES)

    def test_flat_series_equals_its_level(self) -> None:
        self.assert_values_equal(
            evaluate(trima("close", 5), constant(10))[4:], [5.0] * 6
        )

    def test_default_window_is_thirty(self) -> None:
        self.assert_values_equal(evaluate(trima("close")), evaluate(trima("close", 30)))

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), raises(ValueError):
                trima("close", cast(Any, window))


class TestT3(IndicatorAssertions):
    def test_warm_up_is_six_times_window_minus_one(self) -> None:
        for window in (2, 3, 5):
            with self.subTest(window=window):
                result = evaluate(t3("close", window))
                lookback = 6 * (window - 1)
                self.assertEqual(result[:lookback], [None] * lookback)
                self.assertIsNotNone(result[lookback])

    def test_zero_vfactor_is_the_third_ema_pass(self) -> None:
        from polars_ta.overlay.ma import _ema_expr

        alpha = 2.0 / 6.0
        stage = pl.col("close")
        for _ in range(3):
            stage = _ema_expr(stage, 5, alpha, "talib")
        # The warm-up still spans all six passes, as it does in TA-Lib.
        lookback = 6 * (5 - 1)
        self.assert_values_equal(
            evaluate(t3("close", 5, vfactor=0.0))[lookback:], evaluate(stage)[lookback:]
        )

    def test_flat_series_equals_its_level(self) -> None:
        flat = constant(40, 7.0)
        self.assert_values_equal(evaluate(t3("close", 3), flat)[12:], [7.0] * 28)

    def test_default_window_and_vfactor(self) -> None:
        self.assert_values_equal(
            evaluate(t3("close")), evaluate(t3("close", 5, vfactor=0.7))
        )

    def test_invalid_vfactor_raises(self) -> None:
        for vfactor in (-0.1, 1.1, "fast"):
            with self.subTest(vfactor=vfactor), raises(ValueError):
                t3("close", 5, vfactor=cast(Any, vfactor))


class TestKama(IndicatorAssertions):
    def test_known_values(self) -> None:
        for window, expected in KAMA.items():
            with self.subTest(window=window):
                result = evaluate(kama("close", window))
                self.assert_values_equal(result[: len(expected)], expected)

    def test_warm_up_is_window_nulls(self) -> None:
        for window in (3, 10, 30):
            with self.subTest(window=window):
                result = evaluate(kama("close", window))
                self.assertEqual(result[:window], [None] * window)
                self.assertIsNotNone(result[window])

    def test_flat_series_equals_its_level(self) -> None:
        self.assert_values_equal(
            evaluate(kama("close", 4), constant(12))[4:], [5.0] * 8
        )

    def test_null_restarts_the_recursion(self) -> None:
        result = evaluate(kama("close", 3), with_null(VALUES, 10))
        self.assertEqual(result[10:14], [None] * 4)
        self.assertIsNotNone(result[14])

    def test_faster_limit_tracks_price_more_closely(self) -> None:
        slowed = evaluate(kama("close", 10, fast_period=20, slow_period=40))
        quick = evaluate(kama("close", 10, fast_period=2, slow_period=4))
        slow_error = sum(abs(a - b) for a, b in zip(slowed[10:], VALUES[10:]))
        quick_error = sum(abs(a - b) for a, b in zip(quick[10:], VALUES[10:]))
        self.assertLess(quick_error, slow_error)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            kama(cast(Any, pl.Series("close", VALUES)), 10)

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), raises(ValueError):
                kama("close", cast(Any, window))


class TestMa(IndicatorAssertions):
    def test_every_type_is_supported(self) -> None:
        for ma_type in MA_TYPES:
            with self.subTest(ma_type=ma_type):
                result = evaluate(ma("close", 5, ma_type=cast(MaType, ma_type)))
                self.assertEqual(len(result), len(VALUES))
                self.assertTrue(any(value is not None for value in result))

    def test_dispatches_to_the_named_average(self) -> None:
        self.assert_values_equal(
            evaluate(ma("close", 7, ma_type="sma")), evaluate(sma("close", 7))
        )
        self.assert_values_equal(
            evaluate(ma("close", 7, ma_type="trima")), evaluate(trima("close", 7))
        )
        self.assert_values_equal(
            evaluate(ma("close", 7, ma_type="kama")), evaluate(kama("close", 7))
        )

    def test_default_is_a_thirty_period_sma(self) -> None:
        self.assert_values_equal(evaluate(ma("close")), evaluate(sma("close", 30)))

    def test_unknown_type_raises(self) -> None:
        with raises(ValueError):
            ma("close", 5, ma_type=cast(Any, "mesa"))


class TestMavp(IndicatorAssertions):
    def test_constant_periods_match_a_fixed_average(self) -> None:
        frame = pl.DataFrame({"close": VALUES, "periods": [4.0] * len(VALUES)})
        result = frame.select(mavp("close", "periods", 2, 10)).to_series().to_list()
        self.assertEqual(result[:9], [None] * 9)
        self.assert_values_equal(result[9 : len(MAVP_CONSTANT_4)], MAVP_CONSTANT_4[9:])

    def test_periods_are_clamped_to_the_bounds(self) -> None:
        wanted = [1.0] * len(VALUES)
        frame = pl.DataFrame({"close": VALUES, "periods": wanted})
        clamped = frame.select(mavp("close", "periods", 3, 8)).to_series().to_list()
        self.assert_values_equal(clamped[7 : len(MAVP_CLAMPED_3)], MAVP_CLAMPED_3[7:])

    def test_warm_up_is_the_longest_period(self) -> None:
        wanted = [2.0] * len(VALUES)
        frame = pl.DataFrame({"close": VALUES, "periods": wanted})
        result = frame.select(mavp("close", "periods", 2, 12)).to_series().to_list()
        self.assertEqual(result[:11], [None] * 11)
        self.assertIsNotNone(result[11])

    def test_varying_periods_select_row_by_row(self) -> None:
        wanted = [float(3 + index % 4) for index in range(len(VALUES))]
        frame = pl.DataFrame({"close": VALUES, "periods": wanted})
        result = frame.select(mavp("close", "periods", 3, 6)).to_series().to_list()
        self.assert_values_equal(result[: len(MAVP_VARYING)], MAVP_VARYING)

    def test_invalid_bounds_raise(self) -> None:
        with raises(ValueError):
            mavp("close", "periods", 10, 4)
        with raises(ValueError):
            mavp("close", "periods", 0, 4)
