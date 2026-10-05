import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, with_null

from polars_ta import kama, ma, mavp, sma, t3, trima
from polars_ta.overlay import MA_TYPES

VALUES: list[float] = CLOSE[:80]


def evaluate(expr: pl.Expr, values: list[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def reference_sma(values: list[float | None], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    for index in range(window - 1, len(values)):
        span = values[index - window + 1 : index + 1]
        if any(value is None for value in span):
            continue
        result[index] = sum(span) / window
    return result


def reference_trima(values: list[float], window: int) -> list[float | None]:
    """TRIMA as the convolution of two simple moving averages."""
    half = window // 2
    first, second = (half + 1, half + 1) if window % 2 else (half + 1, half)
    return reference_sma(reference_sma(values, first), second)


def reference_kama(
    values: list[float], window: int, fast: int = 2, slow: int = 30
) -> list[float | None]:
    """TA-Lib's KAMA recursion with its per-bar smoothing constant."""
    result: list[float | None] = [None] * len(values)
    fastest = 2.0 / (fast + 1.0)
    slowest = 2.0 / (slow + 1.0)
    previous = None
    for index in range(window, len(values)):
        volatility = sum(
            abs(values[i] - values[i - 1]) for i in range(index - window + 1, index + 1)
        )
        direction = values[index] - values[index - window]
        if volatility <= direction or volatility == 0.0:
            efficiency = 1.0
        else:
            efficiency = abs(direction / volatility)
        factor = (efficiency * (fastest - slowest) + slowest) ** 2
        if previous is None:
            previous = values[index - 1]
        previous += factor * (values[index] - previous)
        result[index] = previous
    return result


class TestTrima(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        for window in (2, 3, 4, 5, 10, 11):
            with self.subTest(window=window):
                self.assert_values_equal(
                    evaluate(trima("close", window)), reference_trima(VALUES, window)
                )

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
            with self.subTest(window=window), self.assertRaises(ValueError):
                trima("close", window)


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
            with self.subTest(vfactor=vfactor), self.assertRaises(ValueError):
                t3("close", 5, vfactor=vfactor)


class TestKama(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        for window in (3, 10, 30):
            with self.subTest(window=window):
                self.assert_values_equal(
                    evaluate(kama("close", window)), reference_kama(VALUES, window)
                )

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

    def test_series_input_keeps_its_name(self) -> None:
        result = kama(pl.Series("close", VALUES), 10)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                kama("close", window)


class TestMa(IndicatorAssertions):
    def test_every_type_is_supported(self) -> None:
        for ma_type in MA_TYPES:
            with self.subTest(ma_type=ma_type):
                result = evaluate(ma("close", 5, ma_type=ma_type))
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
        with self.assertRaises(ValueError):
            ma("close", 5, ma_type="mesa")


class TestMavp(IndicatorAssertions):
    def test_constant_periods_match_a_fixed_average(self) -> None:
        frame = pl.DataFrame({"close": VALUES, "periods": [4.0] * len(VALUES)})
        result = frame.select(mavp("close", "periods", 2, 10)).to_series().to_list()
        expected = reference_sma(VALUES, 4)
        self.assertEqual(result[:9], [None] * 9)
        self.assert_values_equal(result[9:], expected[9:])

    def test_periods_are_clamped_to_the_bounds(self) -> None:
        wanted = [1.0] * len(VALUES)
        frame = pl.DataFrame({"close": VALUES, "periods": wanted})
        clamped = frame.select(mavp("close", "periods", 3, 8)).to_series().to_list()
        expected = reference_sma(VALUES, 3)
        self.assert_values_equal(clamped[7:], expected[7:])

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
        references = {window: reference_sma(VALUES, window) for window in range(3, 7)}
        expected = [
            None if index < 5 else references[int(wanted[index])][index]
            for index in range(len(VALUES))
        ]
        self.assert_values_equal(result, expected)

    def test_invalid_bounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            mavp("close", "periods", 10, 4)
        with self.assertRaises(ValueError):
            mavp("close", "periods", 0, 4)
