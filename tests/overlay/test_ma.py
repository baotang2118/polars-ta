import polars as pl
from _assertions import IndicatorAssertions
from _data import HAND_CHECKED

from polars_ta import dema, ema, sma, tema, wma

# The literal expectations below were worked out against this exact series.
VALUES: list[float] = HAND_CHECKED[:8]
LONG_VALUES: list[float] = HAND_CHECKED


def evaluate(
    expr: pl.Expr, values: list[float | None] | None = None
) -> list[float | None]:
    """Collect an indicator expression over ``values`` into a Python list."""
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def reference_sma(values: list[float | None], window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        chunk = values[index + 1 - window : index + 1]
        result.append(None if any(v is None for v in chunk) else sum(chunk) / window)
    return result


def reference_wma(values: list[float | None], window: int) -> list[float | None]:
    denominator = window * (window + 1) / 2
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        chunk = values[index + 1 - window : index + 1]
        if any(v is None for v in chunk):
            result.append(None)
            continue
        weighted = sum((offset + 1) * v for offset, v in enumerate(chunk))
        result.append(weighted / denominator)
    return result


def reference_ema_talib(values: list[float | None], window: int, alpha: float):
    """TA-Lib EMA seeded at the first complete, null-free window."""
    result: list[float | None] = [None] * len(values)
    seed_index = None
    for index in range(window - 1, len(values)):
        chunk = values[index + 1 - window : index + 1]
        if all(v is not None for v in chunk):
            seed_index = index
            previous = sum(chunk) / window
            break
    if seed_index is None:
        return result
    result[seed_index] = previous
    for index in range(seed_index + 1, len(values)):
        previous = alpha * values[index] + (1.0 - alpha) * previous
        result[index] = previous
    return result


def reference_dema(values: list[float], window: int, alpha: float):
    first = reference_ema_talib(values, window, alpha)
    second = reference_ema_talib(first, window, alpha)
    return [
        None if a is None or b is None else 2.0 * a - b for a, b in zip(first, second)
    ]


def reference_tema(values: list[float], window: int, alpha: float):
    first = reference_ema_talib(values, window, alpha)
    second = reference_ema_talib(first, window, alpha)
    third = reference_ema_talib(second, window, alpha)
    return [
        None if a is None or b is None or c is None else 3.0 * a - 3.0 * b + c
        for a, b, c in zip(first, second, third)
    ]


def reference_ema_recursive(values: list[float], window: int, alpha: float):
    result: list[float | None] = []
    previous = values[0]
    for index, value in enumerate(values):
        if index > 0:
            previous = alpha * value + (1.0 - alpha) * previous
        result.append(None if index + 1 < window else previous)
    return result


def reference_ema_adjust(values: list[float], window: int, alpha: float):
    result: list[float | None] = []
    for index in range(len(values)):
        weights = [(1.0 - alpha) ** offset for offset in range(index + 1)]
        numerator = sum(w * values[index - o] for o, w in enumerate(weights))
        result.append(None if index + 1 < window else numerator / sum(weights))
    return result


class TestSma(IndicatorAssertions):
    def test_matches_reference_mean(self) -> None:
        self.assert_values_equal(evaluate(sma("close", 3)), reference_sma(VALUES, 3))

    def test_known_values(self) -> None:
        self.assert_values_equal(
            evaluate(sma("close", 3)),
            [None, None, 2.0, 11 / 3, 13 / 3, 20 / 3, 6.0, 7.0],
        )

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        result = evaluate(sma("close", 4))
        self.assertEqual(result[:3], [None, None, None])
        self.assertIsNotNone(result[3])

    def test_window_of_one_is_identity(self) -> None:
        self.assert_values_equal(evaluate(sma("close", 1)), VALUES)

    def test_null_blanks_every_overlapping_window(self) -> None:
        values = [1.0, 2.0, None, 4.0, 5.0, 6.0]
        self.assert_values_equal(
            evaluate(sma("close", 3), values), [None, None, None, None, None, 5.0]
        )

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(sma("close", len(VALUES) + 1)), [None] * len(VALUES))

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1):
            with self.subTest(window=window), self.assertRaises(ValueError):
                sma("close", window)

    def test_non_integer_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            sma("close", 2.5)


class TestEma(IndicatorAssertions):
    def test_talib_mode_matches_sma_seeded_recursion(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 3)), reference_ema_talib(VALUES, 3, 0.5)
        )

    def test_recursive_mode_matches_unadjusted_recursion(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 4, mode="recursive")),
            reference_ema_recursive(VALUES, 4, 0.4),
        )

    def test_adjust_mode_matches_weighted_average(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 4, mode="adjust")),
            reference_ema_adjust(VALUES, 4, 0.4),
        )

    def test_explicit_alpha_overrides_window_default(self) -> None:
        self.assert_values_equal(
            evaluate(ema("close", 3, alpha=0.25)),
            reference_ema_talib(VALUES, 3, 0.25),
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
        values = [1.0, 2.0, None, 4.0, 5.0, 6.0]
        self.assert_values_equal(
            evaluate(ema("close", 3, alpha=0.5), values),
            [None, None, None, None, None, 5.0],
        )

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(ema("close", len(VALUES) + 1)), [None] * len(VALUES))

    def test_invalid_alpha_raises(self) -> None:
        for alpha in (0.0, -0.5, 1.5):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                ema("close", 3, alpha=alpha)

    def test_alpha_of_one_is_allowed(self) -> None:
        self.assert_values_equal(evaluate(ema("close", 1, alpha=1.0)), VALUES)

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            ema("close", 3, mode="exponential")

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            ema("close", 0)


class TestWma(IndicatorAssertions):
    def test_matches_reference_weighted_mean(self) -> None:
        self.assert_values_equal(evaluate(wma("close", 3)), reference_wma(VALUES, 3))

    def test_known_values(self) -> None:
        values = [1.0, 3.0, 2.0, 6.0, 5.0]
        self.assert_values_equal(
            evaluate(wma("close", 3), values),
            [None, None, 13 / 6, 25 / 6, 29 / 6],
        )

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
        values = [1.0, 2.0, None, 4.0, 5.0, 6.0]
        self.assert_values_equal(
            evaluate(wma("close", 3), values), reference_wma(values, 3)
        )

    def test_integer_input_matches_float_input(self) -> None:
        integers = pl.DataFrame({"close": [1, 3, 2, 6, 5]})
        self.assert_values_equal(
            integers.select(wma("close", 3)).to_series().to_list(),
            reference_wma([1.0, 3.0, 2.0, 6.0, 5.0], 3),
        )

    def test_window_longer_than_input_is_all_null(self) -> None:
        self.assertEqual(evaluate(wma("close", len(VALUES) + 1)), [None] * len(VALUES))

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1):
            with self.subTest(window=window), self.assertRaises(ValueError):
                wma("close", window)


class TestDema(IndicatorAssertions):
    def test_matches_double_smoothing_reference(self) -> None:
        self.assert_values_equal(
            evaluate(dema("close", 3), LONG_VALUES),
            reference_dema(LONG_VALUES, 3, 0.5),
        )

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
            evaluate(dema("close", 3, alpha=0.25), LONG_VALUES),
            reference_dema(LONG_VALUES, 3, 0.25),
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
        with self.assertRaises(ValueError):
            dema("close", 0)
        with self.assertRaises(ValueError):
            dema("close", 3, alpha=0.0)
        with self.assertRaises(ValueError):
            dema("close", 3, mode="double")


class TestTema(IndicatorAssertions):
    def test_matches_triple_smoothing_reference(self) -> None:
        self.assert_values_equal(
            evaluate(tema("close", 3), LONG_VALUES),
            reference_tema(LONG_VALUES, 3, 0.5),
        )

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
            evaluate(tema("close", 3, alpha=0.25), LONG_VALUES),
            reference_tema(LONG_VALUES, 3, 0.25),
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
        with self.assertRaises(ValueError):
            tema("close", 0)
        with self.assertRaises(ValueError):
            tema("close", 3, alpha=1.5)
        with self.assertRaises(ValueError):
            tema("close", 3, mode="triple")


class TestInputForms(IndicatorAssertions):
    def test_name_expression_and_series_agree(self) -> None:
        series = pl.Series("close", LONG_VALUES)
        for indicator in (sma, ema, wma, dema, tema):
            with self.subTest(indicator=indicator.__name__):
                from_name = evaluate(indicator("close", 3), LONG_VALUES)
                from_expr = evaluate(indicator(pl.col("close"), 3), LONG_VALUES)
                from_series = indicator(series, 3).to_list()
                self.assert_values_equal(from_expr, from_name)
                self.assert_values_equal(from_series, from_name)

    def test_series_input_returns_series_keeping_its_name(self) -> None:
        result = sma(pl.Series("close", VALUES), 3)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")

    def test_unnamed_series_is_supported(self) -> None:
        result = sma(pl.Series(VALUES), 3)
        self.assertEqual(result.name, "")

    def test_expression_works_in_a_lazy_frame(self) -> None:
        frame = pl.LazyFrame({"close": VALUES}).with_columns(
            sma("close", 3).alias("sma"), ema("close", 3).alias("ema")
        )
        collected = frame.collect()
        self.assertEqual(collected.columns, ["close", "sma", "ema"])
        self.assert_values_equal(
            collected["ema"].to_list(), reference_ema_talib(VALUES, 3, 0.5)
        )

    def test_integer_input_produces_float_output(self) -> None:
        frame = pl.DataFrame({"close": [1, 2, 3, 4]})
        result = frame.select(sma("close", 2), ema("close", 2).alias("ema"))
        self.assertEqual(result["close"].dtype, pl.Float64)
        self.assertEqual(result["ema"].dtype, pl.Float64)

    def test_unsupported_input_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            sma([1.0, 2.0], 2)
