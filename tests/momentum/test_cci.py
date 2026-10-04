import polars as pl
from _assertions import IndicatorAssertions
from _data import (
    CLOSE,
    HIGH,
    LOW,
    constant,
    frame,
    frame_from,
    ramp_down,
    ramp_up,
    with_null,
)

from polars_ta import cci
from polars_ta.momentum.cci import CCI_SCALE


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None):
    return (data if data is not None else frame()).select(expr).to_series().to_list()


def reference_cci(high, low, close, window: int) -> list[float | None]:
    typical = [
        None if None in (h, lo, c) else (h + lo + c) / 3.0
        for h, lo, c in zip(high, low, close)
    ]
    result: list[float | None] = []
    for index in range(len(typical)):
        chunk = typical[index + 1 - window : index + 1]
        if index + 1 < window or any(v is None for v in chunk):
            result.append(None)
            continue
        average = sum(chunk) / window
        deviation = sum(abs(v - average) for v in chunk) / window
        if deviation <= 0.0:
            result.append(0.0)
        else:
            result.append((typical[index] - average) / (CCI_SCALE * deviation))
    return result


class TestCci(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            evaluate(cci("high", "low", "close", 5)), reference_cci(HIGH, LOW, CLOSE, 5)
        )

    def test_known_value(self) -> None:
        # Typical prices 9, 10, 11, 10, 9 give mean 9.8 and deviation 0.64.
        result = evaluate(cci("high", "low", "close", 5))
        self.assertAlmostEqual(result[4], -0.8 / (CCI_SCALE * 0.64), places=10)

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        for window in (2, 5, 14):
            with self.subTest(window=window):
                result = evaluate(cci("high", "low", "close", window))
                self.assertEqual(result[: window - 1], [None] * (window - 1))
                self.assertIsNotNone(result[window - 1])

    def test_typical_price_above_its_mean_is_positive(self) -> None:
        rising = ramp_up(10)
        for value in evaluate(cci("high", "low", "close", 5), frame_from(rising, 0.0))[
            4:
        ]:
            self.assertGreater(value, 0.0)

    def test_typical_price_below_its_mean_is_negative(self) -> None:
        falling = ramp_down(10)
        for value in evaluate(cci("high", "low", "close", 5), frame_from(falling, 0.0))[
            4:
        ]:
            self.assertLess(value, 0.0)

    def test_flat_input_reports_zero(self) -> None:
        data = frame_from(constant(8), 0.0)
        self.assert_values_equal(
            evaluate(cci("high", "low", "close", 3), data)[2:], [0.0] * 6
        )

    def test_scale_constant_is_lamberts(self) -> None:
        self.assertEqual(CCI_SCALE, 0.015)

    def test_null_in_any_input_propagates(self) -> None:
        close = with_null(CLOSE, 6)
        result = evaluate(cci("high", "low", "close", 3), frame(close=close))
        self.assert_values_equal(result, reference_cci(HIGH, LOW, close, 3))
        self.assertEqual(result[6:9], [None, None, None])

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        self.assertEqual(
            evaluate(cci("high", "low", "close", len(HIGH) + 1)), [None] * len(HIGH)
        )

    def test_default_window_is_fourteen(self) -> None:
        self.assert_values_equal(
            evaluate(cci("high", "low", "close")),
            evaluate(cci("high", "low", "close", 14)),
        )

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(cci("high", "low", "close", 5))
        from_exprs = evaluate(cci(pl.col("high"), pl.col("low"), pl.col("close"), 5))
        from_series = cci(
            pl.Series("high", HIGH), pl.Series("low", LOW), pl.Series("close", CLOSE), 5
        )
        self.assert_values_equal(from_exprs, from_names)
        self.assert_values_equal(from_series.to_list(), from_names)
        self.assertEqual(from_series.name, "high")

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            cci(pl.Series("high", HIGH), "low", "close", 5)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(cci("high", "low", "close", 5).alias("cci"))
            .collect()
        )
        self.assert_values_equal(
            collected["cci"].to_list(), reference_cci(HIGH, LOW, CLOSE, 5)
        )

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                cci("high", "low", "close", window)
