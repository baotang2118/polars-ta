from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import (
    CLOSE,
    HIGH,
    VOLUME,
    constant,
    frame,
    frame_from,
    ramp_down,
    ramp_up,
    with_null,
)

from polars_ta import mfi

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
MFI_3: list[float | None] = [
    None, None, None, 58.525345622119815, 28.94736842105263, 40.75235109717868,
    74.8730964467005, 100.0, 63.84083044982699, 46.04966139954853, 20.066889632107024,
    83.42541436464089, 60.4, 30.333333333333332, 0.0,
]
# A null close at index 4 blanks every window that overlaps it.
MFI_3_NULL_CLOSE: list[float | None] = [
    None, None, None, 58.525345622119815, None, None, None, None, 63.84083044982699,
    46.04966139954853, 20.066889632107024, 83.42541436464089, 60.4, 30.333333333333332,
    0.0, 42.09183673469388,
]
# fmt: on


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None):
    return (data if data is not None else frame()).select(expr).to_series().to_list()


class TestMfi(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = evaluate(mfi("high", "low", "close", "volume", 3))
        self.assert_values_equal(result[: len(MFI_3)], MFI_3)

    def test_warm_up_is_window_nulls(self) -> None:
        for window in (2, 3, 5):
            with self.subTest(window=window):
                result = evaluate(mfi("high", "low", "close", "volume", window))
                self.assertEqual(result[:window], [None] * window)
                self.assertIsNotNone(result[window])

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        for value in evaluate(mfi("high", "low", "close", "volume", 3)):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)

    def test_monotonic_rise_reaches_one_hundred(self) -> None:
        rising = ramp_up(8)
        data = frame_from(rising, 0.0)
        self.assert_values_equal(
            evaluate(mfi("high", "low", "close", "volume", 3), data)[3:], [100.0] * 5
        )

    def test_monotonic_fall_reaches_zero(self) -> None:
        falling = ramp_down(8, 10.0)
        data = frame_from(falling, 0.0)
        self.assert_values_equal(
            evaluate(mfi("high", "low", "close", "volume", 3), data)[3:], [0.0] * 5
        )

    def test_flat_typical_price_reports_zero(self) -> None:
        data = frame_from(constant(6, 1.5), 0.5)
        self.assert_values_equal(
            evaluate(mfi("high", "low", "close", "volume", 3), data)[3:], [0.0] * 3
        )

    def test_zero_volume_reports_zero(self) -> None:
        data = frame(volume=constant(len(HIGH), 0.0))
        result = evaluate(mfi("high", "low", "close", "volume", 3), data)[3:]
        self.assert_values_equal(result, [0.0] * len(result))

    def test_volume_weights_the_flow(self) -> None:
        heavy = evaluate(mfi("high", "low", "close", "volume", 3))
        flat_volume = evaluate(
            mfi("high", "low", "close", "volume", 3),
            frame(volume=constant(len(HIGH), 1.0)),
        )
        self.assertNotEqual(heavy[5], flat_volume[5])

    def test_null_in_any_input_propagates(self) -> None:
        close = with_null(CLOSE, 4)
        data = frame(close=close)
        result = evaluate(mfi("high", "low", "close", "volume", 3), data)
        self.assert_values_equal(result[: len(MFI_3_NULL_CLOSE)], MFI_3_NULL_CLOSE)

    def test_null_volume_propagates(self) -> None:
        volume = with_null(VOLUME, 3)
        result = evaluate(
            mfi("high", "low", "close", "volume", 3), frame(volume=volume)
        )
        self.assertEqual(result[3:6], [None, None, None])

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        self.assertEqual(
            evaluate(mfi("high", "low", "close", "volume", len(HIGH))),
            [None] * len(HIGH),
        )

    def test_default_window_is_fourteen(self) -> None:
        self.assert_values_equal(
            evaluate(mfi("high", "low", "close", "volume")),
            evaluate(mfi("high", "low", "close", "volume", 14)),
        )

    def test_names_and_expressions_agree(self) -> None:
        from_names = evaluate(mfi("high", "low", "close", "volume", 3))
        from_exprs = evaluate(
            mfi(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 3)
        )
        self.assert_values_equal(from_exprs, from_names)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            mfi(cast(Any, pl.Series("high", HIGH)), "low", "close", "volume", 3)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(mfi("high", "low", "close", "volume", 3).alias("mfi"))
            .collect()
        )
        self.assert_values_equal(collected["mfi"].to_list()[: len(MFI_3)], MFI_3)

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                mfi("high", "low", "close", "volume", cast(Any, window))
