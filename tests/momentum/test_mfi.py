import polars as pl
from _assertions import IndicatorAssertions
from _data import (
    CLOSE,
    HIGH,
    LOW,
    VOLUME,
    constant,
    frame,
    frame_from,
    ramp_down,
    ramp_up,
    with_null,
)

from polars_ta import mfi


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None):
    return (data if data is not None else frame()).select(expr).to_series().to_list()


def reference_mfi(high, low, close, volume, window: int) -> list[float | None]:
    typical = [
        None if None in (h, lo, c) else (h + lo + c) / 3.0
        for h, lo, c in zip(high, low, close)
    ]
    flow = [None if t is None or v is None else t * v for t, v in zip(typical, volume)]
    positive: list[float | None] = [None]
    negative: list[float | None] = [None]
    for index in range(1, len(typical)):
        previous, current = typical[index - 1], typical[index]
        if previous is None or current is None or flow[index] is None:
            positive.append(None)
            negative.append(None)
            continue
        positive.append(flow[index] if current > previous else 0.0)
        negative.append(flow[index] if current < previous else 0.0)
    result: list[float | None] = []
    for index in range(len(typical)):
        chunk_positive = positive[index + 1 - window : index + 1]
        chunk_negative = negative[index + 1 - window : index + 1]
        if index + 1 < window or any(v is None for v in chunk_positive):
            result.append(None)
            continue
        total = sum(chunk_positive) + sum(chunk_negative)
        result.append(0.0 if total <= 0.0 else 100.0 * sum(chunk_positive) / total)
    return result


class TestMfi(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            evaluate(mfi("high", "low", "close", "volume", 3)),
            reference_mfi(HIGH, LOW, CLOSE, VOLUME, 3),
        )

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
        self.assert_values_equal(result, reference_mfi(HIGH, LOW, close, VOLUME, 3))
        self.assertEqual(result[4:7], [None, None, None])

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

    def test_names_expressions_and_series_agree(self) -> None:
        from_names = evaluate(mfi("high", "low", "close", "volume", 3))
        from_exprs = evaluate(
            mfi(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 3)
        )
        from_series = mfi(
            pl.Series("high", HIGH),
            pl.Series("low", LOW),
            pl.Series("close", CLOSE),
            pl.Series("volume", VOLUME),
            3,
        )
        self.assert_values_equal(from_exprs, from_names)
        self.assert_values_equal(from_series.to_list(), from_names)
        self.assertEqual(from_series.name, "high")

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            mfi(pl.Series("high", HIGH), "low", "close", "volume", 3)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(mfi("high", "low", "close", "volume", 3).alias("mfi"))
            .collect()
        )
        self.assert_values_equal(
            collected["mfi"].to_list(), reference_mfi(HIGH, LOW, CLOSE, VOLUME, 3)
        )

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                mfi("high", "low", "close", "volume", window)
