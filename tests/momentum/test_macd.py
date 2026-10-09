from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, ramp_up
from pytest import raises

from polars_ta import ema, macd

VALUES: list[float] = CLOSE

# Frozen expectations for macd(close, 3, 6, 4): warm-up plus the first live bars.
# fmt: off
MACD_LINE: list[float | None] = [
    None, None, None, None, None, None, None, None, 0.31483843537414913,
    -0.00949040330417894, 0.30460506906844387, 0.5875527279060311, 0.17609793064716506,
    -0.21029277275202496, -0.532533342144303, -0.14297163947807334,
    0.08729634568530464, 0.37134971957432406, 0.8804616788812147, 1.0029357102220722,
]
MACD_SIGNAL: list[float | None] = [
    None, None, None, None, None, None, None, None, 0.23421556122448894,
    0.13673317541302177, 0.20388193287519063, 0.3573502508875268, 0.2848493227913821,
    0.08679248457401925, -0.16093784611330966, -0.15375136345921514,
    -0.05733227980140722, 0.11414051994888529, 0.4206689835218171, 0.6535756742019191,
]
MACD_HISTOGRAM: list[float | None] = [
    None, None, None, None, None, None, None, None, 0.08062287414966018,
    -0.1462235787172007, 0.10072313619325324, 0.23020247701850427,
    -0.10875139214421703, -0.2970852573260442, -0.3715954960309934,
    0.010779723981141798, 0.14462862548671185, 0.25720919962543876,
    0.45979269535939765, 0.3493600360201531,
]
# fmt: on


def evaluate(expr: pl.Expr, values: list[float] | None = None) -> pl.DataFrame:
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr.alias("m")).unnest("m")


class TestMacd(IndicatorAssertions):
    def test_known_values(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        size = len(MACD_LINE)
        self.assert_values_equal(result["macd"].to_list()[:size], MACD_LINE)
        self.assert_values_equal(result["signal"].to_list()[:size], MACD_SIGNAL)
        self.assert_values_equal(result["histogram"].to_list()[:size], MACD_HISTOGRAM)

    def test_field_names_and_order(self) -> None:
        self.assertEqual(
            evaluate(macd("close", 3, 6, 4)).columns, ["macd", "signal", "histogram"]
        )

    def test_warm_up_matches_the_talib_lookback(self) -> None:
        for fast, slow, signal in ((3, 6, 4), (12, 26, 9), (2, 5, 3)):
            with self.subTest(periods=(fast, slow, signal)):
                result = evaluate(macd("close", fast, slow, signal))
                lookback = (slow - 1) + (signal - 1)
                self.assertEqual(result["macd"].to_list()[:lookback], [None] * lookback)
                self.assertIsNotNone(result["macd"][lookback])

    def test_all_three_fields_start_on_the_same_row(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        self.assertEqual(result["macd"].null_count(), result["signal"].null_count())
        self.assertEqual(result["macd"].null_count(), result["histogram"].null_count())

    def test_histogram_is_macd_minus_signal(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        for row in result.drop_nulls().iter_rows(named=True):
            self.assertAlmostEqual(
                row["histogram"], row["macd"] - row["signal"], places=10
            )

    def test_macd_line_is_the_gap_between_the_two_emas(self) -> None:
        result = evaluate(macd("close", 3, 6, 4))
        gap = (
            pl.DataFrame({"close": VALUES})
            .select((ema("close", 3) - ema("close", 6)).alias("gap"))
            .to_series()
            .to_list()
        )
        for index, value in enumerate(result["macd"].to_list()):
            if value is not None:
                self.assertAlmostEqual(value, gap[index], places=10)

    def test_rising_input_gives_a_positive_macd_line(self) -> None:
        rising = ramp_up(40, 0.0)
        for value in evaluate(macd("close", 3, 6, 4), rising)["macd"].to_list():
            if value is not None:
                self.assertGreater(value, 0.0)

    def test_constant_input_gives_a_zero_macd_line(self) -> None:
        result = evaluate(macd("close", 3, 6, 4), [7.0] * 30)
        for value in result["macd"].to_list():
            if value is not None:
                self.assertAlmostEqual(value, 0.0, places=10)

    def test_every_mode_is_supported(self) -> None:
        for mode in ("talib", "adjust", "recursive"):
            with self.subTest(mode=mode):
                result = evaluate(macd("close", 3, 6, 4, mode=mode))
                self.assertIsNotNone(result["macd"][-1])

    def test_default_periods_are_twelve_twenty_six_and_nine(self) -> None:
        result = evaluate(macd("close"))
        self.assertEqual(result["macd"].null_count(), 25 + 8)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        result = evaluate(macd("close", 3, 6, 4), VALUES[:5])
        self.assertEqual(result["macd"].to_list(), [None] * 5)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            macd(cast(Any, pl.Series("close", VALUES)), 3, 6, 4)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(macd("close", 3, 6, 4).alias("m"))
            .collect()
        )
        self.assertEqual(collected.columns, ["close", "m"])

    def test_invalid_arguments_raise(self) -> None:
        for periods in ((0, 26, 9), (12, -1, 9), (12, 26, 0), (12, 26, 2.5)):
            with self.subTest(periods=periods), raises(ValueError):
                macd("close", *cast(Any, periods))
        with raises(ValueError):
            macd("close", 12, 26, 9, mode=cast(Any, "convergence"))
