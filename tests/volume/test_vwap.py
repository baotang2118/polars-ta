import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, VOLUME, frame

from polars_ta import vwap

LENGTH = 60
BARS = frame(close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def reference_vwap(
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float],
    window: int,
) -> list:
    weighted = [
        (high[index] + low[index] + close[index]) / 3.0 * volume[index]
        for index in range(len(close))
    ]
    result: list = [None] * (window - 1)
    for index in range(window - 1, len(close)):
        traded = sum(volume[index - window + 1 : index + 1])
        result.append(
            None
            if traded == 0.0
            else sum(weighted[index - window + 1 : index + 1]) / traded
        )
    return result


class TestVwap(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        self.assert_values_equal(
            column(vwap("high", "low", "close", "volume", 14)),
            reference_vwap(
                HIGH[:LENGTH], LOW[:LENGTH], CLOSE[:LENGTH], VOLUME[:LENGTH], 14
            ),
        )

    def test_matches_hand_checked_values(self) -> None:
        bars = pl.DataFrame(
            {
                "high": [4.0, 8.0],
                "low": [0.0, 4.0],
                "close": [2.0, 6.0],
                "volume": [100.0, 300.0],
            }
        )
        # Typical prices of 2 and 6, weighted 1:3.
        self.assert_values_equal(
            column(vwap("high", "low", "close", "volume", 2), bars), [None, 5.0]
        )

    def test_constant_volume_reduces_to_the_typical_price_average(self) -> None:
        bars = BARS.with_columns(pl.lit(100.0).alias("volume"))
        expected = bars.select(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).rolling_mean(
                window_size=10, min_samples=10
            )
        )
        self.assert_values_equal(
            column(vwap("high", "low", "close", "volume", 10), bars),
            expected.to_series().to_list(),
        )

    def test_warm_up_is_window_minus_one(self) -> None:
        for window in (3, 14, 30):
            with self.subTest(window=window):
                result = column(vwap("high", "low", "close", "volume", window))
                self.assertEqual(result[: window - 1], [None] * (window - 1))
                self.assertIsNotNone(result[window - 1])

    def test_window_without_volume_is_null(self) -> None:
        bars = pl.DataFrame(
            {
                "high": [4.0, 5.0, 6.0],
                "low": [2.0, 3.0, 4.0],
                "close": [3.0, 4.0, 5.0],
                "volume": [0.0, 0.0, 0.0],
            }
        )
        self.assert_values_equal(
            column(vwap("high", "low", "close", "volume", 2), bars), [None, None, None]
        )

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            vwap("high", "low", "close", "volume", 0)

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            vwap(pl.Series("high", HIGH[:5]), "low", "close", "volume")
