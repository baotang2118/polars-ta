from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, HIGH, frame
from pytest import mark, raises

from polars_ta import vwap

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
VWAP_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    10.753731343283581, 10.631944444444445, 10.688311688311689, 10.741524390243903,
    10.953333333333333, 11.146506024096386, 11.356139240506328, 11.517466666666667,
    11.755845070422534, 12.089925373134328, 12.561388888888889, 13.222337662337662,
]
# With a constant volume the result reduces to a mean of the typical price.
VWAP_10_FLAT_VOLUME: list[float | None] = [
    None, None, None, None, None, None, None, None, None, 10.3, 10.6, 10.9, 10.9, 10.9,
    10.9, 11.0, 11.033, 11.065999999999999, 11.414, 11.892999999999999, 12.098,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestVwap:
    def test_known_values(self) -> None:
        result = column(vwap("high", "low", "close", "volume", 14))
        assert_values_equal(result[: len(VWAP_14)], VWAP_14)

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
        assert_values_equal(
            column(vwap("high", "low", "close", "volume", 2), bars), [None, 5.0]
        )

    def test_constant_volume_reduces_to_the_typical_price_average(self) -> None:
        bars = BARS.with_columns(pl.lit(100.0).alias("volume"))
        result = column(vwap("high", "low", "close", "volume", 10), bars)
        assert_values_equal(result[: len(VWAP_10_FLAT_VOLUME)], VWAP_10_FLAT_VOLUME)

    @mark.parametrize("window", (3, 14, 30))
    def test_warm_up_is_window_minus_one(self, window: int) -> None:
        result = column(vwap("high", "low", "close", "volume", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_window_without_volume_is_null(self) -> None:
        bars = pl.DataFrame(
            {
                "high": [4.0, 5.0, 6.0],
                "low": [2.0, 3.0, 4.0],
                "close": [3.0, 4.0, 5.0],
                "volume": [0.0, 0.0, 0.0],
            }
        )
        assert_values_equal(
            column(vwap("high", "low", "close", "volume", 2), bars), [None, None, None]
        )

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            vwap("high", "low", "close", "volume", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            vwap(cast(Any, pl.Series("high", HIGH[:5])), "low", "close", "volume")
