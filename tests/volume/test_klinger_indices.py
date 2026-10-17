from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, HIGH, LOW, VOLUME, constant, frame, ramp_down, ramp_up
from pytest import mark, raises

from polars_ta import kvo, marketfi, nvi, pvi, wad

LENGTH: int = 60
BARS: pl.DataFrame = frame(
    high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH], volume=VOLUME[:LENGTH]
)

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
KVO_5_10: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, -4975.865704867215,
    1979.6786409038268, -9505.53603517054, -17974.75287563603, -24371.682272738435,
    -1569.2654764126055, 20170.982618522055, 33355.02533254292, 19851.0541796194,
    13771.33502904349, -3189.4191374936636, 562.475892948496,
]
MARKETFI: list[float | None] = [
    0.013333333333333334, 0.01, 0.008571428571428572, 0.0077777777777777776,
    0.0036363636363636364, 0.0038461538461538464, 0.004, 0.00411764705882353,
    0.002105263157894737, 0.016666666666666666, 0.012, 0.01,
]
PVI: list[float | None] = [
    1000.0, 1111.111111111111, 1222.2222222222224, 1111.111111111111, 1000.0,
    1111.111111111111, 1222.2222222222224, 1333.3333333333333, 1222.2222222222222,
    1222.2222222222222, 1466.6666666666665, 1588.8888888888887,
]
WAD: list[float | None] = [
    0.0, 1.25, 2.75, 1.0, 0.0, 1.25, 2.75, 4.5, 3.5, 2.25, 4.25, 6.0,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestKvo:
    def test_known_values(self) -> None:
        result = column(kvo("high", "low", "close", "volume", 5, 10))
        assert_values_equal(result[: len(KVO_5_10)], KVO_5_10)

    @mark.parametrize(("fast", "slow"), ((5, 10), (8, 13)))
    def test_warm_up_is_the_slow_period(self, fast: int, slow: int) -> None:
        result = column(kvo("high", "low", "close", "volume", fast, slow))
        assert result[:slow] == [None] * slow
        assert result[slow] is not None

    def test_steady_trend_settles_towards_zero(self) -> None:
        rising = ramp_up(80)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in rising],
                "low": [value - 1.0 for value in rising],
                "close": rising,
                "volume": constant(80, 100.0),
            }
        )
        result = column(kvo("high", "low", "close", "volume", 5, 10), bars)
        assert abs(result[-1]) < abs(result[12])

    def test_invalid_period_raises(self) -> None:
        with raises(ValueError):
            kvo("high", "low", "close", "volume", 0, 55)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            kvo(cast(Any, pl.Series("high", HIGH[:LENGTH])), "low", "close", "volume")


class TestMarketfi:
    def test_known_values(self) -> None:
        result = column(marketfi("high", "low", "volume"))
        assert_values_equal(result[: len(MARKETFI)], MARKETFI)

    def test_has_no_warm_up(self) -> None:
        assert column(marketfi("high", "low", "volume"))[0] is not None

    def test_wider_range_on_the_same_volume_reads_higher(self) -> None:
        bars = pl.DataFrame(
            {"high": [2.0, 4.0], "low": [1.0, 1.0], "volume": [100.0, 100.0]}
        )
        result = column(marketfi("high", "low", "volume"), bars)
        assert result[1] > result[0]

    def test_bar_with_no_volume_reports_zero(self) -> None:
        bars = pl.DataFrame({"high": [2.0], "low": [1.0], "volume": [0.0]})
        assert column(marketfi("high", "low", "volume"), bars) == [0.0]

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            marketfi("high", "low", cast(Any, pl.Series("volume", VOLUME[:LENGTH])))


class TestPvi:
    def test_known_values(self) -> None:
        assert_values_equal(column(pvi("close", "volume"))[: len(PVI)], PVI)

    def test_first_bar_is_the_start_value(self) -> None:
        assert column(pvi("close", "volume"))[0] == 1000.0

    def test_start_value_scales_the_index(self) -> None:
        base = column(pvi("close", "volume"))
        scaled = column(pvi("close", "volume", start_value=500.0))
        assert_values_equal(scaled, [value / 2.0 for value in base])

    def test_falling_volume_holds_the_level(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(10), "volume": ramp_down(10, 100.0)})
        assert_values_equal(column(pvi("close", "volume"), bars), [1000.0] * 10)

    def test_complements_the_negative_index(self) -> None:
        rising = ramp_up(10)
        bars = pl.DataFrame({"close": rising, "volume": ramp_up(10, 100.0)})
        assert column(pvi("close", "volume"), bars)[-1] > 1000.0
        assert_values_equal(column(nvi("close", "volume"), bars), [1000.0] * 10)

    def test_invalid_start_value_raises(self) -> None:
        with raises(ValueError):
            pvi("close", "volume", start_value=0.0)


class TestWad:
    def test_known_values(self) -> None:
        assert_values_equal(column(wad("high", "low", "close"))[: len(WAD)], WAD)

    def test_has_no_warm_up(self) -> None:
        assert column(wad("high", "low", "close"))[0] == 0.0

    def test_up_bar_banks_the_move_from_the_lower_of_low_and_prior_close(self) -> None:
        bars = pl.DataFrame(
            {"high": [11.0, 13.0], "low": [9.0, 11.5], "close": [10.0, 12.0]}
        )
        assert_values_equal(column(wad("high", "low", "close"), bars), [0.0, 2.0])

    def test_down_bar_banks_the_move_from_the_higher_of_high_and_prior_close(
        self,
    ) -> None:
        # The prior close of 10.0 sits above this bar's high, so the gap counts.
        bars = pl.DataFrame(
            {"high": [11.0, 9.5], "low": [9.0, 7.0], "close": [10.0, 8.0]}
        )
        assert_values_equal(column(wad("high", "low", "close"), bars), [0.0, -2.0])

    def test_unchanged_close_contributes_nothing(self) -> None:
        flat = constant(5, 3.0)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in flat],
                "low": [value - 1.0 for value in flat],
                "close": flat,
            }
        )
        assert_values_equal(column(wad("high", "low", "close"), bars), [0.0] * 5)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            wad("high", "low", cast(Any, pl.Series("close", CLOSE[:LENGTH])))
