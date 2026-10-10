from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import frame
from pytest import raises

from polars_ta._common import to_exprs
from polars_ta.pattern._candles import (
    BODY_DOJI,
    BODY_LONG,
    EQUAL,
    FAR,
    NEAR,
    PATTERN_DTYPE,
    SHADOW_LONG,
    SHADOW_SHORT,
    Bars,
    CandleSetting,
    RangeType,
    signal,
)

# Four bars: white, black, a doji, and a bar with no range at all.
OPEN: list[float] = [10.0, 20.0, 30.0, 40.0]
HIGH: list[float] = [14.0, 24.0, 31.0, 40.0]
LOW: list[float] = [9.0, 17.0, 29.0, 40.0]
CLOSE: list[float] = [13.0, 18.0, 30.0, 40.0]
BARS: pl.DataFrame = pl.DataFrame(
    {"open": OPEN, "high": HIGH, "low": LOW, "close": CLOSE}
)


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr.alias("out")).to_series().to_list()


def candles() -> Bars:
    return Bars(*to_exprs("open", "high", "low", "close"))


class TestSettings:
    def test_the_defaults_match_ta_lib(self) -> None:
        assert BODY_LONG == CandleSetting(RangeType.REAL_BODY, 10, 1.0)
        assert BODY_DOJI == CandleSetting(RangeType.HIGH_LOW, 10, 0.1)
        assert SHADOW_LONG == CandleSetting(RangeType.REAL_BODY, 0, 1.0)
        assert SHADOW_SHORT == CandleSetting(RangeType.SHADOWS, 10, 1.0)
        assert NEAR == CandleSetting(RangeType.HIGH_LOW, 5, 0.2)
        assert FAR == CandleSetting(RangeType.HIGH_LOW, 5, 0.6)
        assert EQUAL == CandleSetting(RangeType.HIGH_LOW, 5, 0.05)

    def test_a_setting_cannot_be_mutated(self) -> None:
        with raises(Exception):
            cast(Any, BODY_LONG).factor = 2.0


class TestGeometry:
    def test_body_is_the_distance_between_open_and_close(self) -> None:
        assert_values_equal(column(candles().body(0)), [3.0, 2.0, 0.0, 0.0])

    def test_shadows_are_measured_from_the_body(self) -> None:
        assert_values_equal(column(candles().upper_shadow(0)), [1.0, 4.0, 1.0, 0.0])
        assert_values_equal(column(candles().lower_shadow(0)), [1.0, 1.0, 1.0, 0.0])

    def test_colour_treats_an_unchanged_close_as_white(self) -> None:
        assert column(candles().white(0)) == [True, False, True, True]
        assert column(candles().black(0)) == [False, True, False, False]
        assert column(candles().color(0)) == [1, -1, 1, 1]

    def test_an_offset_reads_an_earlier_bar(self) -> None:
        assert_values_equal(column(candles().close(1)), [None, 13.0, 18.0, 30.0])

    def test_gaps_compare_whole_ranges_and_bodies_separately(self) -> None:
        bars = pl.DataFrame(
            {
                "open": [10.0, 20.0],
                "high": [15.0, 25.0],
                "low": [9.0, 14.0],
                "close": [14.0, 24.0],
            }
        )
        # The second bar's body clears the first, but its range overlaps it.
        assert column(candles().body_gap_up(0, 1), bars)[1] is True
        assert column(candles().gap_up(0, 1), bars)[1] is False


class TestAverage:
    def test_it_covers_the_bars_before_the_reference_bar(self) -> None:
        setting = CandleSetting(RangeType.REAL_BODY, 2, 1.0)
        bars = pl.DataFrame(
            {
                "open": [0.0, 0.0, 0.0, 0.0],
                "high": [9.0, 9.0, 9.0, 9.0],
                "low": [0.0, 0.0, 0.0, 0.0],
                "close": [2.0, 4.0, 90.0, 0.0],
            }
        )
        result = column(candles().average(setting, 0), bars)
        # The outsized third body never reaches the average at its own bar.
        assert_values_equal(result, [None, None, 3.0, 47.0])

    def test_a_zero_period_setting_measures_the_reference_bar_itself(self) -> None:
        assert_values_equal(
            column(candles().average(SHADOW_LONG, 0)), [3.0, 2.0, 0.0, 0.0]
        )

    def test_the_factor_scales_the_average(self) -> None:
        doubled = CandleSetting(RangeType.HIGH_LOW, 2, 2.0)
        single = CandleSetting(RangeType.HIGH_LOW, 2, 1.0)
        assert_values_equal(
            column(candles().average(doubled, 0))[2:],
            [value * 2.0 for value in column(candles().average(single, 0))[2:]],
        )

    def test_a_shadows_setting_is_halved(self) -> None:
        shadows = CandleSetting(RangeType.SHADOWS, 2, 1.0)
        # Shadows sum to 2, 5, 2 a bar, so each pair averages 3.5 before halving.
        assert_values_equal(column(candles().average(shadows, 0))[2:], [1.75, 1.75])


class TestSignal:
    def test_it_scores_matches_and_leaves_the_rest_at_zero(self) -> None:
        bars = candles()
        result = column(signal(bars, 1, bars.white(0), -100))
        assert result == [None, 0, -100, -100]

    def test_the_warm_up_is_as_long_as_the_lookback(self) -> None:
        bars = candles()
        assert column(signal(bars, 3, bars.white(0))) == [None, None, None, 100]

    def test_the_result_is_an_integer_column(self) -> None:
        bars = candles()
        series = BARS.select(signal(bars, 1, bars.white(0)).alias("out"))["out"]
        assert series.dtype == PATTERN_DTYPE

    def test_a_null_input_blanks_the_window_it_falls_in(self) -> None:
        bars = frame().with_columns(
            pl.when(pl.int_range(pl.len()) == 20)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        candle = candles()
        result = column(signal(candle, 2, candle.white(0)), bars)
        assert result[19] is not None
        assert result[20:23] == [None, None, None]
        assert result[23] is not None
