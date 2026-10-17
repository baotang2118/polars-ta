from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, HIGH, LOW, VOLUME, constant, frame, ramp_up
from pytest import mark, raises

from polars_ta import ema, hma, sma, vidya, vwma, wilders, zlema

LENGTH: int = 60
BARS: pl.DataFrame = frame(
    high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH], volume=VOLUME[:LENGTH]
)

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
HMA_9: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, 11.388888888888891,
    11.896296296296297, 12.077777777777778, 11.59259259259259, 10.399999999999999,
    9.807407407407405, 10.01011111111111, 10.968000000000002, 12.625370370370371,
    14.248555555555555, 15.173851851851856, 15.965740740740744,
]
ZLEMA_10: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    11.5, 10.5, 10.227272727272727, 10.487768595041322, 11.246356123215627,
    12.830655009903694, 13.875990462648478, 14.402174014894209, 15.429051466731627,
    15.271042109144059, 15.925398089299684, 17.44259843669974,
]
WILDERS_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    10.642857142857142, 10.525510204081632, 10.559402332361515, 10.614445022907121,
    10.736984664128041, 11.004342902404609, 11.274746980804279, 11.47297933931826,
    11.809909386509812, 12.003487287473396, 12.34823819551101, 12.834792610117367,
]
VWMA_20: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, 11.00616504854369, 11.136285714285714,
    11.3753738317757, 11.546100917431193, 11.918198198198198, 12.540884955752212,
    13.12791304347826, 13.803632478632478, 13.996818181818181, 14.312864077669904,
    14.362047619047619, 14.35070093457944,
]
VIDYA_12_26: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, 18.54,
    18.76326938094709, 18.599972096382604, 18.19303726415124, 17.652407419998323,
    16.950359202699836, 16.527596014646463, 16.259226456947427, 15.830436084738812,
    15.590111645804125, 14.93654381613279, 14.688291955422734,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestHma:
    def test_known_values(self) -> None:
        assert_values_equal(column(hma("close", 9))[: len(HMA_9)], HMA_9)

    @mark.parametrize(("window", "lookback"), ((4, 4), (9, 10), (16, 18)))
    def test_warm_up(self, window: int, lookback: int) -> None:
        result = column(hma("close", window))
        assert result[:lookback] == [None] * lookback
        assert result[lookback] is not None

    def test_tracks_a_straight_line_without_lag(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(30)})
        result = column(hma("close", 9), bars)
        # A weighted average of a straight line sits on the line itself.
        assert_values_equal(result[10:14], [11.0, 12.0, 13.0, 14.0])

    def test_constant_series_is_the_constant(self) -> None:
        bars = pl.DataFrame({"close": constant(25, 4.0)})
        assert_values_equal(column(hma("close", 9), bars)[10:14], [4.0] * 4)

    def test_window_below_two_raises(self) -> None:
        with raises(ValueError):
            hma("close", 1)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            hma(cast(Any, pl.Series("close", CLOSE[:LENGTH])), 9)


class TestZlema:
    def test_known_values(self) -> None:
        assert_values_equal(column(zlema("close", 10))[: len(ZLEMA_10)], ZLEMA_10)

    @mark.parametrize("window", (5, 10, 14))
    def test_warm_up_covers_the_lag_and_the_average(self, window: int) -> None:
        lookback = (window - 1) // 2 + window - 1
        result = column(zlema("close", window))
        assert result[:lookback] == [None] * lookback
        assert result[lookback] is not None

    def test_leads_a_plain_ema_on_a_rising_series(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(60)})
        fast = column(zlema("close", 10), bars)
        slow = column(ema("close", 10), bars)
        assert all(f > s for f, s in zip(fast[30:], slow[30:]))

    def test_invalid_mode_raises(self) -> None:
        with raises(ValueError):
            zlema("close", 10, mode=cast(Any, "nope"))


class TestWilders:
    def test_known_values(self) -> None:
        assert_values_equal(column(wilders("close", 14))[: len(WILDERS_14)], WILDERS_14)

    @mark.parametrize("window", (5, 14, 30))
    def test_matches_an_ema_with_wilders_alpha(self, window: int) -> None:
        assert_values_equal(
            column(wilders("close", window)),
            column(ema("close", window, alpha=1.0 / window)),
        )

    @mark.parametrize("window", (5, 14, 30))
    def test_warm_up_is_window_minus_one(self, window: int) -> None:
        result = column(wilders("close", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            wilders("close", 0)


class TestVwma:
    def test_known_values(self) -> None:
        assert_values_equal(
            column(vwma("close", "volume", 20))[: len(VWMA_20)], VWMA_20
        )

    @mark.parametrize("window", (5, 14, 20))
    def test_warm_up_is_window_minus_one(self, window: int) -> None:
        result = column(vwma("close", "volume", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_flat_volume_reduces_to_a_simple_average(self) -> None:
        bars = pl.DataFrame({"close": CLOSE[:40], "volume": constant(40, 7.0)})
        assert_values_equal(
            column(vwma("close", "volume", 10), bars), column(sma("close", 10), bars)
        )

    def test_window_with_no_volume_is_null(self) -> None:
        bars = pl.DataFrame({"close": constant(6, 3.0), "volume": [0.0] * 6})
        assert column(vwma("close", "volume", 3), bars) == [None] * 6

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            vwma(cast(Any, pl.Series("close", CLOSE[:LENGTH])), "volume", 20)


class TestVidya:
    def test_known_values(self) -> None:
        assert_values_equal(
            column(vidya("close", 12, 26))[: len(VIDYA_12_26)], VIDYA_12_26
        )

    @mark.parametrize(("short", "long"), ((5, 10), (12, 26)))
    def test_warm_up_is_long_period_minus_one(self, short: int, long: int) -> None:
        result = column(vidya("close", short, long))
        assert result[: long - 1] == [None] * (long - 1)
        assert result[long - 1] is not None

    def test_first_live_row_is_the_input(self) -> None:
        result = column(vidya("close", 5, 10))
        assert result[9] == CLOSE[9]

    def test_flat_series_never_adapts(self) -> None:
        bars = pl.DataFrame({"close": constant(30, 6.0)})
        assert_values_equal(column(vidya("close", 5, 10), bars)[9:], [6.0] * 21)

    def test_short_period_above_long_period_raises(self) -> None:
        with raises(ValueError):
            vidya("close", 20, 10)

    def test_invalid_alpha_raises(self) -> None:
        with raises(ValueError):
            vidya("close", 5, 10, alpha=0.0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            vidya(cast(Any, pl.Series("close", CLOSE[:LENGTH])))
