from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, HIGH, LOW, OPEN, constant, frame, ramp_up
from pytest import mark, raises

from polars_ta import fisher, fosc, qstick, tsf, vhf

LENGTH: int = 60
BARS: pl.DataFrame = frame(
    high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH], open_=OPEN[:LENGTH]
)

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
FISHER_9: list[float | None] = [
    None, None, None, None, None, None, None, None, 0.11044691579009712,
    0.01890750122858357, 0.3252258012966205, 0.7594637744066214, 0.7546881796202807,
    0.28717443827940325, -0.2685082436955386, -0.4019374039642517,
    -0.3222939464024788, -0.021691312442017635, 0.4402926704786063,
    0.9343385861888611,
]
FOSC_14: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, -30.15873015873015, 0.3996003996004143, 3.3975733004859205,
    9.398144434640816, 20.605154513994293, 17.625511743158786, 7.802588870204532,
    16.425143725946665, -1.8905337087155472, 9.28613869790337, 15.92947762050057,
    4.310252853942205,
]
QSTICK_10: list[float | None] = [
    None, None, None, None, None, None, None, None, None, -0.0875, -0.0375, 0.05,
    -0.075, 0.0125, 0.0875, -0.0625, 0.0125, 0.075, -0.05, 0.0125, 0.0625,
]
VHF_28: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, 0.29131728428455506, 0.28597981943706846, 0.2847699629825489,
    0.28201099764336207, 0.28507146638433034, 0.283794466403162, 0.28537360890302066,
    0.27580025608194625, 0.2727272727272727, 0.26566354218056243, 0.2742551566080978,
    0.2727963525835866,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def field(expr: pl.Expr, name: str, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr.alias("value")).unnest("value")[name].to_list()


class TestFisher:
    def test_known_values(self) -> None:
        result = field(fisher("high", "low", 9), "fisher")
        assert_values_equal(result[: len(FISHER_9)], FISHER_9)

    def test_signal_is_the_previous_reading(self) -> None:
        line = field(fisher("high", "low", 9), "fisher")
        signal = field(fisher("high", "low", 9), "signal")
        assert signal[8] is None
        assert_values_equal(signal[9:], line[8:-1])

    @mark.parametrize("window", (5, 9, 14))
    def test_warm_up_is_window_minus_one(self, window: int) -> None:
        line = field(fisher("high", "low", window), "fisher")
        assert line[: window - 1] == [None] * (window - 1)
        assert line[window - 1] is not None

    def test_flat_range_reads_zero(self) -> None:
        flat = constant(30, 7.0)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in flat],
                "low": [value - 1.0 for value in flat],
            }
        )
        assert_values_equal(
            field(fisher("high", "low", 5), "fisher", bars)[4:], [0.0] * 26
        )

    def test_rising_series_is_pushed_to_an_extreme(self) -> None:
        rising = ramp_up(40)
        bars = pl.DataFrame(
            {"high": [v + 1.0 for v in rising], "low": [v - 1.0 for v in rising]}
        )
        result = field(fisher("high", "low", 9), "fisher", bars)
        assert result[-1] > 3.0

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            fisher("high", "low", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            fisher(cast(Any, pl.Series("high", HIGH[:LENGTH])), "low", 9)


class TestFosc:
    def test_known_values(self) -> None:
        assert_values_equal(column(fosc("close", 14))[: len(FOSC_14)], FOSC_14)

    @mark.parametrize("window", (5, 14, 20))
    def test_warm_up_is_the_window(self, window: int) -> None:
        result = column(fosc("close", window))
        assert result[:window] == [None] * window
        assert result[window] is not None

    def test_straight_line_is_forecast_exactly(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(30)})
        assert_values_equal(column(fosc("close", 10), bars)[10:], [0.0] * 20)

    def test_compares_against_the_previous_forecast(self) -> None:
        forecast = column(tsf("close", 14))
        expected = [
            None
            if index == 0 or forecast[index - 1] is None
            else (CLOSE[index] - forecast[index - 1]) / CLOSE[index] * 100.0
            for index in range(LENGTH)
        ]
        assert_values_equal(column(fosc("close", 14)), expected)

    def test_window_below_two_raises(self) -> None:
        with raises(ValueError):
            fosc("close", 1)


class TestQstick:
    def test_known_values(self) -> None:
        result = column(qstick("open", "close", 10))
        assert_values_equal(result[: len(QSTICK_10)], QSTICK_10)

    @mark.parametrize("window", (5, 10, 20))
    def test_warm_up_is_window_minus_one(self, window: int) -> None:
        result = column(qstick("open", "close", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_all_green_candles_read_positive(self) -> None:
        bars = pl.DataFrame({"open": constant(20, 5.0), "close": constant(20, 6.0)})
        assert_values_equal(column(qstick("open", "close", 5), bars)[4:], [1.0] * 16)

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            qstick("open", "close", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            qstick(cast(Any, pl.Series("open", OPEN[:LENGTH])), "close", 10)


class TestVhf:
    def test_known_values(self) -> None:
        assert_values_equal(column(vhf("close", 28))[: len(VHF_28)], VHF_28)

    @mark.parametrize("window", (5, 14, 28))
    def test_warm_up_is_the_window(self, window: int) -> None:
        result = column(vhf("close", window))
        assert result[:window] == [None] * window
        assert result[window] is not None

    def test_straight_line_reads_its_maximum(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(30)})
        # The window spans window - 1 steps but accumulates window of them, so a
        # perfect trend tops out just below one.
        assert_values_equal(column(vhf("close", 10), bars)[10:], [0.9] * 20)

    def test_motionless_series_reads_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(30, 4.0)})
        assert_values_equal(column(vhf("close", 10), bars)[10:], [0.0] * 20)

    def test_stays_within_the_unit_interval(self) -> None:
        for value in column(vhf("close", 14)):
            if value is not None:
                assert 0.0 <= value <= 1.0

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            vhf("close", 0)
