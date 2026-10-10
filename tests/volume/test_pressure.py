from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, VOLUME, constant, frame, ramp_up
from pytest import mark, raises

from polars_ta import eom, fi, nvi, vpt

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestFi:
    def test_unsmoothed_is_the_change_times_volume(self) -> None:
        bars = pl.DataFrame(
            {"close": [10.0, 12.0, 11.0, 11.0], "volume": [100.0, 50.0, 20.0, 80.0]}
        )
        assert_values_equal(
            column(fi("close", "volume", 1), bars), [None, 100.0, -20.0, 0.0]
        )

    @mark.parametrize("window", (2, 5, 13))
    def test_warm_up_is_the_window(self, window: int) -> None:
        result = column(fi("close", "volume", window))
        assert result[:window] == [None] * window
        assert result[window] is not None

    def test_a_flat_close_reads_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(8, 4.0), "volume": [70.0] * 8})
        assert_values_equal(
            column(fi("close", "volume", 3), bars), [None, None, None] + [0.0] * 5
        )

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            fi("close", "volume", 0)


class TestEom:
    def test_matches_hand_checked_values(self) -> None:
        bars = pl.DataFrame(
            {
                "high": [10.0, 11.0, 13.0],
                "low": [8.0, 9.0, 11.0],
                "volume": [100.0, 100.0, 200.0],
            }
        )
        # Bar 1: the midpoint moves 1.0 over a span of 2.0 on 100 of volume.
        assert_values_equal(
            column(eom("high", "low", "volume", 1), bars),
            [None, 2_000_000.0, 2_000_000.0],
        )

    def test_a_bar_without_volume_reports_zero(self) -> None:
        bars = pl.DataFrame(
            {"high": [10.0, 11.0], "low": [8.0, 9.0], "volume": [100.0, 0.0]}
        )
        assert_values_equal(column(eom("high", "low", "volume", 1), bars), [None, 0.0])

    @mark.parametrize("window", (1, 5, 14))
    def test_warm_up_is_the_window(self, window: int) -> None:
        result = column(eom("high", "low", "volume", window))
        assert result[:window] == [None] * window
        assert result[window] is not None

    def test_a_flat_bar_reads_zero(self) -> None:
        flat = constant(6, 5.0)
        bars = pl.DataFrame({"high": flat, "low": flat, "volume": [40.0] * 6})
        assert_values_equal(
            column(eom("high", "low", "volume", 2), bars), [None, None] + [0.0] * 4
        )

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            eom("high", "low", "volume", 0)


class TestVpt:
    def test_matches_hand_checked_values(self) -> None:
        bars = pl.DataFrame(
            {"close": [10.0, 11.0, 11.0, 5.5], "volume": [100.0, 200.0, 50.0, 400.0]}
        )
        # Returns of +10%, 0%, and -50% on 200, 50, and 400 of volume.
        assert_values_equal(
            column(vpt("close", "volume"), bars), [0.0, 20.0, 20.0, -180.0]
        )

    def test_has_no_warm_up(self) -> None:
        assert column(vpt("close", "volume"))[0] == 0.0

    def test_a_flat_close_never_moves(self) -> None:
        bars = pl.DataFrame({"close": constant(5, 3.0), "volume": [90.0] * 5})
        assert_values_equal(column(vpt("close", "volume"), bars), [0.0] * 5)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            vpt(
                cast(Any, pl.Series("close", CLOSE[:LENGTH])),
                cast(Any, pl.Series("volume", VOLUME[:LENGTH])),
            )


class TestNvi:
    def test_compounds_only_when_volume_falls(self) -> None:
        bars = pl.DataFrame(
            {
                "close": [10.0, 12.0, 15.0, 7.5],
                "volume": [100.0, 50.0, 80.0, 40.0],
            }
        )
        # Volume falls on bars 1 and 3, so only +20% and -50% are taken.
        assert_values_equal(
            column(nvi("close", "volume"), bars), [1000.0, 1200.0, 1200.0, 600.0]
        )

    def test_starts_at_the_given_level(self) -> None:
        assert column(nvi("close", "volume", start_value=100.0))[0] == 100.0

    def test_rising_volume_holds_the_index_flat(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(5), "volume": ramp_up(5, 10.0)})
        assert_values_equal(column(nvi("close", "volume"), bars), [1000.0] * 5)

    def test_invalid_start_value_raises(self) -> None:
        with raises(ValueError):
            nvi("close", "volume", start_value=0.0)
