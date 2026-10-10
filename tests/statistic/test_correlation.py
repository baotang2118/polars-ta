from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import HIGH, LOW, constant, ramp_up
from pytest import mark, raises

from polars_ta import beta, correl

LENGTH: int = 28
BARS: pl.DataFrame = pl.DataFrame({"high": HIGH[:LENGTH], "low": LOW[:LENGTH]})

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
CORREL_5: list[float | None] = [
    None, None, None, None, 0.8481804781738445, 0.7980149866097266,
    0.8577619564342264, 0.9223609647563313, 0.9223609647563313, 0.857761956434246,
    0.8577619564342066, 0.9302033851800108, 0.9302033851800027, 0.9601750892864609,
    0.9470939054947987, 0.9184979117189042,
]
BETA_5: list[float | None] = [
    None, None, None, None, None, 1.0320444903041506, 1.0186262774931125,
    1.0091292384982704, 2.2022172038059007, 1.017998930136406, 0.9714946830903439,
    0.9629573261134974, 1.0649766971583288, 0.9486410110183319, 0.923249003166106,
    0.8849709541601735,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr.alias("out")).to_series().to_list()


class TestCorrel:
    def test_known_values(self) -> None:
        assert_values_equal(column(correl("high", "low", 5))[: len(CORREL_5)], CORREL_5)

    @mark.parametrize("window", (2, 5, 14))
    def test_warm_up_is_window_minus_one_nulls(self, window: int) -> None:
        result = column(correl("high", "low", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_a_series_correlates_perfectly_with_itself(self) -> None:
        result = column(correl("high", "high", 5))
        assert_values_equal(result[4:], [1.0] * (LENGTH - 4))

    def test_a_mirrored_series_correlates_perfectly_in_reverse(self) -> None:
        bars = pl.DataFrame({"a": ramp_up(10), "b": [-value for value in ramp_up(10)]})
        assert_values_equal(column(correl("a", "b", 5), bars)[4:], [-1.0] * 6)

    def test_a_motionless_series_reports_zero(self) -> None:
        bars = pl.DataFrame({"a": ramp_up(8), "b": constant(8)})
        assert_values_equal(column(correl("a", "b", 4), bars)[3:], [0.0] * 5)

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            correl("high", "low", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            correl(cast(Any, pl.Series("high", HIGH[:LENGTH])), "low")


class TestBeta:
    def test_known_values(self) -> None:
        assert_values_equal(column(beta("high", "low", 5))[: len(BETA_5)], BETA_5)

    @mark.parametrize("window", (2, 5, 14))
    def test_warm_up_consumes_a_bar_for_the_first_return(self, window: int) -> None:
        result = column(beta("high", "low", window))
        assert result[:window] == [None] * window
        assert result[window] is not None

    def test_a_series_against_itself_is_one(self) -> None:
        result = column(beta("high", "high", 5))
        assert_values_equal(result[5:], [1.0] * (LENGTH - 5))

    def test_doubled_returns_double_the_beta(self) -> None:
        moves = [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.005, 0.015, -0.005]
        market = [100.0]
        asset = [100.0]
        for move in moves:
            market.append(market[-1] * (1.0 + move))
            asset.append(asset[-1] * (1.0 + 2.0 * move))
        bars = pl.DataFrame({"asset": asset, "market": market})
        result = column(beta("asset", "market", 4), bars)
        assert_values_equal(result[4:], [2.0] * 6)

    def test_a_motionless_reference_reports_zero(self) -> None:
        bars = pl.DataFrame({"asset": ramp_up(8), "market": constant(8)})
        assert_values_equal(column(beta("asset", "market", 4), bars)[4:], [0.0] * 4)

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            beta("high", "low", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            beta("high", cast(Any, pl.Series("low", LOW[:LENGTH])))
