from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, constant, ramp_up
from pytest import mark, raises

from polars_ta import stddev, var, zscore

LENGTH: int = 28
BARS: pl.DataFrame = pl.DataFrame({"close": CLOSE[:LENGTH]})

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
STDDEV_5: list[float | None] = [
    None, None, None, None, 0.7483314773547708, 0.6324555320336803,
    0.7483314773547898, 1.019803902718553, 1.0198039027185601, 0.7483314773547803,
    0.7483314773547993, 1.019803902718553, 1.019803902718553, 1.166190378969066,
    1.4142135623730951, 1.3266499161421565,
]
VAR_5: list[float | None] = [
    None, None, None, None, 0.5599999999999739, 0.4000000000000057,
    0.5600000000000023, 1.039999999999992, 1.0400000000000063, 0.5599999999999881,
    0.5600000000000165, 1.039999999999992, 1.039999999999992, 1.3600000000000136,
    2.0, 1.759999999999991,
]
# Worked out with Python's statistics.pstdev and statistics.stdev.
ZSCORE_5_POPULATION: list[float | None] = [
    None, None, None, None, -1.0690449676496985, 0.0,
    1.0690449676496985, 1.568929081105472, 0.39223227027636837, -1.0690449676496985,
    1.0690449676496985, 1.3728129459672884, -0.39223227027636837, -1.0289915108550522,
    -1.414213562373095, 0.1507556722888813,
]
ZSCORE_5_SAMPLE: list[float | None] = [
    None, None, None, None, -0.9561828874675158, 0.0,
    0.9561828874675158, 1.4032928308912462, 0.350823207722812, -0.9561828874675158,
    0.9561828874675158, 1.2278812270298411, -0.350823207722812, -0.920357986616844,
    -1.2649110640673518, 0.13483997249264792,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr.alias("out")).to_series().to_list()


class TestVar:
    def test_known_values(self) -> None:
        assert_values_equal(column(var("close", 5))[: len(VAR_5)], VAR_5)

    @mark.parametrize("window", (2, 5, 14))
    def test_warm_up_is_window_minus_one_nulls(self, window: int) -> None:
        result = column(var("close", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_a_flat_series_has_no_spread(self) -> None:
        bars = pl.DataFrame({"close": constant(8, 7.0)})
        assert_values_equal(column(var("close", 3), bars), [None, None] + [0.0] * 6)

    def test_a_straight_ramp_has_a_known_spread(self) -> None:
        # Deviations from the mean of three consecutive integers are -1, 0, 1.
        bars = pl.DataFrame({"close": ramp_up(6)})
        result = column(var("close", 3), bars)
        assert_values_equal(result[2:], [2.0 / 3.0] * 4)

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            var("close", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            var(cast(Any, pl.Series("close", CLOSE[:LENGTH])))


class TestStddev:
    def test_known_values(self) -> None:
        assert_values_equal(column(stddev("close", 5))[: len(STDDEV_5)], STDDEV_5)

    def test_it_is_the_square_root_of_the_variance(self) -> None:
        result = column(stddev("close", 5))
        expected = [None if v is None else v**0.5 for v in column(var("close", 5))]
        assert_values_equal(result, expected)

    @mark.parametrize("nbdev", (0.5, 2.0, 3.0))
    def test_nbdev_scales_the_result(self, nbdev: float) -> None:
        scaled = column(stddev("close", 5, nbdev))
        single = column(stddev("close", 5))
        assert_values_equal(scaled, [None if v is None else v * nbdev for v in single])

    def test_a_flat_series_has_no_spread(self) -> None:
        bars = pl.DataFrame({"close": constant(8, 7.0)})
        assert_values_equal(column(stddev("close", 3), bars), [None, None] + [0.0] * 6)

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            stddev("close", 0)

    @mark.parametrize("nbdev", (0.0, -1.0))
    def test_invalid_nbdev_raises(self, nbdev: float) -> None:
        with raises(ValueError):
            stddev("close", 5, nbdev)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            stddev(cast(Any, pl.Series("close", CLOSE[:LENGTH])))


class TestZscore:
    def test_known_population_values(self) -> None:
        result = column(zscore("close", 5))
        assert_values_equal(result[: len(ZSCORE_5_POPULATION)], ZSCORE_5_POPULATION)

    def test_known_sample_values(self) -> None:
        result = column(zscore("close", 5, ddof=1))
        assert_values_equal(result[: len(ZSCORE_5_SAMPLE)], ZSCORE_5_SAMPLE)

    @mark.parametrize("window", (2, 5, 14))
    def test_warm_up_is_window_minus_one_nulls(self, window: int) -> None:
        result = column(zscore("close", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_it_is_the_deviation_over_the_standard_deviation(self) -> None:
        result = column(zscore("close", 5))
        deviation = column(
            pl.col("close") - pl.col("close").rolling_mean(5, min_samples=5)
        )
        expected = [
            None if d is None or s is None or s == 0.0 else d / s
            for d, s in zip(deviation, column(stddev("close", 5)))
        ]
        assert_values_equal(result, expected)

    def test_the_sample_score_is_the_population_one_scaled_down(self) -> None:
        window = 5
        factor = ((window - 1) / window) ** 0.5
        sample = column(zscore("close", window, ddof=1))
        population = column(zscore("close", window))
        assert_values_equal(
            sample, [None if v is None else v * factor for v in population]
        )

    @mark.parametrize("ddof", (0, 1))
    def test_a_flat_window_has_no_score(self, ddof: int) -> None:
        bars = pl.DataFrame({"close": constant(8, 7.0)})
        assert_values_equal(column(zscore("close", 3, ddof), bars), [None] * 8)

    def test_a_straight_ramp_has_a_known_score(self) -> None:
        # The last of three consecutive integers sits one above their mean.
        bars = pl.DataFrame({"close": ramp_up(6)})
        result = column(zscore("close", 3), bars)
        assert_values_equal(result[2:], [(3.0 / 2.0) ** 0.5] * 4)

    @mark.parametrize("ddof", (0, 1))
    def test_a_single_bar_window_has_no_score(self, ddof: int) -> None:
        assert_values_equal(column(zscore("close", 1, ddof)), [None] * LENGTH)

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            zscore("close", 0)

    @mark.parametrize("ddof", (-1, 2, True))
    def test_invalid_ddof_raises(self, ddof: Any) -> None:
        with raises(ValueError):
            zscore("close", 5, ddof)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            zscore(cast(Any, pl.Series("close", CLOSE[:LENGTH])))
