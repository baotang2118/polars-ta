from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, HIGH, LOW, constant, frame
from pytest import mark, raises

from polars_ta import annualized_volatility, cvi

LENGTH: int = 60
BARS: pl.DataFrame = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
CVI_10: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, 8.321359831508705, -0.05144281469068817,
    -6.404802870437297, 1.189373406539954, 7.810589982337282, -0.4239342696221433,
    -6.673309361373407, 0.9348880582410907, 7.583481270736201, -0.589818190364179,
    -6.793051028283513, 0.8213251203959762,
]
HV_20: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, 1.7760386688332128, 1.805619962404439,
    1.8437703466720168, 1.8561983402203732, 1.823899704103263, 1.8258822609683292,
    1.8151149658544337, 1.8639350370731715, 1.8946610618110347, 1.9087753155898273,
    1.8291833847249086, 1.8431107191306828,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestCvi:
    def test_known_values(self) -> None:
        assert_values_equal(column(cvi("high", "low", 10))[: len(CVI_10)], CVI_10)

    @mark.parametrize("window", (3, 5, 10))
    def test_warm_up_is_twice_the_window_less_one(self, window: int) -> None:
        lookback = 2 * window - 1
        result = column(cvi("high", "low", window))
        assert result[:lookback] == [None] * lookback
        assert result[lookback] is not None

    def test_unchanging_range_reads_zero(self) -> None:
        flat = constant(40, 5.0)
        bars = pl.DataFrame({"high": [value + 1.0 for value in flat], "low": flat})
        assert_values_equal(column(cvi("high", "low", 5), bars)[9:], [0.0] * 31)

    def test_widening_range_reads_positive(self) -> None:
        middle = constant(40, 20.0)
        spread = [1.0 + index * 0.1 for index in range(40)]
        bars = pl.DataFrame(
            {
                "high": [m + s for m, s in zip(middle, spread)],
                "low": [m - s for m, s in zip(middle, spread)],
            }
        )
        assert all(value > 0.0 for value in column(cvi("high", "low", 5), bars)[9:])

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            cvi("high", "low", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            cvi(cast(Any, pl.Series("high", HIGH[:LENGTH])), "low", 10)


class TestAnnualizedVolatility:
    def test_known_values(self) -> None:
        result = column(annualized_volatility("close", 20))
        assert_values_equal(result[: len(HV_20)], HV_20)

    @mark.parametrize("window", (5, 14, 20))
    def test_warm_up_is_the_window(self, window: int) -> None:
        result = column(annualized_volatility("close", window))
        assert result[:window] == [None] * window
        assert result[window] is not None

    def test_constant_series_has_no_volatility(self) -> None:
        bars = pl.DataFrame({"close": constant(30, 8.0)})
        assert_values_equal(
            column(annualized_volatility("close", 10), bars)[10:], [0.0] * 20
        )

    @mark.parametrize("periods_per_year", (52.0, 252.0))
    def test_scales_with_the_square_root_of_the_annualisation(
        self, periods_per_year: float
    ) -> None:
        scaled = column(
            annualized_volatility("close", 10, periods_per_year=periods_per_year)
        )
        base = column(annualized_volatility("close", 10, periods_per_year=1.0))
        factor = periods_per_year**0.5
        assert_values_equal(
            scaled[10:20], [None if v is None else v * factor for v in base[10:20]]
        )

    def test_invalid_annualisation_raises(self) -> None:
        with raises(ValueError):
            annualized_volatility("close", 20, periods_per_year=0.0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            annualized_volatility(cast(Any, pl.Series("close", CLOSE[:LENGTH])))
