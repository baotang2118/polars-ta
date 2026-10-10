from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import (
    CLOSE,
    HIGH,
    constant,
    frame,
    frame_from,
    ramp_down,
    ramp_up,
    with_null,
)
from pytest import approx, mark, raises

from polars_ta import cci
from polars_ta.momentum.cci import CCI_SCALE

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
CCI_5: list[float | None] = [
    None, None, None, None, -83.33333333333343, 0.0, 83.33333333333343,
    121.21212121212118, 30.303030303030326, -83.33333333333343, 83.33333333333343,
    106.06060606060608, -30.303030303030326, -76.92307692307689, -111.11111111111111,
    12.820512820512779,
]
CCI_3_NULL_CLOSE: list[float | None] = [
    None, None, 100.00000000000001, -50.000000000000064, -100.00000000000001,
    50.000000000000064, None, None, None, -100.00000000000001, 100.00000000000001,
    80.00000000000001, -100.00000000000001, -80.00000000000001, -100.00000000000001,
    100.00000000000001,
]
# fmt: on


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None):
    return (data if data is not None else frame()).select(expr).to_series().to_list()


class TestCci:
    def test_known_values(self) -> None:
        result = evaluate(cci("high", "low", "close", 5))
        assert_values_equal(result[: len(CCI_5)], CCI_5)

    def test_known_value(self) -> None:
        # Typical prices 9, 10, 11, 10, 9 give mean 9.8 and deviation 0.64, so the
        # fifth reading is -0.8 / (0.015 * 0.64).
        result = evaluate(cci("high", "low", "close", 5))
        assert result[4] == approx(-83.33333333333334, rel=0, abs=5e-11)

    @mark.parametrize("window", (2, 5, 14))
    def test_warm_up_is_window_minus_one_nulls(self, window: int) -> None:
        result = evaluate(cci("high", "low", "close", window))
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    def test_typical_price_above_its_mean_is_positive(self) -> None:
        rising = ramp_up(10)
        for value in evaluate(cci("high", "low", "close", 5), frame_from(rising, 0.0))[
            4:
        ]:
            assert value > 0.0

    def test_typical_price_below_its_mean_is_negative(self) -> None:
        falling = ramp_down(10)
        for value in evaluate(cci("high", "low", "close", 5), frame_from(falling, 0.0))[
            4:
        ]:
            assert value < 0.0

    def test_flat_input_reports_zero(self) -> None:
        data = frame_from(constant(8), 0.0)
        assert_values_equal(
            evaluate(cci("high", "low", "close", 3), data)[2:], [0.0] * 6
        )

    def test_scale_constant_is_lamberts(self) -> None:
        assert CCI_SCALE == 0.015

    def test_null_in_any_input_propagates(self) -> None:
        close = with_null(CLOSE, 6)
        result = evaluate(cci("high", "low", "close", 3), frame(close=close))
        assert_values_equal(result[: len(CCI_3_NULL_CLOSE)], CCI_3_NULL_CLOSE)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        assert evaluate(cci("high", "low", "close", len(HIGH) + 1)) == [None] * len(
            HIGH
        )

    def test_default_window_is_fourteen(self) -> None:
        assert_values_equal(
            evaluate(cci("high", "low", "close")),
            evaluate(cci("high", "low", "close", 14)),
        )

    def test_names_and_expressions_agree(self) -> None:
        from_names = evaluate(cci("high", "low", "close", 5))
        from_exprs = evaluate(cci(pl.col("high"), pl.col("low"), pl.col("close"), 5))
        assert_values_equal(from_exprs, from_names)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            cci(cast(Any, pl.Series("high", HIGH)), "low", "close", 5)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(cci("high", "low", "close", 5).alias("cci"))
            .collect()
        )
        assert_values_equal(collected["cci"].to_list()[: len(CCI_5)], CCI_5)

    @mark.parametrize("window", (0, -1, 2.5))
    def test_invalid_window_raises(self, window: float) -> None:
        with raises(ValueError):
            cci("high", "low", "close", cast(Any, window))
