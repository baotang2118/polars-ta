from collections.abc import Sequence
from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import HAND_CHECKED, WILDER_CLOSE, constant, ramp_down, ramp_up
from pytest import approx, mark, raises

from polars_ta import rsi

VALUES: list[float] = HAND_CHECKED[:10]
NULL_VALUES: list[float | None] = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0]

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
RSI_14_WILDER: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    70.46413502109705, 66.24961855355507, 66.48094183471265, 69.34685316290866,
    66.29471265892624, 57.91502067008556,
]
RSI_3: list[float | None] = [
    None, None, None, 85.71428571428571, 70.58823529411764, 85.71428571428572,
    43.63636363636363, 64.53089244851259, 56.65494726268208, 74.97825456654103,
]
# fmt: on


def evaluate(expr: pl.Expr, values: Sequence[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


class TestRsi:
    def test_matches_wilders_published_value(self) -> None:
        result = evaluate(rsi("close", 14), WILDER_CLOSE)
        assert result[14] == approx(70.4641, rel=0, abs=5e-5)

    def test_known_values_on_wilders_series(self) -> None:
        assert_values_equal(evaluate(rsi("close", 14), WILDER_CLOSE), RSI_14_WILDER)

    def test_known_values_on_a_short_window(self) -> None:
        assert_values_equal(evaluate(rsi("close", 3)), RSI_3)

    @mark.parametrize("window", (2, 3, 5))
    def test_warm_up_is_window_nulls(self, window: int) -> None:
        result = evaluate(rsi("close", window))
        assert result[:window] == [None] * window
        assert result[window] is not None

    def test_output_stays_within_zero_and_one_hundred(self) -> None:
        for value in evaluate(rsi("close", 3)):
            if value is not None:
                assert value >= 0.0
                assert value <= 100.0

    def test_monotonic_rise_reaches_one_hundred(self) -> None:
        rising = ramp_up(10, 0.0)
        assert_values_equal(evaluate(rsi("close", 3), rising)[3:], [100.0] * 7)

    def test_monotonic_fall_reaches_zero(self) -> None:
        falling = ramp_down(10)
        assert_values_equal(evaluate(rsi("close", 3), falling)[3:], [0.0] * 7)

    def test_flat_series_reports_the_neutral_fifty(self) -> None:
        assert_values_equal(evaluate(rsi("close", 3), constant(8))[3:], [50.0] * 5)

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        assert evaluate(rsi("close", len(VALUES))) == [None] * len(VALUES)

    def test_null_delays_the_seed(self) -> None:
        result = evaluate(rsi("close", 2), NULL_VALUES)
        assert result[:5] == [None] * 5
        assert result[5] is not None

    def test_default_window_is_fourteen(self) -> None:
        assert_values_equal(
            evaluate(rsi("close"), WILDER_CLOSE),
            evaluate(rsi("close", 14), WILDER_CLOSE),
        )

    def test_name_and_expression_agree(self) -> None:
        from_name = evaluate(rsi("close", 3))
        assert_values_equal(evaluate(rsi(pl.col("close"), 3)), from_name)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            rsi(cast(Any, pl.Series("close", VALUES)), 3)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(rsi("close", 3).alias("rsi"))
            .collect()
        )
        assert_values_equal(collected["rsi"].to_list(), RSI_3)

    @mark.parametrize("window", (0, -1, 2.5))
    def test_invalid_window_raises(self, window: float) -> None:
        with raises(ValueError):
            rsi("close", cast(Any, window))
