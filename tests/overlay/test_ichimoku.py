from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, HIGH, LOW, constant, frame, frame_from, with_null
from pytest import approx, mark, raises

from polars_ta import ichimoku

# Frozen expectations for ichimoku(3, 6, 12, 6): warm-up plus the first live bars.
# fmt: off
CONVERSION_3: list[float | None] = [
    None, None, 10.25, 10.375, 10.25, 9.875, 10.25, 11.25, 11.625, 11.25, 11.125,
    11.75, 12.375, 11.75,
]
BASE_6: list[float | None] = [
    None, None, None, None, None, 10.25, 10.25, 10.875, 10.875, 10.875, 11.25, 11.75,
    11.75, 11.75, 11.125, 11.125, 11.125,
]
SPAN_A: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, 10.0625, 10.25,
    11.0625, 11.25, 11.0625, 11.1875, 11.75, 12.0625, 11.75, 10.4375, 10.625, 10.625,
]
SPAN_B: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, 11.375, 11.375, 11.375, 11.125, 11.125, 11.125, 11.125, 11.74,
    12.02, 12.02, 12.47, 12.47,
]
LAGGING: list[float | None] = [
    11.0, 12.0, 11.0, 10.0, 12.0, 13.0, 11.0, 10.0, 9.0, 11.0, 11.33, 12.33, 14.48,
    14.79, 14.05, 16.19,
]
CONVERSION_3_NULL_HIGH: list[float | None] = [
    None, None, 10.25, 10.375, 10.25, 9.875, 10.25, 11.25, 11.625, 11.25, None, None,
    None, 11.75, 9.75, 10.125,
]
# fmt: on


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None) -> pl.DataFrame:
    source = data if data is not None else frame()
    return source.select(expr.alias("i")).unnest("i")


class TestIchimoku:
    def test_field_names_and_order(self) -> None:
        columns = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6)).columns
        assert columns == ["conversion", "base", "span_a", "span_b", "lagging"]

    def test_conversion_and_base_are_rolling_midpoints(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        assert_values_equal(
            result["conversion"].to_list()[: len(CONVERSION_3)], CONVERSION_3
        )
        assert_values_equal(result["base"].to_list()[: len(BASE_6)], BASE_6)

    def test_span_a_is_the_displaced_average_of_the_two_lines(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        assert_values_equal(result["span_a"].to_list()[: len(SPAN_A)], SPAN_A)

    def test_span_b_is_the_displaced_long_midpoint(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        assert_values_equal(result["span_b"].to_list()[: len(SPAN_B)], SPAN_B)

    def test_lagging_span_is_the_close_pulled_backward(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        assert_values_equal(result["lagging"].to_list()[: len(LAGGING)], LAGGING)

    def test_lagging_span_tail_is_null(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        assert result["lagging"].to_list()[-6:] == [None] * 6

    def test_each_field_carries_its_own_warm_up(self) -> None:
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))
        assert result["conversion"].to_list()[:2] == [None, None]
        assert result["conversion"][2] is not None
        assert result["base"].to_list()[:5] == [None] * 5
        assert result["base"][5] is not None
        assert result["span_b"].to_list()[: 11 + 6] == [None] * 17

    def test_defaults_are_the_classic_nine_twentysix_fiftytwo(self) -> None:
        result = evaluate(ichimoku("high", "low", "close"))
        assert result["conversion"].to_list()[:8] == [None] * 8
        assert result["conversion"][8] is not None
        assert result["lagging"].to_list()[-26:] == [None] * 26

    def test_constant_input_gives_constant_lines(self) -> None:
        data = frame_from(constant(20), 0.0)
        result = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6), data)
        for value in result["conversion"].to_list()[2:]:
            assert value == approx(5.0, rel=0, abs=5e-11)

    def test_null_blanks_every_overlapping_window(self) -> None:
        high = with_null(HIGH, 10)
        result = evaluate(
            ichimoku("high", "low", "close", 3, 6, 12, 6), frame(high=high)
        )
        assert_values_equal(
            result["conversion"].to_list()[: len(CONVERSION_3_NULL_HIGH)],
            CONVERSION_3_NULL_HIGH,
        )

    def test_names_and_expressions_agree(self) -> None:
        from_names = evaluate(ichimoku("high", "low", "close", 3, 6, 12, 6))[
            "conversion"
        ].to_list()
        from_exprs = evaluate(
            ichimoku(pl.col("high"), pl.col("low"), pl.col("close"), 3, 6, 12, 6)
        )["conversion"].to_list()
        assert_values_equal(from_exprs, from_names)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            ichimoku(
                cast(Any, pl.Series("high", HIGH)),
                cast(Any, pl.Series("low", LOW)),
                cast(Any, pl.Series("close", CLOSE)),
                3,
                6,
                12,
                6,
            )

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(ichimoku("high", "low", "close", 3, 6, 12, 6).alias("i"))
            .collect()
        )
        assert collected.columns[-1] == "i"

    @mark.parametrize(
        "periods",
        (
            (0, 26, 52, 26),
            (9, -1, 52, 26),
            (9, 26, 0, 26),
            (9, 26, 52, 0),
        ),
    )
    def test_invalid_periods_raise(self, periods: tuple[int, int, int, int]) -> None:
        with raises(ValueError):
            ichimoku("high", "low", "close", *periods)
