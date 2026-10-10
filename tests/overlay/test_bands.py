from collections.abc import Sequence
from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import HAND_CHECKED, constant, ramp_up
from pytest import approx, raises

from polars_ta import bbands

# The literal expectations below were worked out against this exact series.
VALUES: list[float] = HAND_CHECKED[:8]
NULL_VALUES: list[float | None] = [1.0, 2.0, None, 4.0, 5.0, 6.0]

# fmt: off
BBANDS_3_LOWER: list[float | None] = [
    None, None, 0.36700683814454793, 0.26732032427147656, 0.9339869909381431,
    3.267320324271477, 1.6795062010614261, 2.679506201061426,
]
BBANDS_3_MIDDLE: list[float | None] = [
    None, None, 2.0, 3.6666666666666665, 4.333333333333333, 6.666666666666667, 6.0,
    7.0,
]
BBANDS_3_UPPER: list[float | None] = [
    None, None, 3.632993161855452, 7.066013009061857, 7.732679675728523,
    10.066013009061857, 10.320493798938575, 11.320493798938575,
]
BBANDS_3_SAMPLE_LOWER: list[float | None] = [
    None, None, 0.0, -0.4966653322655996, 0.1700013344010678, 2.5033346677344017,
    0.7084973778708186, 1.7084973778708186,
]
BBANDS_3_SAMPLE_UPPER: list[float | None] = [
    None, None, 4.0, 7.829998665598932, 8.496665332265598, 10.829998665598932,
    11.291502622129181, 12.291502622129181,
]
# fmt: on


def evaluate(
    expr: pl.Expr, values: Sequence[float | None] | None = None
) -> pl.DataFrame:
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr.alias("b")).unnest("b")


class TestBbands:
    def test_known_values(self) -> None:
        frame = evaluate(bbands("close", 3))
        assert_values_equal(frame["lower"].to_list(), BBANDS_3_LOWER)
        assert_values_equal(frame["middle"].to_list(), BBANDS_3_MIDDLE)
        assert_values_equal(frame["upper"].to_list(), BBANDS_3_UPPER)

    def test_field_names_and_order(self) -> None:
        assert evaluate(bbands("close", 3)).columns == ["lower", "middle", "upper"]

    def test_middle_band_is_the_simple_moving_average(self) -> None:
        middle = evaluate(bbands("close", 3))["middle"].to_list()
        assert_values_equal(middle, BBANDS_3_MIDDLE)

    def test_bands_are_symmetric_about_the_middle(self) -> None:
        frame = evaluate(bbands("close", 3))
        for row in frame.drop_nulls().iter_rows(named=True):
            assert row["upper"] - row["middle"] == approx(
                row["middle"] - row["lower"], rel=0, abs=5e-11
            )

    def test_num_std_scales_the_envelope(self) -> None:
        narrow = evaluate(bbands("close", 3, num_std=1.0))
        wide = evaluate(bbands("close", 3, num_std=3.0))
        for index in range(2, len(VALUES)):
            spread = narrow["upper"][index] - narrow["middle"][index]
            assert wide["upper"][index] - wide["middle"][index] == approx(
                3.0 * spread, rel=0, abs=5e-11
            ), f"{index=}"

    def test_ddof_one_is_the_sample_deviation(self) -> None:
        frame = evaluate(bbands("close", 3, ddof=1))
        assert_values_equal(frame["lower"].to_list(), BBANDS_3_SAMPLE_LOWER)
        assert_values_equal(frame["upper"].to_list(), BBANDS_3_SAMPLE_UPPER)

    def test_constant_input_collapses_the_bands(self) -> None:
        frame = evaluate(bbands("close", 3), constant(6, 4.0))
        assert_values_equal(frame["upper"].to_list()[2:], [4.0] * 4)
        assert_values_equal(frame["lower"].to_list()[2:], [4.0] * 4)

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        middle = evaluate(bbands("close", 4))["middle"].to_list()
        assert middle[:3] == [None, None, None]
        assert middle[3] is not None

    def test_null_blanks_every_overlapping_window(self) -> None:
        frame = evaluate(bbands("close", 3), NULL_VALUES)
        assert frame["middle"].to_list()[:5] == [None] * 5
        assert frame["middle"][5] is not None

    def test_default_window_is_twenty(self) -> None:
        values = ramp_up(25, 0.0)
        middle = evaluate(bbands("close"), values)["middle"].to_list()
        assert middle[:19] == [None] * 19

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            bbands(cast(Any, pl.Series("close", VALUES)), 3)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(bbands("close", 3).alias("bb"))
            .collect()
        )
        assert collected.columns == ["close", "bb"]

    def test_invalid_arguments_raise(self) -> None:
        for window in (0, -1, 2.5):
            with raises(ValueError):
                bbands("close", cast(Any, window))
        for num_std in (0.0, -1.0):
            with raises(ValueError):
                bbands("close", 3, num_std=num_std)
        for ddof in (-1, 3, 4):
            with raises(ValueError):
                bbands("close", 3, ddof=ddof)
