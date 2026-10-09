from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, constant, frame, ramp_up
from pytest import approx, raises

from polars_ta import cumulative_return, daily_log_return, daily_return

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])
STEPS = pl.DataFrame({"close": [10.0, 12.0, 6.0, 6.0]})


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestDailyReturn:
    def test_matches_hand_checked_values(self) -> None:
        assert_values_equal(
            column(daily_return("close"), STEPS), [None, 20.0, -50.0, 0.0]
        )

    def test_first_row_is_null(self) -> None:
        assert column(daily_return("close"))[0] is None

    def test_a_flat_series_reads_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(5, 7.0)})
        assert_values_equal(column(daily_return("close"), bars), [None] + [0.0] * 4)

    def test_a_zero_reference_price_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": [0.0, 5.0]})
        assert_values_equal(column(daily_return("close"), bars), [None, 0.0])

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            daily_return(cast(Any, pl.Series("close", CLOSE[:LENGTH])))


class TestDailyLogReturn:
    def test_matches_hand_checked_values(self) -> None:
        # 100 * ln(1.2) and 100 * ln(0.5).
        assert_values_equal(
            column(daily_log_return("close"), STEPS),
            [None, 18.232155679395458, -69.31471805599453, 0.0],
        )

    def test_sums_to_the_whole_period_return(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(10)})
        total = sum(value for value in column(daily_log_return("close"), bars) if value)
        # The series runs from 1 to 10, so the total is 100 * ln(10).
        assert total == approx(230.25850929940458, rel=0, abs=5e-8)

    def test_first_row_is_null(self) -> None:
        assert column(daily_log_return("close"))[0] is None


class TestCumulativeReturn:
    def test_matches_hand_checked_values(self) -> None:
        assert_values_equal(
            column(cumulative_return("close"), STEPS), [0.0, 20.0, -40.0, -40.0]
        )

    def test_has_no_warm_up(self) -> None:
        assert column(cumulative_return("close"))[0] == 0.0

    def test_leading_nulls_do_not_set_the_base(self) -> None:
        bars = pl.DataFrame({"close": [None, 10.0, 15.0]})
        assert_values_equal(column(cumulative_return("close"), bars), [None, 0.0, 50.0])

    def test_a_zero_base_price_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": [0.0, 5.0]})
        assert_values_equal(column(cumulative_return("close"), bars), [0.0, 0.0])
