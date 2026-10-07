import math

import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, frame, ramp_up

from polars_ta import cumulative_return, daily_log_return, daily_return

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])
STEPS = pl.DataFrame({"close": [10.0, 12.0, 6.0, 6.0]})


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestDailyReturn(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        self.assert_values_equal(
            column(daily_return("close"), STEPS), [None, 20.0, -50.0, 0.0]
        )

    def test_first_row_is_null(self) -> None:
        self.assertIsNone(column(daily_return("close"))[0])

    def test_a_flat_series_reads_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(5, 7.0)})
        self.assert_values_equal(
            column(daily_return("close"), bars), [None] + [0.0] * 4
        )

    def test_a_zero_reference_price_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": [0.0, 5.0]})
        self.assert_values_equal(column(daily_return("close"), bars), [None, 0.0])

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            daily_return(pl.Series("close", CLOSE[:LENGTH]))


class TestDailyLogReturn(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        self.assert_values_equal(
            column(daily_log_return("close"), STEPS),
            [None, 100.0 * math.log(1.2), 100.0 * math.log(0.5), 0.0],
        )

    def test_sums_to_the_whole_period_return(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(10)})
        total = sum(value for value in column(daily_log_return("close"), bars) if value)
        self.assertAlmostEqual(total, 100.0 * math.log(10.0 / 1.0))

    def test_first_row_is_null(self) -> None:
        self.assertIsNone(column(daily_log_return("close"))[0])


class TestCumulativeReturn(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        self.assert_values_equal(
            column(cumulative_return("close"), STEPS), [0.0, 20.0, -40.0, -40.0]
        )

    def test_has_no_warm_up(self) -> None:
        self.assertEqual(column(cumulative_return("close"))[0], 0.0)

    def test_leading_nulls_do_not_set_the_base(self) -> None:
        bars = pl.DataFrame({"close": [None, 10.0, 15.0]})
        self.assert_values_equal(
            column(cumulative_return("close"), bars), [None, 0.0, 50.0]
        )

    def test_a_zero_base_price_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": [0.0, 5.0]})
        self.assert_values_equal(column(cumulative_return("close"), bars), [0.0, 0.0])
