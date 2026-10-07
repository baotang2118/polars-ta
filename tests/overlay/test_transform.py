import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, OPEN, frame

from polars_ta import avgprice, medprice, typprice, wclprice

BARS = pl.DataFrame(
    {
        "open": [1.0, 2.0, 3.0],
        "high": [4.0, 6.0, 8.0],
        "low": [0.0, 2.0, 4.0],
        "close": [2.0, 4.0, 6.0],
    }
)


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestAvgprice(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        self.assert_values_equal(
            column(avgprice("open", "high", "low", "close")), [1.75, 3.5, 5.25]
        )

    def test_a_null_input_nulls_the_row(self) -> None:
        bars = BARS.with_columns(pl.Series("high", [4.0, None, 8.0]))
        self.assert_values_equal(
            column(avgprice("open", "high", "low", "close"), bars),
            [1.75, None, 5.25],
        )

    def test_series_input_keeps_its_name(self) -> None:
        result = avgprice(
            pl.Series("open", OPEN[:5]),
            pl.Series("high", HIGH[:5]),
            pl.Series("low", LOW[:5]),
            pl.Series("close", CLOSE[:5]),
        )
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "open")

    def test_mixing_series_with_names_raises(self) -> None:
        with self.assertRaises(TypeError):
            avgprice(pl.Series("open", OPEN[:5]), "high", "low", "close")


class TestMedprice(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        self.assert_values_equal(column(medprice("high", "low")), [2.0, 4.0, 6.0])

    def test_sits_inside_the_bar(self) -> None:
        bars = frame()
        for value, high, low in zip(
            column(medprice("high", "low"), bars), bars["high"], bars["low"]
        ):
            self.assertLessEqual(low, value)
            self.assertLessEqual(value, high)


class TestTypprice(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        self.assert_values_equal(
            column(typprice("high", "low", "close")), [2.0, 4.0, 6.0]
        )

    def test_a_null_input_nulls_the_row(self) -> None:
        bars = BARS.with_columns(pl.Series("close", [2.0, 4.0, None]))
        self.assert_values_equal(
            column(typprice("high", "low", "close"), bars), [2.0, 4.0, None]
        )


class TestWclprice(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        self.assert_values_equal(
            column(wclprice("high", "low", "close")), [2.0, 4.0, 6.0]
        )

    def test_leans_towards_the_close(self) -> None:
        bars = pl.DataFrame({"high": [10.0], "low": [0.0], "close": [8.0]})
        self.assert_values_equal(column(wclprice("high", "low", "close"), bars), [6.5])
