import math

import polars as pl
from _assertions import IndicatorAssertions
from _data import HAND_CHECKED, constant, ramp_up

from polars_ta import bbands

# The literal expectations below were worked out against this exact series.
VALUES: list[float] = HAND_CHECKED[:8]


def evaluate(expr: pl.Expr, values: list[float | None] | None = None) -> pl.DataFrame:
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr.alias("b")).unnest("b")


def reference_bbands(values, window: int, num_std: float, ddof: int = 0):
    lower: list[float | None] = []
    middle: list[float | None] = []
    upper: list[float | None] = []
    for index in range(len(values)):
        chunk = values[index + 1 - window : index + 1]
        if index + 1 < window or any(v is None for v in chunk):
            lower.append(None)
            middle.append(None)
            upper.append(None)
            continue
        mean = sum(chunk) / window
        variance = sum((v - mean) ** 2 for v in chunk) / (window - ddof)
        deviation = math.sqrt(variance) * num_std
        lower.append(mean - deviation)
        middle.append(mean)
        upper.append(mean + deviation)
    return lower, middle, upper


class TestBbands(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        frame = evaluate(bbands("close", 3))
        lower, middle, upper = reference_bbands(VALUES, 3, 2.0)
        self.assert_values_equal(frame["lower"].to_list(), lower)
        self.assert_values_equal(frame["middle"].to_list(), middle)
        self.assert_values_equal(frame["upper"].to_list(), upper)

    def test_field_names_and_order(self) -> None:
        self.assertEqual(
            evaluate(bbands("close", 3)).columns, ["lower", "middle", "upper"]
        )

    def test_middle_band_is_the_simple_moving_average(self) -> None:
        middle = evaluate(bbands("close", 3))["middle"].to_list()
        self.assert_values_equal(
            middle, [None, None, 2.0, 11 / 3, 13 / 3, 20 / 3, 6.0, 7.0]
        )

    def test_bands_are_symmetric_about_the_middle(self) -> None:
        frame = evaluate(bbands("close", 3))
        for row in frame.drop_nulls().iter_rows(named=True):
            self.assertAlmostEqual(
                row["upper"] - row["middle"], row["middle"] - row["lower"], places=10
            )

    def test_num_std_scales_the_envelope(self) -> None:
        narrow = evaluate(bbands("close", 3, num_std=1.0))
        wide = evaluate(bbands("close", 3, num_std=3.0))
        for index in range(2, len(VALUES)):
            spread = narrow["upper"][index] - narrow["middle"][index]
            self.assertAlmostEqual(
                wide["upper"][index] - wide["middle"][index], 3.0 * spread, places=10
            )

    def test_ddof_one_is_the_sample_deviation(self) -> None:
        frame = evaluate(bbands("close", 3, ddof=1))
        lower, _, upper = reference_bbands(VALUES, 3, 2.0, ddof=1)
        self.assert_values_equal(frame["lower"].to_list(), lower)
        self.assert_values_equal(frame["upper"].to_list(), upper)

    def test_constant_input_collapses_the_bands(self) -> None:
        frame = evaluate(bbands("close", 3), constant(6, 4.0))
        self.assert_values_equal(frame["upper"].to_list()[2:], [4.0] * 4)
        self.assert_values_equal(frame["lower"].to_list()[2:], [4.0] * 4)

    def test_warm_up_is_window_minus_one_nulls(self) -> None:
        middle = evaluate(bbands("close", 4))["middle"].to_list()
        self.assertEqual(middle[:3], [None, None, None])
        self.assertIsNotNone(middle[3])

    def test_null_blanks_every_overlapping_window(self) -> None:
        values = [1.0, 2.0, None, 4.0, 5.0, 6.0]
        frame = evaluate(bbands("close", 3), values)
        self.assertEqual(frame["middle"].to_list()[:5], [None] * 5)
        self.assertIsNotNone(frame["middle"][5])

    def test_default_window_is_twenty(self) -> None:
        values = ramp_up(25, 0.0)
        self.assertEqual(
            evaluate(bbands("close"), values)["middle"].to_list()[:19], [None] * 19
        )

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            bbands(pl.Series("close", VALUES), 3)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(bbands("close", 3).alias("bb"))
            .collect()
        )
        self.assertEqual(collected.columns, ["close", "bb"])

    def test_invalid_arguments_raise(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                bbands("close", window)
        for num_std in (0.0, -1.0):
            with self.subTest(num_std=num_std), self.assertRaises(ValueError):
                bbands("close", 3, num_std=num_std)
        for ddof in (-1, 3, 4):
            with self.subTest(ddof=ddof), self.assertRaises(ValueError):
                bbands("close", 3, ddof=ddof)
