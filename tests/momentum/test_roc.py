from collections.abc import Sequence
from typing import Any, cast

import polars as pl
from _assertions import IndicatorAssertions
from _data import HAND_CHECKED, constant, with_null

from polars_ta import mom, roc, rocp, rocr, rocr100

VALUES: list[float] = HAND_CHECKED

# The literal expectations below were worked out against this exact series.
# fmt: off
MOM: dict[int, list[float | None]] = {
    1: [
        None, 2.0, -1.0, 4.0, -1.0, 4.0, -5.0, 4.0, -1.0, 4.0, -5.0, 4.0, -1.0, 4.0,
        -5.0, 4.0,
    ],
    3: [
        None, None, None, 5.0, 2.0, 7.0, -2.0, 3.0, -2.0, 7.0, -2.0, 3.0, -2.0, 7.0,
        -2.0, 3.0,
    ],
    10: [
        None, None, None, None, None, None, None, None, None, None, 5.0, 7.0, 7.0, 7.0,
        3.0, 3.0,
    ],
}
ROC: dict[int, list[float | None]] = {
    1: [
        None, 200.0, -33.333333333333336, 200.0, -16.666666666666664, 80.0,
        -55.55555555555556, 100.0, -12.5, 57.14285714285714, -45.45454545454546,
        66.66666666666667, -9.999999999999998, 44.44444444444444, -38.46153846153846,
        50.0,
    ],
    3: [
        None, None, None, 500.0, 66.66666666666667, 350.0, -33.333333333333336,
        60.00000000000001, -22.22222222222222, 175.0, -25.0, 42.85714285714286,
        -18.181818181818176, 116.66666666666666, -19.999999999999996,
        33.33333333333333,
    ],
    10: [
        None, None, None, None, None, None, None, None, None, None, 500.0,
        233.33333333333334, 350.0, 116.66666666666666, 60.00000000000001,
        33.33333333333333,
    ],
}
ROCP: dict[int, list[float | None]] = {
    1: [
        None, 2.0, -0.3333333333333333, 2.0, -0.16666666666666666, 0.8,
        -0.5555555555555556, 1.0, -0.125, 0.5714285714285714, -0.45454545454545453,
        0.6666666666666666, -0.1, 0.4444444444444444, -0.38461538461538464, 0.5,
    ],
    3: [
        None, None, None, 5.0, 0.6666666666666666, 3.5, -0.3333333333333333, 0.6,
        -0.2222222222222222, 1.75, -0.25, 0.42857142857142855, -0.18181818181818182,
        1.1666666666666667, -0.2, 0.3333333333333333,
    ],
    10: [
        None, None, None, None, None, None, None, None, None, None, 5.0,
        2.3333333333333335, 3.5, 1.1666666666666667, 0.6, 0.3333333333333333,
    ],
}
ROCR: dict[int, list[float | None]] = {
    1: [
        None, 3.0, 0.6666666666666666, 3.0, 0.8333333333333334, 1.8,
        0.4444444444444444, 2.0, 0.875, 1.5714285714285714, 0.5454545454545454,
        1.6666666666666667, 0.9, 1.4444444444444444, 0.6153846153846154, 1.5,
    ],
    3: [
        None, None, None, 6.0, 1.6666666666666667, 4.5, 0.6666666666666666, 1.6,
        0.7777777777777778, 2.75, 0.75, 1.4285714285714286, 0.8181818181818182,
        2.1666666666666665, 0.8, 1.3333333333333333,
    ],
    10: [
        None, None, None, None, None, None, None, None, None, None, 6.0,
        3.3333333333333335, 4.5, 2.1666666666666665, 1.6, 1.3333333333333333,
    ],
}
ROCR100: dict[int, list[float | None]] = {
    1: [
        None, 300.0, 66.66666666666666, 300.0, 83.33333333333334, 180.0,
        44.44444444444444, 200.0, 87.5, 157.14285714285714, 54.54545454545454,
        166.66666666666669, 90.0, 144.44444444444443, 61.53846153846154, 150.0,
    ],
    3: [
        None, None, None, 600.0, 166.66666666666669, 450.0, 66.66666666666666, 160.0,
        77.77777777777779, 275.0, 75.0, 142.85714285714286, 81.81818181818183,
        216.66666666666666, 80.0, 133.33333333333331,
    ],
    10: [
        None, None, None, None, None, None, None, None, None, None, 600.0,
        333.33333333333337, 450.0, 216.66666666666666, 160.0, 133.33333333333331,
    ],
}
# fmt: on


def evaluate(expr: pl.Expr, values: Sequence[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


class TestRateOfChange(IndicatorAssertions):
    def test_known_values(self) -> None:
        cases = (
            (MOM, mom),
            (ROC, roc),
            (ROCP, rocp),
            (ROCR, rocr),
            (ROCR100, rocr100),
        )
        for table, function in cases:
            for window, expected in table.items():
                with self.subTest(function=function.__name__, window=window):
                    self.assert_values_equal(
                        evaluate(function("close", window)), expected
                    )

    def test_warm_up_is_window_nulls(self) -> None:
        for function in (mom, roc, rocp, rocr, rocr100):
            for window in (1, 4, 7):
                with self.subTest(function=function.__name__, window=window):
                    result = evaluate(function("close", window))
                    self.assertEqual(result[:window], [None] * window)
                    self.assertIsNotNone(result[window])

    def test_families_are_consistent(self) -> None:
        ratio = evaluate(rocr("close", 4))
        self.assert_values_equal(
            evaluate(rocr100("close", 4)),
            [None if value is None else value * 100.0 for value in ratio],
        )
        self.assert_values_equal(
            evaluate(roc("close", 4)),
            [None if value is None else (value - 1.0) * 100.0 for value in ratio],
        )
        self.assert_values_equal(
            evaluate(rocp("close", 4)),
            [None if value is None else value - 1.0 for value in ratio],
        )

    def test_flat_series_has_no_change(self) -> None:
        flat = constant(8, 4.0)
        self.assert_values_equal(evaluate(mom("close", 3), flat)[3:], [0.0] * 5)
        self.assert_values_equal(evaluate(roc("close", 3), flat)[3:], [0.0] * 5)
        self.assert_values_equal(evaluate(rocr("close", 3), flat)[3:], [1.0] * 5)

    def test_zero_reference_price_reports_zero(self) -> None:
        values = [0.0, 0.0, 5.0, 6.0]
        for function in (roc, rocp, rocr, rocr100):
            with self.subTest(function=function.__name__):
                self.assert_values_equal(
                    evaluate(function("close", 2), values), [None, None, 0.0, 0.0]
                )

    def test_null_blanks_both_endpoints(self) -> None:
        values = with_null(VALUES, 4)
        result = evaluate(roc("close", 3), values)
        self.assertIsNone(result[4])
        self.assertIsNone(result[7])
        self.assertIsNotNone(result[6])
        self.assertIsNotNone(result[8])

    def test_default_window_is_ten(self) -> None:
        for function in (mom, roc, rocp, rocr, rocr100):
            with self.subTest(function=function.__name__):
                self.assert_values_equal(
                    evaluate(function("close")), evaluate(function("close", 10))
                )

    def test_name_and_expression_agree(self) -> None:
        from_name = evaluate(roc("close", 3))
        self.assert_values_equal(evaluate(roc(pl.col("close"), 3)), from_name)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            mom(cast(Any, pl.Series("close", VALUES)), 3)

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(roc("close", 3).alias("roc"))
            .collect()
        )
        self.assert_values_equal(collected["roc"].to_list(), ROC[3])

    def test_invalid_window_raises(self) -> None:
        for function in (mom, roc, rocp, rocr, rocr100):
            for window in (0, -1, 2.5):
                with (
                    self.subTest(function=function.__name__, window=window),
                    self.assertRaises(ValueError),
                ):
                    function("close", cast(Any, window))
