import polars as pl
from _assertions import IndicatorAssertions
from _data import HAND_CHECKED, constant, with_null

from polars_ta import mom, roc, rocp, rocr, rocr100

VALUES: list[float] = HAND_CHECKED


def evaluate(expr: pl.Expr, values: list[float | None] | None = None):
    data = VALUES if values is None else values
    return pl.DataFrame({"close": data}).select(expr).to_series().to_list()


def reference_change(
    values: list[float], window: int, kind: str
) -> list[float | None]:
    """TA-Lib's rate-of-change family, reporting 0.0 on a zero reference price."""
    result: list[float | None] = [None] * len(values)
    for index in range(window, len(values)):
        now, before = values[index], values[index - window]
        if kind == "mom":
            result[index] = now - before
        elif before == 0.0:
            result[index] = 0.0
        elif kind == "roc":
            result[index] = (now / before - 1.0) * 100.0
        elif kind == "rocp":
            result[index] = (now - before) / before
        elif kind == "rocr":
            result[index] = now / before
        else:
            result[index] = now / before * 100.0
    return result


class TestRateOfChange(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        cases = (
            ("mom", mom),
            ("roc", roc),
            ("rocp", rocp),
            ("rocr", rocr),
            ("rocr100", rocr100),
        )
        for kind, function in cases:
            for window in (1, 3, 10):
                with self.subTest(kind=kind, window=window):
                    self.assert_values_equal(
                        evaluate(function("close", window)),
                        reference_change(VALUES, window, kind),
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

    def test_name_expression_and_series_agree(self) -> None:
        from_name = evaluate(roc("close", 3))
        self.assert_values_equal(evaluate(roc(pl.col("close"), 3)), from_name)
        self.assert_values_equal(roc(pl.Series("close", VALUES), 3).to_list(), from_name)

    def test_series_input_keeps_its_name(self) -> None:
        result = mom(pl.Series("close", VALUES), 3)
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.name, "close")

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame({"close": VALUES})
            .with_columns(roc("close", 3).alias("roc"))
            .collect()
        )
        self.assert_values_equal(
            collected["roc"].to_list(), reference_change(VALUES, 3, "roc")
        )

    def test_invalid_window_raises(self) -> None:
        for function in (mom, roc, rocp, rocr, rocr100):
            for window in (0, -1, 2.5):
                with (
                    self.subTest(function=function.__name__, window=window),
                    self.assertRaises(ValueError),
                ):
                    function("close", window)
