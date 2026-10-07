import polars as pl
from _assertions import IndicatorAssertions
from _data import HIGH, LOW, ramp_down, ramp_up

from polars_ta import sar, sarext

LENGTH: int = 80
BARS: pl.DataFrame = pl.DataFrame({"high": HIGH[:LENGTH], "low": LOW[:LENGTH]})


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def reference_sar(
    high: list[float], low: list[float], acceleration: float, maximum: float
) -> list[float | None]:
    """TA-Lib's Parabolic SAR scan, written out step by step."""
    size = len(high)
    result: list[float | None] = [None] * size
    up = high[1] - high[0]
    down = low[0] - low[1]
    is_long = not (down > up and down > 0.0)
    extreme = high[1] if is_long else low[1]
    stop = low[0] if is_long else high[0]
    factor = acceleration
    previous_high, previous_low = high[1], low[1]
    for index in range(1, size):
        if index > 1:
            previous_high, previous_low = high[index - 1], low[index - 1]
        if is_long:
            if low[index] <= stop:
                is_long = False
                stop = max(extreme, previous_high, high[index])
                result[index] = stop
                factor = acceleration
                extreme = low[index]
                stop = max(stop + factor * (extreme - stop), previous_high, high[index])
            else:
                result[index] = stop
                if high[index] > extreme:
                    extreme = high[index]
                    factor = min(factor + acceleration, maximum)
                stop = min(stop + factor * (extreme - stop), previous_low, low[index])
        elif high[index] >= stop:
            is_long = True
            stop = min(extreme, previous_low, low[index])
            result[index] = stop
            factor = acceleration
            extreme = high[index]
            stop = min(stop + factor * (extreme - stop), previous_low, low[index])
        else:
            result[index] = stop
            if low[index] < extreme:
                extreme = low[index]
                factor = min(factor + acceleration, maximum)
            stop = max(stop + factor * (extreme - stop), previous_high, high[index])
    return result


class TestSar(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        for acceleration, maximum in ((0.02, 0.2), (0.05, 0.3)):
            with self.subTest(acceleration=acceleration, maximum=maximum):
                self.assert_values_equal(
                    column(sar("high", "low", acceleration, maximum)),
                    reference_sar(HIGH[:LENGTH], LOW[:LENGTH], acceleration, maximum),
                )

    def test_warm_up_is_one_row(self) -> None:
        result = column(sar("high", "low"))
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[1])

    def test_stays_below_price_in_a_sustained_rise(self) -> None:
        rising = ramp_up(40)
        bars = pl.DataFrame({"high": rising, "low": [value - 1.0 for value in rising]})
        result = column(sar("high", "low"), bars)
        for index in range(2, 40):
            self.assertLessEqual(result[index], rising[index])

    def test_stays_above_price_in_a_sustained_fall(self) -> None:
        falling = ramp_down(40, 60.0)
        bars = pl.DataFrame(
            {"high": [value + 1.0 for value in falling], "low": falling}
        )
        result = column(sar("high", "low"), bars)
        for index in range(2, 40):
            self.assertGreaterEqual(result[index], falling[index])

    def test_null_ends_the_scan(self) -> None:
        highs = list(HIGH[:20])
        lows = list(LOW[:20])
        highs[10] = None
        result = column(sar("high", "low"), pl.DataFrame({"high": highs, "low": lows}))
        self.assertIsNotNone(result[9])
        self.assertEqual(result[10:], [None] * 10)

    def test_too_short_an_input_is_all_null(self) -> None:
        bars = pl.DataFrame({"high": [2.0], "low": [1.0]})
        self.assertEqual(column(sar("high", "low"), bars), [None])

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            sar(pl.Series("high", HIGH[:LENGTH]), pl.Series("low", LOW[:LENGTH]))

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(BARS).with_columns(sar("high", "low").alias("sar")).collect()
        )
        self.assert_values_equal(collected["sar"].to_list(), column(sar("high", "low")))

    def test_invalid_arguments_raise(self) -> None:
        for acceleration in (0.0, -0.1, "fast"):
            with self.subTest(acceleration=acceleration), self.assertRaises(ValueError):
                sar("high", "low", acceleration)


class TestSarext(IndicatorAssertions):
    def test_matches_plain_sar_up_to_the_sign(self) -> None:
        plain = column(sar("high", "low"))
        extended = column(sarext("high", "low"))
        self.assert_values_equal(
            [None if value is None else abs(value) for value in extended], plain
        )

    def test_short_readings_are_negative(self) -> None:
        extended = column(sarext("high", "low"))
        self.assertTrue(any(value < 0.0 for value in extended[1:]))
        self.assertTrue(any(value > 0.0 for value in extended[1:]))

    def test_positive_start_value_begins_long(self) -> None:
        rising = ramp_up(30)
        bars = pl.DataFrame({"high": rising, "low": [value - 1.0 for value in rising]})
        result = column(sarext("high", "low", start_value=0.5), bars)
        self.assertGreater(result[1], 0.0)

    def test_negative_start_value_begins_short(self) -> None:
        rising = ramp_up(30)
        bars = pl.DataFrame({"high": rising, "low": [value - 1.0 for value in rising]})
        result = column(sarext("high", "low", start_value=-100.0), bars)
        self.assertLess(result[1], 0.0)

    def test_offset_on_reverse_widens_the_stop(self) -> None:
        plain = column(sarext("high", "low"))
        offset = column(sarext("high", "low", offset_on_reverse=0.05))
        self.assertNotEqual(plain, offset)

    def test_asymmetric_acceleration_changes_the_result(self) -> None:
        symmetric = column(sarext("high", "low"))
        skewed = column(
            sarext(
                "high",
                "low",
                acceleration_long=0.05,
                acceleration_max_long=0.5,
            )
        )
        self.assertNotEqual(symmetric, skewed)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            sarext("high", "low", acceleration_long=0.0)
        with self.assertRaises(ValueError):
            sarext("high", "low", offset_on_reverse=-0.1)
