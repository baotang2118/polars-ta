import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, frame_from, ramp_up

from polars_ta import adx, adxr, dx, minus_di, minus_dm, plus_di, plus_dm

LENGTH: int = 60
BARS: pl.DataFrame = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH])


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def reference_directional(
    high: list[float], low: list[float], close: list[float], window: int
) -> dict[str, list[float | None]]:
    """TA-Lib's running Wilder sums, seeded with ``window - 1`` terms."""
    size = len(high)
    plus_raw = [None] * size
    minus_raw = [None] * size
    ranges = [None] * size
    for index in range(1, size):
        up = high[index] - high[index - 1]
        down = low[index - 1] - low[index]
        plus_raw[index] = up if up > down and up > 0.0 else 0.0
        minus_raw[index] = down if down > up and down > 0.0 else 0.0
        ranges[index] = max(
            high[index] - low[index],
            abs(high[index] - close[index - 1]),
            abs(low[index] - close[index - 1]),
        )

    def wilder(values: list[float | None]) -> list[float | None]:
        if window == 1:
            return list(values)
        running: list[float | None] = [None] * size
        total = sum(values[1:window])
        running[window - 1] = total
        for index in range(window, size):
            total = total - total / window + values[index]
            running[index] = total
        return running

    plus_sum = wilder(plus_raw)
    minus_sum = wilder(minus_raw)
    range_sum = wilder(ranges)
    start = window if window > 1 else 1
    positive: list[float | None] = [None] * size
    negative: list[float | None] = [None] * size
    index_dx: list[float | None] = [None] * size
    for index in range(start, size):
        total_range = range_sum[index]
        positive[index] = (
            0.0 if total_range == 0.0 else 100.0 * plus_sum[index] / total_range
        )
        negative[index] = (
            0.0 if total_range == 0.0 else 100.0 * minus_sum[index] / total_range
        )
        spread = positive[index] + negative[index]
        index_dx[index] = (
            0.0
            if spread == 0.0
            else 100.0 * abs(positive[index] - negative[index]) / spread
        )

    trend: list[float | None] = [None] * size
    first = 2 * window - 1
    if first < size:
        average = sum(index_dx[window : 2 * window]) / window
        trend[first] = average
        for index in range(first + 1, size):
            average = (average * (window - 1) + index_dx[index]) / window
            trend[index] = average
    return {
        "plus_dm": plus_sum,
        "minus_dm": minus_sum,
        "plus_di": positive,
        "minus_di": negative,
        "dx": index_dx,
        "adx": trend,
    }


class TestDirectionalMovement(IndicatorAssertions):
    def test_matches_reference(self) -> None:
        for window in (2, 5, 14):
            expected = reference_directional(
                HIGH[:LENGTH], LOW[:LENGTH], CLOSE[:LENGTH], window
            )
            cases = (
                ("plus_dm", plus_dm("high", "low", window)),
                ("minus_dm", minus_dm("high", "low", window)),
                ("plus_di", plus_di("high", "low", "close", window)),
                ("minus_di", minus_di("high", "low", "close", window)),
                ("dx", dx("high", "low", "close", window)),
            )
            for name, expr in cases:
                with self.subTest(window=window, name=name):
                    self.assert_values_equal(column(expr), expected[name])

    def test_adx_struct_matches_reference(self) -> None:
        window = 5
        expected = reference_directional(
            HIGH[:LENGTH], LOW[:LENGTH], CLOSE[:LENGTH], window
        )
        fields = BARS.select(adx("high", "low", "close", window).alias("a")).unnest("a")
        for name in ("adx", "plus_di", "minus_di"):
            with self.subTest(name=name):
                self.assert_values_equal(fields[name].to_list(), expected[name])

    def test_movement_warm_up_is_window_minus_one(self) -> None:
        for window in (2, 5, 14):
            with self.subTest(window=window):
                for expr in (
                    plus_dm("high", "low", window),
                    minus_dm("high", "low", window),
                ):
                    result = column(expr)
                    self.assertEqual(result[: window - 1], [None] * (window - 1))
                    self.assertIsNotNone(result[window - 1])

    def test_indicator_warm_up_is_window(self) -> None:
        for window in (2, 5, 14):
            with self.subTest(window=window):
                for expr in (
                    plus_di("high", "low", "close", window),
                    minus_di("high", "low", "close", window),
                    dx("high", "low", "close", window),
                ):
                    result = column(expr)
                    self.assertEqual(result[:window], [None] * window)
                    self.assertIsNotNone(result[window])

    def test_adxr_averages_two_adx_readings(self) -> None:
        window = 5
        trend = BARS.select(adx("high", "low", "close", window).struct.field("adx"))
        line = trend.to_series().to_list()
        expected = [
            None
            if line[index] is None
            or index < window - 1
            or line[index - window + 1] is None
            else (line[index] + line[index - window + 1]) / 2.0
            for index in range(LENGTH)
        ]
        self.assert_values_equal(column(adxr("high", "low", "close", window)), expected)

    def test_adxr_warm_up_is_three_windows_less_two(self) -> None:
        for window in (3, 5, 7):
            with self.subTest(window=window):
                result = column(adxr("high", "low", "close", window))
                lookback = 3 * window - 2
                self.assertEqual(result[:lookback], [None] * lookback)
                self.assertIsNotNone(result[lookback])

    def test_only_one_movement_is_non_zero_per_bar(self) -> None:
        rising = ramp_up(30)
        bars = frame_from(rising)
        self.assert_values_equal(
            column(minus_dm("high", "low", 5), bars)[4:], [0.0] * 26
        )
        for value in column(plus_dm("high", "low", 5), bars)[4:]:
            self.assertGreater(value, 0.0)

    def test_flat_market_reports_zero(self) -> None:
        bars = frame_from(constant(30), 0.0)
        for expr in (
            plus_di("high", "low", "close", 4),
            minus_di("high", "low", "close", 4),
            dx("high", "low", "close", 4),
        ):
            self.assert_values_equal(column(expr, bars)[4:], [0.0] * 26)

    def test_window_of_one_is_unsmoothed(self) -> None:
        expected = reference_directional(HIGH[:LENGTH], LOW[:LENGTH], CLOSE[:LENGTH], 1)
        self.assert_values_equal(column(plus_dm("high", "low", 1)), expected["plus_dm"])
        self.assert_values_equal(
            column(plus_di("high", "low", "close", 1)), expected["plus_di"]
        )

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                dx("high", "low", "close", window)
            with self.subTest(window=window), self.assertRaises(ValueError):
                plus_dm("high", "low", window)
