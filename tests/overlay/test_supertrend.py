import polars as pl
from _assertions import IndicatorAssertions

from polars_ta import supertrend

HIGH: list[float] = [
    10.0,
    11,
    12,
    11,
    10,
    11,
    12,
    13,
    12,
    11,
    13,
    14,
    16,
    18,
    20,
    19,
    17,
    14,
    11,
    9,
]
LOW: list[float] = [
    8.0,
    9,
    10,
    9,
    8,
    9,
    10,
    11,
    10,
    9,
    11,
    12,
    14,
    16,
    18,
    17,
    15,
    12,
    9,
    7,
]
CLOSE: list[float] = [
    9.0,
    10,
    11,
    10,
    9,
    10,
    11,
    12,
    11,
    10,
    12,
    13,
    15,
    17,
    19,
    18,
    16,
    13,
    10,
    8,
]


def frame(high=None, low=None, close=None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "high": HIGH if high is None else high,
            "low": LOW if low is None else low,
            "close": CLOSE if close is None else close,
        }
    )


def evaluate(expr: pl.Expr, data: pl.DataFrame | None = None) -> pl.DataFrame:
    source = data if data is not None else frame()
    return source.select(expr.alias("s")).unnest("s")


class TestSupertrend(IndicatorAssertions):
    def test_field_names_and_types(self) -> None:
        result = evaluate(supertrend("high", "low", "close", 5, 2.0))
        self.assertEqual(result.columns, ["supertrend", "direction"])
        self.assertEqual(result["supertrend"].dtype, pl.Float64)
        self.assertEqual(result["direction"].dtype, pl.Int8)

    def test_warm_up_follows_the_atr(self) -> None:
        for window in (3, 5, 7):
            with self.subTest(window=window):
                result = evaluate(supertrend("high", "low", "close", window, 2.0))
                self.assertEqual(
                    result["supertrend"].to_list()[:window], [None] * window
                )
                self.assertIsNotNone(result["supertrend"][window])

    def test_direction_is_only_ever_plus_or_minus_one(self) -> None:
        for value in evaluate(supertrend("high", "low", "close", 5, 2.0))["direction"]:
            if value is not None:
                self.assertIn(value, (1, -1))

    def test_band_sits_below_price_while_rising(self) -> None:
        result = evaluate(supertrend("high", "low", "close", 5, 2.0))
        for index, direction in enumerate(result["direction"].to_list()):
            if direction == 1:
                self.assertLessEqual(result["supertrend"][index], CLOSE[index])

    def test_band_sits_above_price_while_falling(self) -> None:
        result = evaluate(supertrend("high", "low", "close", 5, 2.0))
        for index, direction in enumerate(result["direction"].to_list()):
            if direction == -1:
                self.assertGreaterEqual(result["supertrend"][index], CLOSE[index])

    def test_sustained_rise_then_fall_flips_the_direction(self) -> None:
        directions = evaluate(supertrend("high", "low", "close", 5, 2.0))[
            "direction"
        ].to_list()
        self.assertIn(1, directions)
        self.assertIn(-1, directions)

    def test_monotonic_rise_never_flips(self) -> None:
        rising = [float(index) for index in range(1, 30)]
        data = frame(rising, [v - 1 for v in rising], rising)
        directions = evaluate(supertrend("high", "low", "close", 5, 2.0), data)[
            "direction"
        ]
        self.assertEqual(set(directions.drop_nulls().to_list()), {1})

    def test_band_ratchets_upward_while_the_trend_holds(self) -> None:
        rising = [float(index) for index in range(1, 30)]
        data = frame(rising, [v - 1 for v in rising], rising)
        values = (
            evaluate(supertrend("high", "low", "close", 5, 2.0), data)["supertrend"]
            .drop_nulls()
            .to_list()
        )
        for earlier, later in zip(values, values[1:]):
            self.assertGreaterEqual(later, earlier)

    def test_larger_multiplier_gives_a_looser_band(self) -> None:
        rising = [float(index) for index in range(1, 30)]
        data = frame(rising, [v - 1 for v in rising], rising)
        tight = evaluate(supertrend("high", "low", "close", 5, 1.0), data)["supertrend"]
        loose = evaluate(supertrend("high", "low", "close", 5, 4.0), data)["supertrend"]
        self.assertGreater(tight[-1], loose[-1])

    def test_null_input_propagates(self) -> None:
        close = list(CLOSE)
        close[8] = None
        result = evaluate(
            supertrend("high", "low", "close", 3, 2.0), frame(close=close)
        )
        self.assertIsNone(result["supertrend"][8])

    def test_input_shorter_than_warm_up_is_all_null(self) -> None:
        result = evaluate(supertrend("high", "low", "close", len(HIGH), 2.0))
        self.assertEqual(result["supertrend"].to_list(), [None] * len(HIGH))

    def test_series_input_returns_a_struct_series(self) -> None:
        result = supertrend(
            pl.Series("high", HIGH),
            pl.Series("low", LOW),
            pl.Series("close", CLOSE),
            5,
            2.0,
        )
        self.assertIsInstance(result, pl.Series)
        self.assertEqual(result.struct.fields, ["supertrend", "direction"])

    def test_expression_works_in_a_lazy_frame(self) -> None:
        collected = (
            pl.LazyFrame(frame())
            .with_columns(supertrend("high", "low", "close", 5, 2.0).alias("s"))
            .collect()
        )
        eager = evaluate(supertrend("high", "low", "close", 5, 2.0))
        self.assert_values_equal(
            collected["s"].struct.field("supertrend").to_list(),
            eager["supertrend"].to_list(),
        )

    def test_invalid_arguments_raise(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                supertrend("high", "low", "close", window, 2.0)
        for multiplier in (0.0, -1.0):
            with self.subTest(multiplier=multiplier), self.assertRaises(ValueError):
                supertrend("high", "low", "close", 5, multiplier)
