import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HAND_CHECKED, constant, frame, ramp_up

from polars_ta import dpo, kst, stc

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
# dpo(close, 4) over HAND_CHECKED: a four-bar mean against the bar three back.
DPO_4: list[float | None] = [
    None, None, None, -2.0, -1.0, -3.5, 0.0, -1.5, 2.0, -3.5, 0.0, -1.5, 2.0, -3.5,
    0.0, -1.5,
]
# kst(close, (3, 5, 7, 9), (2, 2, 3, 4), 5)
KST_LINE: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None,
    157.6767676767677, 83.8888888888889, -37.60683760683757, -56.69774669774669,
    -30.701709401709387, 36.95233100233102, 169.95128205128208, 263.36744857084665,
    288.86165395994254, 361.65397164035494, 341.7460736617385, 390.8626179430882,
]
KST_SIGNAL: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, 23.31187257187259, -0.8330147630147451, 16.37946386946389,
    76.57432110500073, 145.68620123653858, 224.15733744495145, 285.11608597683295,
    329.29835315519415, 367.49336649067584, 407.0237327215087, 430.47480361470764,
    433.76533624772367,
]
# stc(close, 5, 10, 4, smooth_k=3, smooth_d=3)
STC: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, 100.0, 100.0, 100.0, 50.0, 51.97947677650601,
    66.59740415972848, 83.29870207986424, 91.64935103993213, 45.824675519966064,
    22.912337759983032, 11.456168879991516, 5.728084439995758,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def field(name: str, expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    return column(expr.struct.field(name), bars)


class TestDpo(IndicatorAssertions):
    def test_matches_hand_checked_values(self) -> None:
        bars = pl.DataFrame({"close": HAND_CHECKED})
        self.assert_values_equal(column(dpo("close", 4), bars), DPO_4)

    def test_warm_up_covers_the_average_and_the_shift(self) -> None:
        for window in (4, 10, 20):
            with self.subTest(window=window):
                result = column(dpo("close", window))
                warm_up = max(window - 1, window // 2 + 1)
                self.assertEqual(result[:warm_up], [None] * warm_up)
                self.assertIsNotNone(result[warm_up])

    def test_a_flat_series_detrends_to_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(20, 6.0)})
        self.assert_values_equal(column(dpo("close", 5), bars), [None] * 4 + [0.0] * 16)

    def test_a_rising_series_reads_negative(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(20)})
        self.assertLess(column(dpo("close", 6), bars)[-1], 0.0)

    def test_invalid_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            dpo("close", 0)


class TestKst(IndicatorAssertions):
    def test_known_values(self) -> None:
        indicator = kst("close", (3, 5, 7, 9), (2, 2, 3, 4), 5)
        line = field("kst", indicator)
        signal = field("signal", indicator)
        self.assert_values_equal(line[: len(KST_LINE)], KST_LINE)
        self.assert_values_equal(signal[: len(KST_SIGNAL)], KST_SIGNAL)

    def test_warm_up_follows_the_slowest_term(self) -> None:
        indicator = kst("close", (3, 5, 7, 9), (2, 2, 3, 4), 5)
        line = field("kst", indicator)
        warm_up = 9 + 4 - 1
        self.assertEqual(line[:warm_up], [None] * warm_up)
        self.assertIsNotNone(line[warm_up])

    def test_signal_trails_the_line(self) -> None:
        indicator = kst("close", (3, 5, 7, 9), (2, 2, 3, 4), 5)
        line = field("kst", indicator)
        signal = field("signal", indicator)
        self.assertIsNotNone(line[12])
        self.assertIsNone(signal[12])
        self.assertIsNotNone(signal[16])

    def test_a_flat_series_reads_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(40, 5.0)})
        self.assertAlmostEqual(
            field("kst", kst("close", (3, 5, 7, 9), (2, 2, 3, 4), 5), bars)[-1], 0.0
        )

    def test_invalid_periods_raise(self) -> None:
        with self.assertRaises(ValueError):
            kst("close", (10, 15, 20), (10, 10, 10, 15))
        with self.assertRaises(ValueError):
            kst("close", (10, 15, 20, 0))
        with self.assertRaises(ValueError):
            kst("close", signal_period=0)


class TestStc(IndicatorAssertions):
    def test_stays_inside_zero_and_one_hundred(self) -> None:
        result = column(stc("close", 5, 10, 4, smooth_k=3, smooth_d=3))
        emitted = [value for value in result if value is not None]
        self.assertTrue(emitted)
        for value in emitted:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_known_values(self) -> None:
        result = column(stc("close", 5, 10, 4, smooth_k=3, smooth_d=3))
        self.assert_values_equal(result[: len(STC)], STC)

    def test_a_flat_series_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(60, 8.0)})
        result = column(stc("close", 5, 10, 4, smooth_k=3, smooth_d=3), bars)
        self.assertAlmostEqual(result[-1], 0.0)

    def test_invalid_period_raises(self) -> None:
        with self.assertRaises(ValueError):
            stc("close", 0)
        with self.assertRaises(ValueError):
            stc("close", cycle=0)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            stc(pl.Series("close", CLOSE[:LENGTH]), 5, 10, 4)
