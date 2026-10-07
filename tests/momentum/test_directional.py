import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, HIGH, LOW, constant, frame, frame_from, ramp_up

from polars_ta import adx, adxr, dx, minus_di, minus_dm, plus_di, plus_dm

LENGTH: int = 60
BARS: pl.DataFrame = frame(high=HIGH[:LENGTH], low=LOW[:LENGTH], close=CLOSE[:LENGTH])

# Frozen expectations: each table covers the warm-up plus the first live bars.
# fmt: off
PLUS_DM: dict[int, list[float | None]] = {
    2: [
        None, 1.25, 1.875, 0.9375, 0.46875, 1.484375, 1.9921875, 2.24609375,
        1.123046875, 0.5615234375, 2.53076171875, 2.515380859375, 1.2576904296875,
    ],
    5: [
        None, None, None, None, 2.5, 3.25, 3.85, 4.33, 3.464, 2.7712, 4.46696,
        4.823568, 3.8588544, 3.0870835199999997, 2.4696668159999997, 4.2257334528,
    ],
    14: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        9.75, 9.053571428571429, 10.65688775510204, 9.895681486880466,
        10.438847094960433, 12.093215159606117, 11.789414076777108, 10.94731307129303,
        12.555362137629242, 11.65855055637001, 13.385796945200722, 14.009668591972101,
    ],
}
MINUS_DM: dict[int, list[float | None]] = {
    2: [
        None, 0.0, 0.0, 1.25, 0.875, 0.4375, 0.21875, 0.109375, 0.3046875, 1.40234375,
        0.701171875, 0.3505859375, 1.42529296875,
    ],
    5: [
        None, None, None, None, 1.5, 1.2, 0.96, 0.768, 0.8644000000000001,
        1.9415200000000001, 1.5532160000000002, 1.2425728, 2.24405824,
        3.0452465920000003, 3.6861972736000004, 2.9489578188800003,
    ],
    14: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        5.5, 6.357142857142857, 5.903061224489796, 5.481413994169096,
        5.089884423157018, 4.726321250074374, 4.388726875069062, 4.075246383992701,
        3.7841573565646507, 5.433860402524319, 5.045727516629725, 4.685318408299031,
    ],
}
PLUS_DI: dict[int, list[float | None]] = {
    2: [
        None, None, 44.11764705882353, 16.666666666666668, 9.74025974025974,
        30.254777070063696, 36.532951289398284, 36.07277289836888, 21.963330786860197,
        11.104673619157976, 41.9812084885793, 38.614046923019266, 20.100277029927035,
        11.172439220108872,
    ],
    5: [
        None, None, None, None, None, 28.761061946902654, 31.97674418604651,
        32.97289064879683, 27.699590583418622, 22.161657262037284, 33.07981051548982,
        33.72448287850984, 26.719124151093293, 21.966128948025286, 17.339430064437494,
        27.902843772116178, 23.949357434219067,
    ],
    14: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, 24.235181644359468, 27.724316432174145, 26.253263076476845,
        27.83634590422246, 31.433683824845833, 30.056524513793374, 28.491993915307194,
        32.137235216021004, 29.55468077057812, 32.89724480341161, 34.0757481307317,
        31.981441514690196,
    ],
}
MINUS_DI: dict[int, list[float | None]] = {
    2: [
        None, None, 0.0, 22.22222222222222, 18.181818181818183, 8.9171974522293,
        4.011461318051576, 1.7565872020075282, 5.958747135217723, 27.732715334105833,
        11.631297586262757, 5.381905404392474, 22.77888329626595, 34.869548244377455,
    ],
    5: [
        None, None, None, None, None, 10.619469026548671, 7.973421926910299,
        5.8483094730429475, 6.912103377686797, 15.526595268255857, 11.502250069314933,
        8.687578389876961, 15.538101338273847, 21.668438474384573, 25.880640827830348,
        19.47219582821699, 16.713227573782518,
    ],
    14: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, 17.017208413001914, 15.357048048845233, 14.542202455767185,
        13.572742480720242, 12.285044619542422, 11.188840772392096, 10.606428666096717,
        9.686088997614354, 13.774955023096314, 12.400496885274478, 11.396110403708288,
        12.834521106668435,
    ],
}
DX: dict[int, list[float | None]] = {
    2: [
        None, None, 100.0, 14.285714285714281, 30.23255813953489, 54.47154471544716,
        80.21201413427563, 90.71310116086235, 57.318741450068394, 42.814520139234205,
        56.60975978244448, 75.53454297640343, 6.24687201419537, 51.46847547045385,
    ],
    5: [
        None, None, None, None, None, 46.067415730337075, 60.08316008316009,
        69.87053746567283, 60.05914425653821, 17.605119760987282, 48.39964811659991,
        59.03251042244189, 26.459434293983264, 0.6822354184444729, 19.762139642674427,
        17.79554806713006, 17.795548067130074,
    ],
    14: [
        None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, 17.49710312862109, 28.70677039205115, 28.706770392051144,
        34.44558665698323, 43.79962521019144, 45.74497912792334, 45.74497912792332,
        53.6809224052235, 36.417859182156896, 45.24893991184676, 49.87620576327675,
        42.723438900086514,
    ],
}
ADX_5: list[float | None] = [
    None, None, None, None, None, None, None, None, None, 50.737075459339096,
    50.26958999079126, 52.022174077121385, 46.90962612049376, 37.664147980083904,
    34.08374631260201, 30.82610666350762, 28.21999494423211, 29.653046013768442,
    35.184209351321876, 40.37800911111405, 44.533048918947785,
]
ADXR_5: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None,
    44.2006117197115, 42.17666815169663, 41.424140370314504, 37.564810532362934,
    33.65859699692617, 34.63397783196194, 35.602057887310835, 36.37652193158995,
    40.212579864433664, 40.664802436200745, 43.677623163758696, 47.055068624785015,
]
# A window of one leaves the movement unsmoothed.
PLUS_DM_1: list[float | None] = [
    None, 1.25, 1.25, 0.0, 0.0, 1.25, 1.25, 1.25, 0.0, 0.0, 2.25, 1.25, 0.0,
]
PLUS_DI_1: list[float | None] = [
    None, 50.0, 41.666666666666664, 0.0, 0.0, 50.0, 41.666666666666664,
    35.714285714285715, 0.0, 0.0, 64.28571428571429, 35.714285714285715, 0.0,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestDirectionalMovement(IndicatorAssertions):
    def test_known_values(self) -> None:
        for window in (2, 5, 14):
            cases = (
                ("plus_dm", PLUS_DM[window], plus_dm("high", "low", window)),
                ("minus_dm", MINUS_DM[window], minus_dm("high", "low", window)),
                ("plus_di", PLUS_DI[window], plus_di("high", "low", "close", window)),
                (
                    "minus_di",
                    MINUS_DI[window],
                    minus_di("high", "low", "close", window),
                ),
                ("dx", DX[window], dx("high", "low", "close", window)),
            )
            for name, expected, expr in cases:
                with self.subTest(window=window, name=name):
                    self.assert_values_equal(column(expr)[: len(expected)], expected)

    def test_adx_struct_known_values(self) -> None:
        fields = BARS.select(adx("high", "low", "close", 5).alias("a")).unnest("a")
        for name, expected in (
            ("adx", ADX_5),
            ("plus_di", PLUS_DI[5]),
            ("minus_di", MINUS_DI[5]),
        ):
            with self.subTest(name=name):
                self.assert_values_equal(
                    fields[name].to_list()[: len(expected)], expected
                )

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
        result = column(adxr("high", "low", "close", 5))
        self.assert_values_equal(result[: len(ADXR_5)], ADXR_5)

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
        self.assert_values_equal(
            column(plus_dm("high", "low", 1))[: len(PLUS_DM_1)], PLUS_DM_1
        )
        self.assert_values_equal(
            column(plus_di("high", "low", "close", 1))[: len(PLUS_DI_1)], PLUS_DI_1
        )

    def test_invalid_window_raises(self) -> None:
        for window in (0, -1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                dx("high", "low", "close", window)
            with self.subTest(window=window), self.assertRaises(ValueError):
                plus_dm("high", "low", window)
