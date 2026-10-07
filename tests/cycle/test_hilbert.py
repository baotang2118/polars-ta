import polars as pl
from _assertions import IndicatorAssertions
from _data import CLOSE, constant, with_null

from polars_ta import (
    ht_dcperiod,
    ht_dcphase,
    ht_phasor,
    ht_sine,
    ht_trendline,
    ht_trendmode,
    mama,
)
from polars_ta._hilbert import LONG_LOOKBACK, SHORT_LOOKBACK

VALUES: list[float] = CLOSE
BARS: pl.DataFrame = pl.DataFrame({"close": VALUES})

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
HT_DCPERIOD: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, 15.646696015449795, 17.21176482512415, 18.933208777083244,
    20.826708859320846, 21.982465453438753, 22.229407749705253, 21.899782799074323,
    21.385313217321528, 20.942648351473153, 20.50203837091814, 20.002862130524466,
    19.650478707333598,
]
HT_DCPHASE: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, 205.31717875694204, 236.12191815321856,
    257.1836652620746, 277.34615111195495, 298.02958102675757, -38.80777871342639,
    -23.06077289035619, -14.525611142494995, -8.194162812130116, -5.349721248782885,
    -6.2154990973464805, -6.153884538545981,
]
HT_TRENDLINE: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, 21.293597368421054, 21.280301754385967,
    21.269789473684213, 21.2763216374269, 21.315055555555553, 21.369111111111113,
    21.32322222222222, 21.206611111111112, 21.06322222222222, 20.872888888888887,
    20.62322222222222, 20.420850877192983,
]
HT_IN_PHASE: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, 0.5870495654688908, -1.9260493337142064,
    -3.4616651066691935, -4.1687902390534575, -3.040181988555617, -0.7625228351334536,
    -0.38443002807835414, -0.09963069373918879, -0.8157490693637127,
    -1.0026745421944687, 1.462043795005446, 2.872163856007581,
]
HT_QUADRATURE: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, -4.929202624742045, -5.476465301134699, -3.176143168390866,
    0.3326452500365109, 5.117167212846117, 4.2248241661565125, 1.5421895045034884,
    0.3929516988141592, -0.35384909524350777, 3.4822114884244835, 5.352950862444049,
    3.3880099071072847,
]
HT_LEAD_SINE: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, -0.9415715723726591, -0.9812189440813858,
    -0.8463450520767927, -0.6108895201323697, -0.2918779387220065, 0.107864384639729,
    0.37362293098201765, 0.5071531651006531, 0.5991051724353916, 0.6380998925303104,
    0.6263929695099713, 0.6272308714645938,
]
MAMA: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, 16.035297122955242, 14.802648561477621, 14.78051613340374,
    14.643990326733551, 14.586290810396873, 14.633976269877028, 15.451988134938514,
    16.465994067469257, 17.717997033734626, 17.81459718204789, 19.962298591023945,
    19.970683661472744,
]
FAMA: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, 14.118966970910883, 14.289887368552566, 14.302153087673846,
    14.310699018650338, 14.317588813444, 14.325498499854826, 14.607120908625749,
    15.071839198336626, 15.733378657186126, 15.78540912030767, 16.829631487986738,
    16.908157792323887,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def unnest(expr: pl.Expr, bars: pl.DataFrame | None = None) -> dict[str, list]:
    source = BARS if bars is None else bars
    result = source.select(expr.alias("out")).unnest("out")
    return {name: result[name].to_list() for name in result.columns}


class TestHilbertCore(IndicatorAssertions):
    def test_known_values(self) -> None:
        cases = (
            (ht_dcperiod("close"), HT_DCPERIOD),
            (ht_dcphase("close"), HT_DCPHASE),
            (ht_trendline("close"), HT_TRENDLINE),
        )
        for expr, expected in cases:
            with self.subTest(expr=str(expr)):
                self.assert_values_equal(column(expr)[: len(expected)], expected)

    def test_short_lookback_indicators_warm_up_in_thirty_two_rows(self) -> None:
        for expr in (ht_dcperiod("close"),):
            with self.subTest(expr=str(expr)):
                result = column(expr)
                self.assertEqual(result[:SHORT_LOOKBACK], [None] * SHORT_LOOKBACK)
                self.assertIsNotNone(result[SHORT_LOOKBACK])

    def test_long_lookback_indicators_warm_up_in_sixty_three_rows(self) -> None:
        for expr in (ht_dcphase("close"), ht_trendline("close"), ht_trendmode("close")):
            with self.subTest(expr=str(expr)):
                result = column(expr)
                self.assertEqual(result[:LONG_LOOKBACK], [None] * LONG_LOOKBACK)
                self.assertIsNotNone(result[LONG_LOOKBACK])

    def test_null_input_ends_the_recursion(self) -> None:
        bars = pl.DataFrame({"close": with_null(VALUES, 70)})
        result = column(ht_dcperiod("close"), bars)
        self.assertIsNotNone(result[69])
        self.assertEqual(result[70:], [None] * (len(VALUES) - 70))

    def test_input_shorter_than_the_priming_window_is_all_null(self) -> None:
        bars = pl.DataFrame({"close": VALUES[:10]})
        self.assertEqual(column(ht_dcperiod("close"), bars), [None] * 10)


class TestHtDcperiod(IndicatorAssertions):
    def test_period_stays_within_its_clamp(self) -> None:
        for value in column(ht_dcperiod("close")):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 50.0)

    def test_series_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ht_dcperiod(pl.Series("close", VALUES))


class TestHtDcphase(IndicatorAssertions):
    def test_phase_is_wrapped_below_three_hundred_and_fifteen(self) -> None:
        for value in column(ht_dcphase("close")):
            if value is not None:
                self.assertLessEqual(value, 315.0)


class TestHtPhasor(IndicatorAssertions):
    def test_field_names_and_warm_up(self) -> None:
        fields = unnest(ht_phasor("close"))
        self.assertEqual(list(fields), ["in_phase", "quadrature"])
        for name in fields:
            self.assertEqual(fields[name][:SHORT_LOOKBACK], [None] * SHORT_LOOKBACK)
            self.assertIsNotNone(fields[name][SHORT_LOOKBACK])

    def test_known_values(self) -> None:
        fields = unnest(ht_phasor("close"))
        self.assert_values_equal(fields["in_phase"][: len(HT_IN_PHASE)], HT_IN_PHASE)
        self.assert_values_equal(
            fields["quadrature"][: len(HT_QUADRATURE)], HT_QUADRATURE
        )


class TestHtSine(IndicatorAssertions):
    def test_both_lines_stay_within_minus_one_and_one(self) -> None:
        fields = unnest(ht_sine("close"))
        self.assertEqual(list(fields), ["sine", "lead_sine"])
        for name in fields:
            for value in fields[name]:
                if value is not None:
                    self.assertGreaterEqual(value, -1.0)
                    self.assertLessEqual(value, 1.0)

    def test_lead_sine_leads_by_forty_five_degrees(self) -> None:
        fields = unnest(ht_sine("close"))
        self.assert_values_equal(fields["lead_sine"][: len(HT_LEAD_SINE)], HT_LEAD_SINE)


class TestHtTrendmode(IndicatorAssertions):
    def test_reports_only_zero_or_one(self) -> None:
        for value in column(ht_trendmode("close")):
            if value is not None:
                self.assertIn(value, (0, 1))

    def test_a_straight_line_is_always_trending(self) -> None:
        rising = [float(index) + 10.0 for index in range(120)]
        bars = pl.DataFrame({"close": rising})
        result = column(ht_trendmode("close"), bars)
        self.assertEqual(set(result[LONG_LOOKBACK:]), {1})


class TestHtTrendline(IndicatorAssertions):
    def test_tracks_a_flat_series_at_its_level(self) -> None:
        bars = pl.DataFrame({"close": constant(120, 20.0)})
        result = column(ht_trendline("close"), bars)
        self.assert_values_equal(result[LONG_LOOKBACK:], [20.0] * (120 - LONG_LOOKBACK))


class TestMama(IndicatorAssertions):
    def test_field_names_and_warm_up(self) -> None:
        fields = unnest(mama("close"))
        self.assertEqual(list(fields), ["mama", "fama"])
        for name in fields:
            self.assertEqual(fields[name][:SHORT_LOOKBACK], [None] * SHORT_LOOKBACK)
            self.assertIsNotNone(fields[name][SHORT_LOOKBACK])

    def test_known_values(self) -> None:
        fields = unnest(mama("close"))
        self.assert_values_equal(fields["mama"][: len(MAMA)], MAMA)
        self.assert_values_equal(fields["fama"][: len(FAMA)], FAMA)

    def test_flat_series_settles_on_its_level(self) -> None:
        bars = pl.DataFrame({"close": constant(120, 7.0)})
        fields = unnest(mama("close"), bars)
        self.assertAlmostEqual(fields["mama"][-1], 7.0, places=6)

    def test_a_faster_limit_tracks_price_more_closely(self) -> None:
        quick = unnest(mama("close", fast_limit=0.9, slow_limit=0.5))["mama"]
        slow = unnest(mama("close", fast_limit=0.1, slow_limit=0.01))["mama"]
        quick_error = sum(
            abs(a - b) for a, b in zip(quick[SHORT_LOOKBACK:], VALUES[SHORT_LOOKBACK:])
        )
        slow_error = sum(
            abs(a - b) for a, b in zip(slow[SHORT_LOOKBACK:], VALUES[SHORT_LOOKBACK:])
        )
        self.assertLess(quick_error, slow_error)

    def test_dispatcher_exposes_the_mama_line(self) -> None:
        from polars_ta import ma

        self.assert_values_equal(
            column(ma("close", ma_type="mama")), unnest(mama("close"))["mama"]
        )

    def test_invalid_limits_raise(self) -> None:
        for limit in (0.0, -0.1, 1.5, "fast"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                mama("close", fast_limit=limit)
