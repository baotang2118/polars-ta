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
from polars_ta._hilbert import LONG_LOOKBACK, SHORT_LOOKBACK, hilbert_transform

VALUES: list[float] = CLOSE
BARS: pl.DataFrame = pl.DataFrame({"close": VALUES})


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


def unnest(expr: pl.Expr, bars: pl.DataFrame | None = None) -> dict[str, list]:
    source = BARS if bars is None else bars
    result = source.select(expr.alias("out")).unnest("out")
    return {name: result[name].to_list() for name in result.columns}


REFERENCE = hilbert_transform(VALUES, 0.5, 0.05)


class TestHilbertCore(IndicatorAssertions):
    def test_every_indicator_draws_on_the_same_recursion(self) -> None:
        cases = (
            (ht_dcperiod("close"), REFERENCE.smooth_period, SHORT_LOOKBACK),
            (ht_dcphase("close"), REFERENCE.dc_phase, LONG_LOOKBACK),
            (ht_trendline("close"), REFERENCE.trendline, LONG_LOOKBACK),
        )
        for expr, expected, lookback in cases:
            with self.subTest(expr=str(expr)):
                self.assert_values_equal(column(expr)[lookback:], expected[lookback:])

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

    def test_matches_the_shared_recursion(self) -> None:
        fields = unnest(ht_phasor("close"))
        self.assert_values_equal(
            fields["in_phase"][SHORT_LOOKBACK:], REFERENCE.in_phase[SHORT_LOOKBACK:]
        )
        self.assert_values_equal(
            fields["quadrature"][SHORT_LOOKBACK:], REFERENCE.quadrature[SHORT_LOOKBACK:]
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
        import math

        fields = unnest(ht_sine("close"))
        phase = column(ht_dcphase("close"))
        expected = [
            None if value is None else math.sin((value + 45.0) * math.pi / 180.0)
            for value in phase
        ]
        self.assert_values_equal(fields["lead_sine"], expected)


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

    def test_matches_the_shared_recursion(self) -> None:
        fields = unnest(mama("close"))
        self.assert_values_equal(
            fields["mama"][SHORT_LOOKBACK:], REFERENCE.mama[SHORT_LOOKBACK:]
        )
        self.assert_values_equal(
            fields["fama"][SHORT_LOOKBACK:], REFERENCE.fama[SHORT_LOOKBACK:]
        )

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
