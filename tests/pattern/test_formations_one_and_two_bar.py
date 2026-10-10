"""Hand-built one- and two-bar formations, each paired with a broken variant.

Every recogniser gets the textbook bars it is named for, then the same bars
with the one condition its name is about taken away.
"""

from __future__ import annotations

from _formations import Bar, score

from polars_ta import (
    cdlbelthold,
    cdlclosingmarubozu,
    cdlcounterattack,
    cdldarkcloudcover,
    cdldoji,
    cdldojistar,
    cdldragonflydoji,
    cdlengulfing,
    cdlgravestonedoji,
    cdlhammer,
    cdlhangingman,
    cdlharami,
    cdlharamicross,
    cdlhighwave,
    cdlhomingpigeon,
    cdlinneck,
    cdlinvertedhammer,
    cdlkicking,
    cdlkickingbylength,
    cdllongleggeddoji,
    cdllongline,
    cdlmarubozu,
    cdlmatchinglow,
    cdlonneck,
    cdlpiercing,
    cdlrickshawman,
    cdlseparatinglines,
    cdlshootingstar,
    cdlshortline,
    cdlspinningtop,
    cdltakuri,
    cdlthrusting,
)

# A long white and a long black session, both wider than the baseline body of
# 1.0, used as the first bar of most of the two-bar formations.
LONG_WHITE: Bar = (99.0, 103.2, 98.8, 103.0)
LONG_BLACK: Bar = (103.0, 103.2, 98.8, 99.0)
# The same black session with its shadows trimmed away, for the kickings.
BLACK_MARUBOZU: Bar = (103.0, 103.05, 98.95, 99.0)


class TestDoji:
    def test_a_bar_closing_where_it_opened_is_a_doji(self) -> None:
        assert score(cdldoji, [(100.0, 101.0, 99.0, 100.1)]) == 100

    def test_a_bar_with_a_body_to_speak_of_is_not(self) -> None:
        assert score(cdldoji, [(100.0, 101.0, 99.0, 100.5)]) == 0

    def test_a_doji_gapping_off_a_long_white_bar_is_a_doji_star(self) -> None:
        star: Bar = (104.0, 104.3, 103.8, 104.05)
        assert score(cdldojistar, [LONG_WHITE, star]) == -100

    def test_a_doji_overlapping_the_long_bar_is_not_a_star(self) -> None:
        overlapping: Bar = (103.0, 103.3, 102.8, 103.05)
        assert score(cdldojistar, [LONG_WHITE, overlapping]) == 0

    def test_a_doji_sitting_on_its_high_is_a_dragonfly(self) -> None:
        assert score(cdldragonflydoji, [(101.0, 101.05, 99.0, 101.0)]) == 100

    def test_a_doji_without_the_lower_shadow_is_not_a_dragonfly(self) -> None:
        assert score(cdldragonflydoji, [(101.0, 101.05, 100.95, 101.0)]) == 0

    def test_a_doji_sitting_on_its_low_is_a_gravestone(self) -> None:
        assert score(cdlgravestonedoji, [(99.0, 101.0, 98.95, 99.0)]) == 100

    def test_a_doji_without_the_upper_shadow_is_not_a_gravestone(self) -> None:
        assert score(cdlgravestonedoji, [(99.0, 99.05, 98.95, 99.0)]) == 0

    def test_a_doji_with_a_shadow_longer_than_its_body_is_long_legged(self) -> None:
        assert score(cdllongleggeddoji, [(100.0, 102.0, 99.9, 100.1)]) == 100

    def test_a_doji_with_shadows_shorter_than_its_body_is_not(self) -> None:
        assert score(cdllongleggeddoji, [(100.0, 100.15, 99.95, 100.1)]) == 0

    def test_a_centred_doji_between_two_long_shadows_is_a_rickshaw_man(self) -> None:
        assert score(cdlrickshawman, [(100.0, 102.0, 98.0, 100.1)]) == 100

    def test_a_doji_parked_at_the_top_of_its_range_is_not(self) -> None:
        # Both legs are still long, but the body is nowhere near the midpoint.
        assert score(cdlrickshawman, [(101.5, 102.0, 98.0, 101.6)]) == 0

    def test_a_dragonfly_with_a_very_long_lower_shadow_is_a_takuri(self) -> None:
        assert score(cdltakuri, [(101.15, 101.2, 99.0, 101.0)]) == 100

    def test_a_dragonfly_whose_lower_shadow_is_merely_long_is_not(self) -> None:
        # A 0.25 lower shadow clears the dragonfly bar but not twice the body.
        assert score(cdltakuri, [(101.15, 101.2, 100.75, 101.0)]) == 0


class TestSingleBar:
    def test_a_long_white_bar_opening_on_its_low_is_a_belt_hold(self) -> None:
        assert score(cdlbelthold, [(100.0, 103.0, 99.95, 102.5)]) == 100

    def test_a_long_white_bar_opening_above_its_low_is_not(self) -> None:
        assert score(cdlbelthold, [(100.0, 103.0, 99.0, 102.5)]) == 0

    def test_a_long_white_bar_closing_on_its_high_is_a_closing_marubozu(self) -> None:
        assert score(cdlclosingmarubozu, [(100.0, 102.55, 99.0, 102.5)]) == 100

    def test_a_long_white_bar_closing_below_its_high_is_not(self) -> None:
        assert score(cdlclosingmarubozu, [(100.0, 104.0, 99.0, 102.5)]) == 0

    def test_a_long_bar_with_no_shadow_at_either_end_is_a_marubozu(self) -> None:
        assert score(cdlmarubozu, [(100.0, 102.55, 99.95, 102.5)]) == 100

    def test_a_long_bar_with_an_upper_shadow_is_not_a_marubozu(self) -> None:
        assert score(cdlmarubozu, [(100.0, 104.0, 99.95, 102.5)]) == 0

    def test_a_long_body_with_short_shadows_is_a_long_line(self) -> None:
        assert score(cdllongline, [(100.0, 102.9, 99.6, 102.5)]) == 100

    def test_a_short_body_with_short_shadows_is_not_a_long_line(self) -> None:
        assert score(cdllongline, [(100.0, 100.9, 99.6, 100.5)]) == 0

    def test_a_short_body_with_short_shadows_is_a_short_line(self) -> None:
        assert score(cdlshortline, [(100.0, 100.9, 99.6, 100.5)]) == 100

    def test_a_long_body_with_short_shadows_is_not_a_short_line(self) -> None:
        assert score(cdlshortline, [(100.0, 102.9, 99.6, 102.5)]) == 0

    def test_a_small_body_between_two_very_long_shadows_is_a_high_wave(self) -> None:
        assert score(cdlhighwave, [(100.0, 103.0, 97.0, 100.5)]) == 100

    def test_a_small_body_with_shadows_shorter_than_itself_is_not(self) -> None:
        assert score(cdlhighwave, [(100.0, 100.9, 99.6, 100.5)]) == 0

    def test_a_small_body_with_a_longer_shadow_each_side_is_a_top(self) -> None:
        assert score(cdlspinningtop, [(100.0, 101.2, 99.3, 100.5)]) == 100

    def test_a_small_body_pinned_under_its_high_is_not_a_spinning_top(self) -> None:
        assert score(cdlspinningtop, [(100.0, 100.6, 99.3, 100.5)]) == 0

    def test_a_long_lower_shadow_near_the_prior_low_is_a_hammer(self) -> None:
        decline: Bar = (100.0, 100.2, 98.0, 98.2)
        assert score(cdlhammer, [decline, (98.3, 98.45, 96.5, 98.4)]) == 100

    def test_the_same_bar_without_the_lower_shadow_is_not_a_hammer(self) -> None:
        decline: Bar = (100.0, 100.2, 98.0, 98.2)
        assert score(cdlhammer, [decline, (98.3, 98.45, 98.25, 98.4)]) == 0

    def test_a_long_lower_shadow_at_the_prior_high_is_a_hanging_man(self) -> None:
        advance: Bar = (99.0, 101.0, 98.9, 100.9)
        assert score(cdlhangingman, [advance, (100.7, 100.85, 99.0, 100.8)]) == -100

    def test_the_same_bar_well_below_the_prior_high_is_not(self) -> None:
        advance: Bar = (99.0, 101.0, 98.9, 100.9)
        assert score(cdlhangingman, [advance, (99.0, 99.15, 97.5, 99.1)]) == 0

    def test_a_long_upper_shadow_gapping_down_is_an_inverted_hammer(self) -> None:
        decline: Bar = (102.0, 102.2, 100.0, 100.2)
        assert score(cdlinvertedhammer, [decline, (99.5, 101.5, 99.45, 99.6)]) == 100

    def test_the_same_bar_without_the_gap_is_not_an_inverted_hammer(self) -> None:
        decline: Bar = (102.0, 102.2, 100.0, 100.2)
        assert score(cdlinvertedhammer, [decline, (100.5, 102.4, 100.45, 100.6)]) == 0

    def test_a_long_upper_shadow_gapping_up_is_a_shooting_star(self) -> None:
        advance: Bar = (99.8, 102.0, 99.6, 101.8)
        assert score(cdlshootingstar, [advance, (102.2, 104.0, 102.15, 102.3)]) == -100

    def test_the_same_bar_without_the_gap_is_not_a_shooting_star(self) -> None:
        advance: Bar = (99.8, 102.0, 99.6, 101.8)
        assert score(cdlshootingstar, [advance, (101.0, 102.8, 100.95, 101.1)]) == 0


class TestTwoBar:
    def test_a_white_body_swallowing_a_black_one_is_engulfing(self) -> None:
        black: Bar = (102.0, 102.2, 100.8, 101.0)
        assert score(cdlengulfing, [black, (100.5, 102.8, 100.3, 102.5)]) == 100

    def test_a_white_body_stopping_inside_the_black_one_is_not(self) -> None:
        black: Bar = (102.0, 102.2, 100.8, 101.0)
        assert score(cdlengulfing, [black, (100.5, 102.0, 100.3, 101.5)]) == 0

    def test_a_small_body_inside_a_long_one_is_a_harami(self) -> None:
        assert score(cdlharami, [LONG_WHITE, (101.0, 101.6, 100.4, 101.5)]) == -100

    def test_a_small_body_poking_above_the_long_one_is_not(self) -> None:
        assert score(cdlharami, [LONG_WHITE, (102.8, 103.8, 102.6, 103.3)]) == 0

    def test_a_contained_doji_is_a_harami_cross(self) -> None:
        assert score(cdlharamicross, [LONG_WHITE, (101.0, 101.6, 100.4, 101.1)]) == -100

    def test_a_contained_bar_with_a_real_body_is_only_a_harami(self) -> None:
        assert score(cdlharamicross, [LONG_WHITE, (101.0, 101.6, 100.4, 101.5)]) == 0

    def test_a_black_bar_opening_clear_of_a_white_high_is_dark_cloud(self) -> None:
        cloud: Bar = (103.5, 103.6, 100.3, 100.5)
        assert score(cdldarkcloudcover, [LONG_WHITE, cloud]) == -100

    def test_a_black_bar_opening_within_the_white_range_is_not(self) -> None:
        assert score(cdldarkcloudcover, [LONG_WHITE, (103.0, 103.1, 100.3, 100.5)]) == 0

    def test_a_white_bar_closing_past_the_black_midpoint_is_piercing(self) -> None:
        assert score(cdlpiercing, [LONG_BLACK, (98.5, 101.8, 98.4, 101.5)]) == 100

    def test_a_white_bar_stopping_short_of_the_midpoint_is_not(self) -> None:
        assert score(cdlpiercing, [LONG_BLACK, (98.5, 100.8, 98.4, 100.5)]) == 0

    def test_two_long_opposite_bars_closing_level_are_a_counterattack(self) -> None:
        assert score(cdlcounterattack, [LONG_BLACK, (96.0, 99.2, 95.8, 99.0)]) == 100

    def test_two_long_opposite_bars_closing_apart_are_not(self) -> None:
        assert score(cdlcounterattack, [LONG_BLACK, (96.0, 100.2, 95.8, 100.0)]) == 0

    def test_two_opposite_bars_opening_level_are_separating_lines(self) -> None:
        reply: Bar = (103.0, 106.2, 102.95, 106.0)
        assert score(cdlseparatinglines, [LONG_BLACK, reply]) == 100

    def test_two_opposite_bars_opening_apart_are_not(self) -> None:
        reply: Bar = (101.0, 104.2, 100.95, 104.0)
        assert score(cdlseparatinglines, [LONG_BLACK, reply]) == 0

    def test_two_opposite_marubozu_split_by_a_gap_are_a_kicking(self) -> None:
        white: Bar = (103.2, 106.25, 103.15, 106.2)
        assert score(cdlkicking, [BLACK_MARUBOZU, white]) == 100

    def test_two_opposite_marubozu_that_overlap_are_not(self) -> None:
        white: Bar = (102.0, 105.05, 101.95, 105.0)
        assert score(cdlkicking, [BLACK_MARUBOZU, white]) == 0

    def test_a_kicking_is_scored_by_length_from_its_longer_bar(self) -> None:
        # The black bar's body of 4.0 beats the white reply's 3.0, so the
        # reading is black even though the later bar is white.
        white: Bar = (103.2, 106.25, 103.15, 106.2)
        assert score(cdlkickingbylength, [BLACK_MARUBOZU, white]) == -100

    def test_kicking_by_length_needs_the_gap_too(self) -> None:
        white: Bar = (102.0, 105.05, 101.95, 105.0)
        assert score(cdlkickingbylength, [BLACK_MARUBOZU, white]) == 0

    def test_two_black_bars_closing_at_one_price_are_a_matching_low(self) -> None:
        black: Bar = (102.0, 102.2, 99.8, 100.0)
        assert score(cdlmatchinglow, [black, (101.0, 101.2, 99.9, 100.0)]) == 100

    def test_two_black_bars_closing_at_different_prices_are_not(self) -> None:
        black: Bar = (102.0, 102.2, 99.8, 100.0)
        assert score(cdlmatchinglow, [black, (101.0, 101.2, 98.8, 99.0)]) == 0


class TestNecklines:
    def test_a_rally_closing_just_above_the_black_close_is_an_in_neck(self) -> None:
        assert score(cdlinneck, [LONG_BLACK, (98.5, 99.1, 98.4, 99.05)]) == -100

    def test_a_rally_closing_well_into_the_black_body_is_not(self) -> None:
        assert score(cdlinneck, [LONG_BLACK, (98.5, 101.2, 98.4, 101.0)]) == 0

    def test_a_rally_closing_level_with_the_black_low_is_an_on_neck(self) -> None:
        assert score(cdlonneck, [LONG_BLACK, (98.5, 98.9, 98.4, 98.8)]) == -100

    def test_a_rally_closing_above_the_black_low_is_not(self) -> None:
        assert score(cdlonneck, [LONG_BLACK, (98.5, 99.1, 98.4, 99.05)]) == 0

    def test_a_rally_into_the_lower_half_of_the_black_body_is_thrusting(self) -> None:
        assert score(cdlthrusting, [LONG_BLACK, (98.5, 100.7, 98.4, 100.5)]) == -100

    def test_a_rally_past_the_black_midpoint_is_not_thrusting(self) -> None:
        assert score(cdlthrusting, [LONG_BLACK, (98.5, 101.7, 98.4, 101.5)]) == 0

    def test_a_small_black_bar_inside_a_long_one_is_a_homing_pigeon(self) -> None:
        assert score(cdlhomingpigeon, [LONG_BLACK, (102.0, 102.2, 101.3, 101.5)]) == 100

    def test_a_small_white_bar_inside_a_long_black_one_is_not(self) -> None:
        assert score(cdlhomingpigeon, [LONG_BLACK, (101.5, 102.2, 101.3, 102.0)]) == 0
