"""Hand-built multi-bar formations, each paired with its defining break."""

from _formations import BASELINE, Bar, score

from polars_ta import (
    cdl2crows,
    cdl3blackcrows,
    cdl3inside,
    cdl3linestrike,
    cdl3outside,
    cdl3starsinsouth,
    cdl3whitesoldiers,
    cdlabandonedbaby,
    cdladvanceblock,
    cdlbreakaway,
    cdlconcealbabyswall,
    cdleveningdojistar,
    cdleveningstar,
    cdlgapsidesidewhite,
    cdlhikkake,
    cdlhikkakemod,
    cdlidentical3crows,
    cdlladderbottom,
    cdlmathold,
    cdlmorningdojistar,
    cdlmorningstar,
    cdlrisefall3methods,
    cdlstalledpattern,
    cdlsticksandwich,
    cdltasukigap,
    cdltristar,
    cdlunique3river,
    cdlupsidegap2crows,
    cdlxsidegap3methods,
)

# Bars that open several formations: body 3.0 against the baseline's 1.0.
LONG_WHITE: Bar = (100.0, 103.2, 99.8, 103.0)
LONG_BLACK: Bar = (103.0, 103.2, 99.8, 100.0)
# ... and body 4.0, for the five-bar continuations.
TALL_WHITE: Bar = (100.0, 104.2, 99.8, 104.0)

MORNING_LONG_BLACK: Bar = (101.0, 101.3, 96.7, 97.0)
MORNING_STAR: Bar = (96.5, 96.7, 96.0, 96.3)
MORNING_STAR_TOUCHING: Bar = (97.5, 97.7, 97.0, 97.3)
MORNING_RALLY: Bar = (96.6, 99.7, 96.4, 99.5)

EVENING_LONG_WHITE: Bar = (101.0, 105.3, 100.7, 105.0)
EVENING_STAR: Bar = (105.5, 106.0, 105.3, 105.7)
EVENING_STAR_TOUCHING: Bar = (104.8, 105.2, 104.6, 105.0)
EVENING_DECLINE: Bar = (105.6, 105.8, 102.3, 102.5)

MORNING_DOJI: Bar = (96.4, 96.7, 96.1, 96.4)
MORNING_SMALL_BODY: Bar = (96.6, 96.8, 96.0, 96.1)
MORNING_DOJI_RALLY: Bar = (96.5, 99.7, 96.3, 99.5)

EVENING_DOJI: Bar = (105.6, 105.9, 105.3, 105.6)
EVENING_SMALL_BODY: Bar = (105.6, 105.9, 105.0, 105.1)
EVENING_DOJI_DECLINE: Bar = (105.5, 105.7, 102.3, 102.5)

ABANDONED_DOJI: Bar = (96.0, 96.2, 95.8, 96.0)
ABANDONED_RALLY: Bar = (96.5, 99.7, 96.4, 99.5)
ABANDONED_RALLY_OVERLAPPING: Bar = (96.5, 99.7, 96.1, 99.5)

TRISTAR_FIRST: Bar = (100.0, 100.2, 99.8, 100.0)
TRISTAR_MIDDLE: Bar = (99.0, 99.2, 98.8, 99.0)
TRISTAR_LAST: Bar = (99.5, 99.7, 99.3, 99.5)
TRISTAR_LAST_WITH_BODY: Bar = (99.5, 100.5, 99.3, 100.3)

SOUTH_FIRST: Bar = (101.0, 101.2, 96.0, 99.0)
SOUTH_SECOND: Bar = (100.0, 100.1, 97.0, 98.6)
SOUTH_THIRD: Bar = (99.0, 99.1, 98.4, 98.5)
SOUTH_THIRD_WHITE: Bar = (98.5, 99.1, 98.4, 99.0)

CROWS_GAPPED: Bar = (104.5, 104.7, 103.3, 103.5)
CROWS_SECOND: Bar = (104.0, 104.2, 101.3, 101.5)
CROWS_UNGAPPED: Bar = (103.0, 103.2, 101.8, 102.0)
CROWS_SECOND_LOWER: Bar = (102.5, 102.7, 101.3, 101.5)

BLACK_CROWS_FIRST: Bar = (102.5, 102.6, 100.4, 100.5)
BLACK_CROWS_SECOND: Bar = (102.0, 102.1, 98.9, 99.0)
BLACK_CROWS_THIRD: Bar = (101.0, 101.1, 97.4, 97.5)
BLACK_CROWS_THIRD_OFF_ITS_LOW: Bar = (101.0, 101.1, 95.0, 97.5)

IDENTICAL_FIRST: Bar = (102.0, 102.1, 99.9, 100.0)
IDENTICAL_SECOND: Bar = (100.0, 100.1, 97.9, 98.0)
IDENTICAL_THIRD: Bar = (98.0, 98.1, 95.9, 96.0)
IDENTICAL_THIRD_GAPPED: Bar = (98.5, 98.6, 95.9, 96.0)

SOLDIER_FIRST: Bar = (100.0, 101.3, 99.9, 101.2)
SOLDIER_SECOND: Bar = (101.0, 102.7, 100.9, 102.6)
SOLDIER_THIRD: Bar = (102.0, 104.1, 101.9, 104.0)
SOLDIER_SECOND_BLACK: Bar = (101.5, 101.6, 101.2, 101.3)
SOLDIER_THIRD_AFTER_BLACK: Bar = (101.6, 104.1, 101.5, 104.0)

BLOCK_SECOND: Bar = (102.0, 104.2, 101.8, 104.0)
BLOCK_THIRD: Bar = (103.0, 105.0, 102.9, 104.2)
BLOCK_THIRD_UNCHECKED: Bar = (103.0, 105.2, 102.9, 105.0)

STALLED_SECOND: Bar = (102.0, 105.1, 101.9, 105.0)
STALLED_THIRD: Bar = (105.3, 105.6, 105.2, 105.5)
STALLED_THIRD_LONG: Bar = (105.3, 107.6, 105.2, 107.5)

GAP_CROWS_FIRST: Bar = (104.5, 104.7, 103.9, 104.0)
GAP_CROWS_SECOND: Bar = (105.0, 105.2, 103.3, 103.5)
GAP_CROWS_SECOND_SHALLOW: Bar = (105.0, 105.2, 104.1, 104.2)

INSIDE_HARAMI: Bar = (101.0, 102.2, 100.8, 102.0)
INSIDE_HARAMI_SPILLING: Bar = (99.5, 100.7, 99.3, 100.5)
INSIDE_CONFIRMATION: Bar = (102.0, 104.2, 101.8, 104.0)

OUTSIDE_FIRST: Bar = (102.0, 102.2, 100.8, 101.0)
OUTSIDE_ENGULFING: Bar = (100.5, 102.7, 100.3, 102.5)
OUTSIDE_CONTAINED: Bar = (101.2, 101.9, 101.1, 101.8)
OUTSIDE_CONFIRMATION: Bar = (102.5, 103.7, 102.3, 103.5)

STRIKE_FIRST: Bar = (100.0, 101.2, 99.8, 101.0)
STRIKE_SECOND: Bar = (100.5, 102.2, 100.3, 102.0)
STRIKE_THIRD: Bar = (101.5, 103.2, 101.3, 103.0)
STRIKE_FOURTH: Bar = (103.5, 103.7, 99.3, 99.5)
STRIKE_FOURTH_FALLING_SHORT: Bar = (103.5, 103.7, 100.3, 100.5)

RIVER_SECOND: Bar = (102.0, 102.1, 99.0, 100.5)
RIVER_SECOND_NO_NEW_LOW: Bar = (102.0, 102.1, 99.9, 100.5)
RIVER_THIRD: Bar = (100.0, 100.9, 99.9, 100.8)

SANDWICH_BLACK: Bar = (101.0, 101.2, 98.8, 99.0)
SANDWICH_WHITE: Bar = (99.5, 101.2, 99.3, 101.0)
SANDWICH_BLACK_CLOSING_LOWER: Bar = (101.0, 101.2, 98.3, 98.5)

SWALLOW_FIRST: Bar = (101.0, 101.0, 99.0, 99.0)
SWALLOW_SECOND: Bar = (99.0, 99.0, 97.0, 97.0)
SWALLOW_THIRD: Bar = (96.5, 97.5, 95.8, 96.0)
SWALLOW_FOURTH: Bar = (97.0, 97.8, 95.3, 95.5)
SWALLOW_FOURTH_SHORT: Bar = (97.0, 97.2, 95.3, 95.5)

SIDE_BY_SIDE_FIRST: Bar = (102.0, 103.2, 101.8, 103.0)
SIDE_BY_SIDE_SECOND: Bar = (102.0, 103.2, 101.8, 103.0)
BASE_REACHING_UP: Bar = (100.0, 102.7, 99.8, 102.5)

TASUKI_WHITE: Bar = (102.0, 103.2, 101.8, 103.0)
TASUKI_BLACK: Bar = (102.5, 102.7, 101.3, 101.5)
TASUKI_BLACK_FILLING: Bar = (102.2, 102.4, 100.7, 100.9)

XGAP_SECOND: Bar = (102.0, 103.2, 101.8, 103.0)
XGAP_SECOND_UNGAPPED: Bar = (101.0, 102.2, 100.8, 102.0)
XGAP_THIRD: Bar = (102.5, 102.7, 100.3, 100.5)
XGAP_THIRD_AFTER_OVERLAP: Bar = (101.5, 101.7, 100.3, 100.5)

BREAKAWAY_GAPPED: Bar = (104.0, 104.7, 103.8, 104.5)
BREAKAWAY_GAPLESS: Bar = (102.8, 104.7, 102.6, 104.5)
BREAKAWAY_DRIFT: Bar = (104.6, 105.2, 104.2, 105.0)
BREAKAWAY_DRIFT_ON: Bar = (105.0, 105.7, 104.8, 105.5)
BREAKAWAY_RETURN: Bar = (105.5, 105.7, 103.3, 103.5)

RISE_PAUSE_ONE: Bar = (103.5, 103.7, 102.8, 103.0)
RISE_PAUSE_TWO: Bar = (103.0, 103.2, 102.3, 102.5)
RISE_PAUSE_THREE: Bar = (102.5, 102.7, 101.8, 102.0)
RISE_RESUMPTION: Bar = (102.5, 105.2, 102.3, 105.0)
RISE_RESUMPTION_FALLING_SHORT: Bar = (102.5, 104.0, 102.3, 103.8)

MAT_GAPPED_BLACK: Bar = (105.0, 105.2, 104.3, 104.5)
MAT_DRIFT_ONE: Bar = (104.2, 104.4, 103.3, 103.5)
MAT_DRIFT_TWO: Bar = (103.6, 103.8, 102.8, 103.0)
MAT_DRIFT_TWO_DEEP: Bar = (102.5, 103.8, 101.3, 101.5)
MAT_BREAKOUT: Bar = (103.5, 106.2, 103.3, 106.0)

LADDER_FIRST: Bar = (104.0, 104.2, 101.8, 102.0)
LADDER_SECOND: Bar = (103.0, 103.2, 100.8, 101.0)
LADDER_THIRD: Bar = (102.0, 102.2, 99.8, 100.0)
LADDER_THIRD_OPENING_HIGHER: Bar = (103.5, 103.7, 99.8, 100.0)
LADDER_PROBE: Bar = (100.5, 101.5, 99.3, 99.5)
LADDER_RALLY: Bar = (100.8, 102.7, 100.6, 102.5)

HIKKAKE_WIDE: Bar = (100.0, 102.0, 98.0, 101.0)
HIKKAKE_INSIDE: Bar = (100.0, 101.0, 99.0, 100.5)
HIKKAKE_OUTSIDE: Bar = (100.0, 102.5, 99.0, 100.5)
HIKKAKE_BREAK: Bar = (100.0, 100.5, 98.5, 99.0)
HIKKAKE_CONFIRMATION: Bar = (99.5, 101.8, 99.3, 101.5)

MOD_WIDE: Bar = (100.0, 103.0, 97.0, 100.0)
MOD_CLOSING_LOW: Bar = (101.5, 102.0, 98.0, 98.2)
MOD_CLOSING_MID: Bar = (101.5, 102.0, 98.0, 100.0)
MOD_INSIDE: Bar = (99.0, 101.0, 98.5, 100.0)
MOD_BREAK: Bar = (99.0, 100.5, 98.0, 98.5)
MOD_CONFIRMATION: Bar = (99.0, 101.8, 98.8, 101.5)


class TestStars:
    def test_a_morning_star_turns_a_long_black_bar_around(self) -> None:
        bars = [MORNING_LONG_BLACK, MORNING_STAR, MORNING_RALLY]
        assert score(cdlmorningstar, bars) == 100

    def test_a_star_that_touches_the_black_body_is_no_morning_star(self) -> None:
        bars = [MORNING_LONG_BLACK, MORNING_STAR_TOUCHING, MORNING_RALLY]
        assert score(cdlmorningstar, bars) == 0

    def test_an_evening_star_turns_a_long_white_bar_around(self) -> None:
        bars = [EVENING_LONG_WHITE, EVENING_STAR, EVENING_DECLINE]
        assert score(cdleveningstar, bars) == -100

    def test_a_star_that_touches_the_white_body_is_no_evening_star(self) -> None:
        bars = [EVENING_LONG_WHITE, EVENING_STAR_TOUCHING, EVENING_DECLINE]
        assert score(cdleveningstar, bars) == 0

    def test_a_morning_doji_star_needs_only_a_doji_in_the_middle(self) -> None:
        bars = [MORNING_LONG_BLACK, MORNING_DOJI, MORNING_DOJI_RALLY]
        assert score(cdlmorningdojistar, bars) == 100

    def test_a_middle_bar_with_a_body_is_no_morning_doji_star(self) -> None:
        bars = [MORNING_LONG_BLACK, MORNING_SMALL_BODY, MORNING_DOJI_RALLY]
        assert score(cdlmorningdojistar, bars) == 0

    def test_an_evening_doji_star_needs_only_a_doji_in_the_middle(self) -> None:
        bars = [EVENING_LONG_WHITE, EVENING_DOJI, EVENING_DOJI_DECLINE]
        assert score(cdleveningdojistar, bars) == -100

    def test_a_middle_bar_with_a_body_is_no_evening_doji_star(self) -> None:
        bars = [EVENING_LONG_WHITE, EVENING_SMALL_BODY, EVENING_DOJI_DECLINE]
        assert score(cdleveningdojistar, bars) == 0

    def test_an_abandoned_baby_is_a_doji_with_clear_air_on_both_sides(self) -> None:
        bars = [MORNING_LONG_BLACK, ABANDONED_DOJI, ABANDONED_RALLY]
        assert score(cdlabandonedbaby, bars) == 100

    def test_a_doji_the_rally_reaches_back_over_is_not_abandoned(self) -> None:
        bars = [MORNING_LONG_BLACK, ABANDONED_DOJI, ABANDONED_RALLY_OVERLAPPING]
        assert score(cdlabandonedbaby, bars) == 0

    def test_a_tristar_is_three_doji_with_the_middle_one_gapped_away(self) -> None:
        bars = [TRISTAR_FIRST, TRISTAR_MIDDLE, TRISTAR_LAST]
        assert score(cdltristar, bars) == 100

    def test_a_third_bar_with_a_body_is_no_tristar(self) -> None:
        bars = [TRISTAR_FIRST, TRISTAR_MIDDLE, TRISTAR_LAST_WITH_BODY]
        assert score(cdltristar, bars) == 0

    def test_three_stars_in_the_south_shrink_into_the_second_bar(self) -> None:
        bars = [SOUTH_FIRST, SOUTH_SECOND, SOUTH_THIRD]
        assert score(cdl3starsinsouth, bars) == 100

    def test_a_white_third_bar_is_no_three_stars_in_the_south(self) -> None:
        bars = [SOUTH_FIRST, SOUTH_SECOND, SOUTH_THIRD_WHITE]
        assert score(cdl3starsinsouth, bars) == 0


class TestCrowsAndSoldiers:
    def test_two_crows_fill_the_gap_the_white_bar_left(self) -> None:
        bars = [LONG_WHITE, CROWS_GAPPED, CROWS_SECOND]
        assert score(cdl2crows, bars) == -100

    def test_crows_that_never_gapped_away_score_nothing(self) -> None:
        bars = [LONG_WHITE, CROWS_UNGAPPED, CROWS_SECOND_LOWER]
        assert score(cdl2crows, bars) == 0

    def test_three_black_crows_step_down_closing_on_their_lows(self) -> None:
        bars = [LONG_WHITE, BLACK_CROWS_FIRST, BLACK_CROWS_SECOND, BLACK_CROWS_THIRD]
        assert score(cdl3blackcrows, bars) == -100

    def test_a_crow_closing_well_off_its_low_breaks_the_run(self) -> None:
        bars = [
            LONG_WHITE,
            BLACK_CROWS_FIRST,
            BLACK_CROWS_SECOND,
            BLACK_CROWS_THIRD_OFF_ITS_LOW,
        ]
        assert score(cdl3blackcrows, bars) == 0

    def test_identical_three_crows_each_open_at_the_prior_close(self) -> None:
        bars = [IDENTICAL_FIRST, IDENTICAL_SECOND, IDENTICAL_THIRD]
        assert score(cdlidentical3crows, bars) == -100

    def test_a_crow_opening_away_from_the_prior_close_is_not_identical(self) -> None:
        bars = [IDENTICAL_FIRST, IDENTICAL_SECOND, IDENTICAL_THIRD_GAPPED]
        assert score(cdlidentical3crows, bars) == 0

    def test_three_white_soldiers_advance_in_even_strides(self) -> None:
        bars = [SOLDIER_FIRST, SOLDIER_SECOND, SOLDIER_THIRD]
        assert score(cdl3whitesoldiers, bars) == 100

    def test_a_black_middle_bar_is_no_white_soldier(self) -> None:
        bars = [SOLDIER_FIRST, SOLDIER_SECOND_BLACK, SOLDIER_THIRD_AFTER_BLACK]
        assert score(cdl3whitesoldiers, bars) == 0

    def test_an_advance_block_loses_ground_to_a_long_upper_shadow(self) -> None:
        bars = [LONG_WHITE, BLOCK_SECOND, BLOCK_THIRD]
        assert score(cdladvanceblock, bars) == -100

    def test_an_advance_that_never_weakens_is_no_block(self) -> None:
        bars = [LONG_WHITE, BLOCK_SECOND, BLOCK_THIRD_UNCHECKED]
        assert score(cdladvanceblock, bars) == 0

    def test_a_stalled_pattern_ends_on_a_small_body_riding_on_top(self) -> None:
        bars = [LONG_WHITE, STALLED_SECOND, STALLED_THIRD]
        assert score(cdlstalledpattern, bars) == -100

    def test_a_long_third_body_means_the_advance_has_not_stalled(self) -> None:
        bars = [LONG_WHITE, STALLED_SECOND, STALLED_THIRD_LONG]
        assert score(cdlstalledpattern, bars) == 0

    def test_an_upside_gap_is_met_by_two_crows_the_second_engulfing(self) -> None:
        bars = [LONG_WHITE, GAP_CROWS_FIRST, GAP_CROWS_SECOND]
        assert score(cdlupsidegap2crows, bars) == -100

    def test_a_second_crow_that_swallows_nothing_scores_nothing(self) -> None:
        bars = [LONG_WHITE, GAP_CROWS_FIRST, GAP_CROWS_SECOND_SHALLOW]
        assert score(cdlupsidegap2crows, bars) == 0


class TestThreeBar:
    def test_three_inside_up_resolves_a_harami_upwards(self) -> None:
        bars = [LONG_BLACK, INSIDE_HARAMI, INSIDE_CONFIRMATION]
        assert score(cdl3inside, bars) == 100

    def test_a_middle_bar_outside_the_first_body_is_not_inside(self) -> None:
        bars = [LONG_BLACK, INSIDE_HARAMI_SPILLING, INSIDE_CONFIRMATION]
        assert score(cdl3inside, bars) == 0

    def test_three_outside_up_extends_an_engulfing_pair(self) -> None:
        bars = [OUTSIDE_FIRST, OUTSIDE_ENGULFING, OUTSIDE_CONFIRMATION]
        assert score(cdl3outside, bars) == 100

    def test_a_middle_bar_that_engulfs_nothing_is_not_outside(self) -> None:
        bars = [OUTSIDE_FIRST, OUTSIDE_CONTAINED, OUTSIDE_CONFIRMATION]
        assert score(cdl3outside, bars) == 0

    def test_a_three_line_strike_undoes_the_whole_white_run(self) -> None:
        bars = [STRIKE_FIRST, STRIKE_SECOND, STRIKE_THIRD, STRIKE_FOURTH]
        assert score(cdl3linestrike, bars) == 100

    def test_a_fourth_bar_stopping_inside_the_run_does_not_strike(self) -> None:
        bars = [
            STRIKE_FIRST,
            STRIKE_SECOND,
            STRIKE_THIRD,
            STRIKE_FOURTH_FALLING_SHORT,
        ]
        assert score(cdl3linestrike, bars) == 0

    def test_a_unique_3_river_digs_out_a_low_nobody_follows(self) -> None:
        bars = [LONG_BLACK, RIVER_SECOND, RIVER_THIRD]
        assert score(cdlunique3river, bars) == 100

    def test_a_second_bar_without_a_new_low_is_no_unique_3_river(self) -> None:
        bars = [LONG_BLACK, RIVER_SECOND_NO_NEW_LOW, RIVER_THIRD]
        assert score(cdlunique3river, bars) == 0

    def test_a_stick_sandwich_closes_twice_at_the_same_price(self) -> None:
        bars = [SANDWICH_BLACK, SANDWICH_WHITE, SANDWICH_BLACK]
        assert score(cdlsticksandwich, bars) == 100

    def test_closes_that_do_not_match_make_no_sandwich(self) -> None:
        bars = [SANDWICH_BLACK, SANDWICH_WHITE, SANDWICH_BLACK_CLOSING_LOWER]
        assert score(cdlsticksandwich, bars) == 0

    def test_a_concealing_baby_swallow_covers_the_third_bar_whole(self) -> None:
        bars = [SWALLOW_FIRST, SWALLOW_SECOND, SWALLOW_THIRD, SWALLOW_FOURTH]
        assert score(cdlconcealbabyswall, bars) == 100

    def test_a_fourth_bar_that_swallows_nothing_scores_nothing(self) -> None:
        bars = [SWALLOW_FIRST, SWALLOW_SECOND, SWALLOW_THIRD, SWALLOW_FOURTH_SHORT]
        assert score(cdlconcealbabyswall, bars) == 0


class TestGaps:
    def test_side_by_side_white_lines_hold_the_gap_open(self) -> None:
        bars = [BASELINE, SIDE_BY_SIDE_FIRST, SIDE_BY_SIDE_SECOND]
        assert score(cdlgapsidesidewhite, bars) == 100

    def test_white_lines_without_a_gap_below_them_score_nothing(self) -> None:
        bars = [BASE_REACHING_UP, SIDE_BY_SIDE_FIRST, SIDE_BY_SIDE_SECOND]
        assert score(cdlgapsidesidewhite, bars) == 0

    def test_a_tasuki_gap_survives_the_bar_that_probes_it(self) -> None:
        bars = [BASELINE, TASUKI_WHITE, TASUKI_BLACK]
        assert score(cdltasukigap, bars) == 100

    def test_a_probe_that_closes_the_gap_is_no_tasuki(self) -> None:
        bars = [BASELINE, TASUKI_WHITE, TASUKI_BLACK_FILLING]
        assert score(cdltasukigap, bars) == 0

    def test_an_upside_gap_three_methods_is_bridged_by_one_black_bar(self) -> None:
        bars = [BASELINE, XGAP_SECOND, XGAP_THIRD]
        assert score(cdlxsidegap3methods, bars) == 100

    def test_two_white_bars_without_a_gap_bridge_nothing(self) -> None:
        bars = [BASELINE, XGAP_SECOND_UNGAPPED, XGAP_THIRD_AFTER_OVERLAP]
        assert score(cdlxsidegap3methods, bars) == 0

    def test_a_breakaway_closes_back_inside_the_gap_it_opened(self) -> None:
        bars = [
            LONG_WHITE,
            BREAKAWAY_GAPPED,
            BREAKAWAY_DRIFT,
            BREAKAWAY_DRIFT_ON,
            BREAKAWAY_RETURN,
        ]
        assert score(cdlbreakaway, bars) == -100

    def test_a_run_that_never_gapped_away_cannot_break_away(self) -> None:
        bars = [
            LONG_WHITE,
            BREAKAWAY_GAPLESS,
            BREAKAWAY_DRIFT,
            BREAKAWAY_DRIFT_ON,
            BREAKAWAY_RETURN,
        ]
        assert score(cdlbreakaway, bars) == 0


class TestContinuation:
    def test_rising_three_methods_resumes_past_the_first_close(self) -> None:
        bars = [
            TALL_WHITE,
            RISE_PAUSE_ONE,
            RISE_PAUSE_TWO,
            RISE_PAUSE_THREE,
            RISE_RESUMPTION,
        ]
        assert score(cdlrisefall3methods, bars) == 100

    def test_a_last_bar_short_of_the_first_close_resumes_nothing(self) -> None:
        bars = [
            TALL_WHITE,
            RISE_PAUSE_ONE,
            RISE_PAUSE_TWO,
            RISE_PAUSE_THREE,
            RISE_RESUMPTION_FALLING_SHORT,
        ]
        assert score(cdlrisefall3methods, bars) == 0

    def test_a_mat_hold_gives_back_little_before_the_new_high(self) -> None:
        bars = [
            TALL_WHITE,
            MAT_GAPPED_BLACK,
            MAT_DRIFT_ONE,
            MAT_DRIFT_TWO,
            MAT_BREAKOUT,
        ]
        assert score(cdlmathold, bars) == 100

    def test_a_drift_through_the_penetration_floor_does_not_hold(self) -> None:
        bars = [
            TALL_WHITE,
            MAT_GAPPED_BLACK,
            MAT_DRIFT_ONE,
            MAT_DRIFT_TWO_DEEP,
            MAT_BREAKOUT,
        ]
        assert score(cdlmathold, bars) == 0

    def test_a_ladder_bottom_rallies_over_the_probing_bar(self) -> None:
        bars = [
            LADDER_FIRST,
            LADDER_SECOND,
            LADDER_THIRD,
            LADDER_PROBE,
            LADDER_RALLY,
        ]
        assert score(cdlladderbottom, bars) == 100

    def test_a_rung_that_opens_higher_breaks_the_ladder(self) -> None:
        bars = [
            LADDER_FIRST,
            LADDER_SECOND,
            LADDER_THIRD_OPENING_HIGHER,
            LADDER_PROBE,
            LADDER_RALLY,
        ]
        assert score(cdlladderbottom, bars) == 0


class TestHikkake:
    def test_a_break_below_an_inside_bar_scores_then_is_confirmed(self) -> None:
        bars = [HIKKAKE_WIDE, HIKKAKE_INSIDE, HIKKAKE_BREAK, HIKKAKE_CONFIRMATION]
        assert score(cdlhikkake, bars, index=2) == 100
        assert score(cdlhikkake, bars) == 200

    def test_a_middle_bar_that_is_not_inside_scores_nothing(self) -> None:
        bars = [HIKKAKE_WIDE, HIKKAKE_OUTSIDE, HIKKAKE_BREAK, HIKKAKE_CONFIRMATION]
        assert score(cdlhikkake, bars, index=2) == 0

    def test_a_modified_hikkake_needs_the_second_bar_to_close_low(self) -> None:
        bars = [MOD_WIDE, MOD_CLOSING_LOW, MOD_INSIDE, MOD_BREAK, MOD_CONFIRMATION]
        assert score(cdlhikkakemod, bars, index=3) == 100
        assert score(cdlhikkakemod, bars) == 200

    def test_a_second_bar_closing_mid_range_is_no_modified_hikkake(self) -> None:
        bars = [MOD_WIDE, MOD_CLOSING_MID, MOD_INSIDE, MOD_BREAK, MOD_CONFIRMATION]
        assert score(cdlhikkakemod, bars, index=3) == 0
