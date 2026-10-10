from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, HIGH, LOW, VOLUME, constant, frame, ramp_down, ramp_up
from pytest import approx, mark, raises

from polars_ta import ao, ppo, pvo, tsi

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
TSI_13_25: list[float | None] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None, None, 12.599668390650313,
    13.085913291626373, 14.312875394835844, 16.306208397915697, 18.349250186730423,
    21.576021687465865, 22.183717534404472, 23.280754248355496, 24.20699951943854,
    24.982276356800057, 25.454738315750195, 23.993513657573896,
]
# fmt: on


def column(expr: pl.Expr, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr).to_series().to_list()


class TestTsi:
    def test_known_values(self) -> None:
        result = column(tsi("close", 13, 25))
        assert_values_equal(result[: len(TSI_13_25)], TSI_13_25)

    @mark.parametrize(("fast", "slow"), ((3, 5), (5, 10), (13, 25)))
    def test_warm_up_covers_both_passes(self, fast: int, slow: int) -> None:
        result = column(tsi("close", fast, slow))
        warm_up = slow + fast - 1
        assert result[:warm_up] == [None] * warm_up
        assert result[warm_up] is not None

    def test_a_rising_series_pins_at_one_hundred(self) -> None:
        bars = pl.DataFrame({"close": ramp_up(40)})
        assert column(tsi("close", 5, 10), bars)[-1] == approx(100.0, rel=0, abs=5e-8)

    def test_a_falling_series_pins_at_minus_one_hundred(self) -> None:
        bars = pl.DataFrame({"close": ramp_down(40, 100.0)})
        assert column(tsi("close", 5, 10), bars)[-1] == approx(-100.0, rel=0, abs=5e-8)

    def test_a_flat_series_reports_zero(self) -> None:
        bars = pl.DataFrame({"close": constant(40, 9.0)})
        assert column(tsi("close", 5, 10), bars)[-1] == approx(0.0, rel=0, abs=5e-8)

    def test_invalid_period_raises(self) -> None:
        with raises(ValueError):
            tsi("close", 0)


class TestAo:
    def test_matches_hand_checked_values(self) -> None:
        middle = ramp_up(10)
        bars = pl.DataFrame(
            {
                "high": [value + 1.0 for value in middle],
                "low": [value - 1.0 for value in middle],
            }
        )
        # Median prices step by one, so a 2-bar average leads a 4-bar one by 1.0.
        assert_values_equal(
            column(ao("high", "low", 2, 4), bars), [None] * 3 + [1.0] * 7
        )

    @mark.parametrize(("fast", "slow"), ((5, 34), (3, 10)))
    def test_warm_up_is_the_slow_period_minus_one(self, fast: int, slow: int) -> None:
        result = column(ao("high", "low", fast, slow))
        assert result[: slow - 1] == [None] * (slow - 1)
        assert result[slow - 1] is not None

    def test_a_flat_series_reads_zero(self) -> None:
        flat = constant(40, 5.0)
        bars = pl.DataFrame({"high": flat, "low": flat})
        assert column(ao("high", "low", 5, 34), bars)[-1] == approx(
            0.0, rel=0, abs=5e-8
        )

    def test_invalid_period_raises(self) -> None:
        with raises(ValueError):
            ao("high", "low", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            ao(cast(Any, pl.Series("high", HIGH[:5])), "low")


class TestPvo:
    def test_is_ppo_read_on_volume(self) -> None:
        assert_values_equal(
            column(pvo("volume", 5, 10)),
            column(ppo("volume", 5, 10, ma_type="ema")),
        )

    def test_defaults_to_exponential_averages(self) -> None:
        simple = column(ppo("volume", 5, 10, ma_type="sma"))
        exponential = column(pvo("volume", 5, 10))
        assert simple[-1] != approx(exponential[-1], rel=0, abs=5e-8)

    def test_a_flat_volume_reads_zero(self) -> None:
        bars = BARS.with_columns(pl.lit(100.0).alias("volume"))
        assert column(pvo("volume", 5, 10), bars)[-1] == approx(0.0, rel=0, abs=5e-8)

    def test_invalid_period_raises(self) -> None:
        with raises(ValueError):
            pvo("volume", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            pvo(cast(Any, pl.Series("volume", VOLUME[:LENGTH])), 5, 10)


class TestAoAgainstLow:
    def test_uses_the_median_price(self) -> None:
        median = [(h + low) / 2.0 for h, low in zip(HIGH[:LENGTH], LOW[:LENGTH])]
        bars = pl.DataFrame({"median": median})
        expected = bars.select(
            pl.col("median").rolling_mean(window_size=5, min_samples=5)
            - pl.col("median").rolling_mean(window_size=34, min_samples=34)
        )
        assert_values_equal(column(ao("high", "low")), expected.to_series().to_list())
