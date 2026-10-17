import math
from typing import Any, cast

import polars as pl
from _assertions import assert_values_equal
from _data import CLOSE, constant, frame
from pytest import approx, mark, raises

from polars_ta import msw

LENGTH: int = 60
BARS: pl.DataFrame = frame(close=CLOSE[:LENGTH])

# Frozen expectations: the warm-up plus the first live bars.
# fmt: off
MSW_SINE_5: list[float | None] = [
    None, None, None, None, -0.8090169943749473, -0.587785252292474,
    0.8090169943749475, 0.8967360766495205, 0.14379831648484462, -0.8090169943749473,
    -0.07647848980893095, 0.9856083237621691, 0.19688101235068506,
    -0.9246379388312668, -0.761694642422713, 0.18115426001686147,
]
MSW_LEAD_5: list[float | None] = [
    None, None, None, None, -0.9876883405951378, 0.15643446504022973,
    0.9876883405951378, 0.32114694513923636, -0.5980770663205373,
    -0.9876883405951378, 0.6509573703295313, 0.816463237202434, -0.5540509569788138,
    -0.9231174801152227, -0.08043939200917054, 0.8235028884301893,
]
# fmt: on


def field(expr: pl.Expr, name: str, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    return source.select(expr.alias("value")).unnest("value")[name].to_list()


class TestMsw:
    def test_known_sine(self) -> None:
        result = field(msw("close", 5), "sine")
        assert_values_equal(result[: len(MSW_SINE_5)], MSW_SINE_5)

    def test_known_lead(self) -> None:
        result = field(msw("close", 5), "lead")
        assert_values_equal(result[: len(MSW_LEAD_5)], MSW_LEAD_5)

    @mark.parametrize("window", (4, 5, 10))
    def test_warm_up_is_window_minus_one(self, window: int) -> None:
        result = field(msw("close", window), "sine")
        assert result[: window - 1] == [None] * (window - 1)
        assert result[window - 1] is not None

    @mark.parametrize("name", ("sine", "lead"))
    def test_stays_within_the_unit_interval(self, name: str) -> None:
        for value in field(msw("close", 8), name):
            if value is not None:
                assert -1.0 <= value <= 1.0

    def test_matching_cycle_traces_that_cycle(self) -> None:
        period = 8
        wave = [math.sin(2.0 * math.pi * index / period) for index in range(32)]
        bars = pl.DataFrame({"close": wave})
        sine = field(msw("close", period), "sine", bars)
        # One full cycle later the phase comes back round to the same reading.
        assert_values_equal(sine[20:24], sine[28:32])

    def test_lead_sits_an_eighth_of_a_cycle_from_the_sine(self) -> None:
        sine = field(msw("close", 8), "sine")
        lead = field(msw("close", 8), "lead")
        # lead = sin(phase + pi/4), so lead * sqrt(2) - sine recovers cos(phase).
        for first, second in zip(sine[7:], lead[7:]):
            cosine = second * math.sqrt(2.0) - first
            assert first**2 + cosine**2 == approx(1.0, rel=0, abs=1e-9)

    def test_constant_series_holds_one_phase(self) -> None:
        bars = pl.DataFrame({"close": constant(20, 6.0)})
        sine = field(msw("close", 5), "sine", bars)
        assert len(set(sine[4:])) == 1

    def test_invalid_window_raises(self) -> None:
        with raises(ValueError):
            msw("close", 0)

    def test_series_input_is_rejected(self) -> None:
        with raises(TypeError):
            msw(cast(Any, pl.Series("close", CLOSE[:LENGTH])), 5)
