from typing import Any, Callable, cast

import polars as pl
from _data import frame
from pytest import mark, raises

import polars_ta as ta
from polars_ta.pattern import __all__ as PATTERN_NAMES
from polars_ta.pattern._candles import PATTERN_DTYPE

BARS: pl.DataFrame = frame()
ALLOWED: set[int] = {-200, -100, 0, 100, 200}

# How many leading bars each recogniser needs before it can report anything.
# These are TA-Lib's lookbacks: the longest averaging period a pattern consults
# plus the bars it spans.
# fmt: off
WARM_UP: dict[str, int] = {
    "cdl2crows": 12, "cdl3blackcrows": 13, "cdl3inside": 12, "cdl3linestrike": 8,
    "cdl3outside": 3, "cdl3starsinsouth": 12, "cdl3whitesoldiers": 12,
    "cdlabandonedbaby": 12, "cdladvanceblock": 12, "cdlbelthold": 10,
    "cdlbreakaway": 14, "cdlclosingmarubozu": 10, "cdlconcealbabyswall": 13,
    "cdlcounterattack": 11, "cdldarkcloudcover": 11, "cdldoji": 10,
    "cdldojistar": 11, "cdldragonflydoji": 10, "cdlengulfing": 2,
    "cdleveningdojistar": 12, "cdleveningstar": 12, "cdlgapsidesidewhite": 7,
    "cdlgravestonedoji": 10, "cdlhammer": 11, "cdlhangingman": 11, "cdlharami": 11,
    "cdlharamicross": 11, "cdlhighwave": 10, "cdlhikkake": 5, "cdlhikkakemod": 10,
    "cdlhomingpigeon": 11, "cdlidentical3crows": 12, "cdlinneck": 11,
    "cdlinvertedhammer": 11, "cdlkicking": 11, "cdlkickingbylength": 11,
    "cdlladderbottom": 14, "cdllongleggeddoji": 10, "cdllongline": 10,
    "cdlmarubozu": 10, "cdlmatchinglow": 6, "cdlmathold": 14,
    "cdlmorningdojistar": 12, "cdlmorningstar": 12, "cdlonneck": 11,
    "cdlpiercing": 11, "cdlrickshawman": 10, "cdlrisefall3methods": 14,
    "cdlseparatinglines": 11, "cdlshootingstar": 11, "cdlshortline": 10,
    "cdlspinningtop": 10, "cdlstalledpattern": 12, "cdlsticksandwich": 7,
    "cdltakuri": 10, "cdltasukigap": 7, "cdlthrusting": 11, "cdltristar": 12,
    "cdlunique3river": 12, "cdlupsidegap2crows": 12, "cdlxsidegap3methods": 2,
}
# fmt: on

# TA-Lib exposes a penetration depth on these; the rest take only the bars.
PENETRATION: dict[str, float] = {
    "cdlabandonedbaby": 0.3,
    "cdldarkcloudcover": 0.5,
    "cdleveningdojistar": 0.3,
    "cdleveningstar": 0.3,
    "cdlmathold": 0.5,
    "cdlmorningdojistar": 0.3,
    "cdlmorningstar": 0.3,
}


def recogniser(name: str) -> Callable[..., pl.Expr]:
    return cast(Callable[..., pl.Expr], getattr(ta, name))


def column(name: str, bars: pl.DataFrame | None = None) -> list:
    source = BARS if bars is None else bars
    expr = recogniser(name)("open", "high", "low", "close")
    return source.select(expr.alias("out")).to_series().to_list()


class TestEveryRecogniser:
    def test_the_catalogue_is_complete(self) -> None:
        assert len(PATTERN_NAMES) == 61
        assert set(PATTERN_NAMES) == set(WARM_UP)

    @mark.parametrize("name", PATTERN_NAMES)
    def test_it_is_re_exported_from_the_package_root(self, name: str) -> None:
        assert getattr(ta, name) is not None

    @mark.parametrize("name", PATTERN_NAMES)
    def test_it_scores_an_integer_from_the_allowed_set(self, name: str) -> None:
        expr = recogniser(name)("open", "high", "low", "close")
        series = BARS.select(expr.alias("out"))["out"]
        assert series.dtype == PATTERN_DTYPE
        assert set(series.drop_nulls().to_list()) <= ALLOWED

    @mark.parametrize("name", PATTERN_NAMES)
    def test_its_warm_up_is_the_ta_lib_lookback(self, name: str) -> None:
        result = column(name)
        warm_up = WARM_UP[name]
        assert result[:warm_up] == [None] * warm_up
        assert result[warm_up] is not None

    @mark.parametrize("name", PATTERN_NAMES)
    def test_a_null_bar_blanks_the_window_it_falls_in(self, name: str) -> None:
        gap = 40
        bars = BARS.with_columns(
            pl.when(pl.int_range(pl.len()) == gap)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        result = column(name, bars)
        warm_up = WARM_UP[name]
        assert result[gap - 1] is not None
        assert result[gap : gap + warm_up + 1] == [None] * (warm_up + 1)
        assert result[gap + warm_up + 1] is not None

    @mark.parametrize("name", PATTERN_NAMES)
    def test_series_input_is_rejected(self, name: str) -> None:
        with raises(TypeError):
            recogniser(name)(
                cast(Any, pl.Series("open", BARS["open"])), "high", "low", "close"
            )

    @mark.parametrize("name", tuple(PENETRATION))
    def test_invalid_penetration_raises(self, name: str) -> None:
        with raises(ValueError):
            recogniser(name)("open", "high", "low", "close", penetration=0.0)

    @mark.parametrize("name", tuple(PENETRATION))
    def test_the_default_penetration_matches_ta_lib(self, name: str) -> None:
        explicit = BARS.select(
            recogniser(name)(
                "open", "high", "low", "close", penetration=PENETRATION[name]
            ).alias("out")
        )
        assert explicit["out"].to_list() == column(name)
