"""Compare all indicator categories on XAU daily data with expected CSV fixtures.

Run with:

    uv run pytest tests/uat/test_gold_prices.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ta import (
    ad,
    adosc,
    adx,
    adxr,
    annualized_volatility,
    ao,
    apo,
    aroon,
    aroonosc,
    atr,
    avgprice,
    bbands,
    beta,
    bop,
    cci,
    cmf,
    cmo,
    correl,
    cumulative_return,
    cvi,
    cycle,
    daily_log_return,
    daily_return,
    dema,
    donchian,
    dpo,
    dx,
    ema,
    eom,
    fi,
    fisher,
    fosc,
    hma,
    ht_dcperiod,
    ht_dcphase,
    ht_phasor,
    ht_sine,
    ht_trendline,
    ht_trendmode,
    ichimoku,
    kama,
    keltner,
    kst,
    kvo,
    linearreg,
    linearreg_angle,
    linearreg_intercept,
    linearreg_slope,
    ma,
    macd,
    macdext,
    macdfix,
    mama,
    marketfi,
    mass,
    mavp,
    medprice,
    mfi,
    midpoint,
    midprice,
    minus_di,
    minus_dm,
    mom,
    momentum,
    msw,
    natr,
    nvi,
    obv,
    overlay,
    pattern,
    plus_di,
    plus_dm,
    ppo,
    pvi,
    pvo,
    qstick,
    returns,
    roc,
    rocp,
    rocr,
    rocr100,
    rsi,
    sar,
    sarext,
    sma,
    statistic,
    stc,
    stddev,
    stoch,
    stochf,
    stochrsi,
    supertrend,
    t3,
    tema,
    trima,
    trix,
    true_range,
    tsf,
    tsi,
    typprice,
    ulcer,
    ultosc,
    var,
    vhf,
    vidya,
    volatility,
    volume,
    vortex,
    vpt,
    vwap,
    vwma,
    wad,
    wclprice,
    wilders,
    willr,
    wma,
    zlema,
    zscore,
)

DATA_DIR = Path(__file__).resolve().parent
DATA_PATH = DATA_DIR / "XAU-1d-data.csv"
DEFAULT_WINDOW = 14
NON_INDICATOR_EXPORTS = frozenset({"MA_TYPES", "MaType"})

CATEGORY_MODULES: Mapping[str, ModuleType] = {
    "cycle": cycle,
    "momentum": momentum,
    "overlay": overlay,
    "pattern": pattern,
    "returns": returns,
    "statistic": statistic,
    "volatility": volatility,
    "volume": volume,
}

CATEGORY_INDICATORS: Mapping[str, Mapping[str, pl.Expr]] = {
    "cycle": {
        "ht_dcperiod": ht_dcperiod("Close"),
        "ht_dcphase": ht_dcphase("Close"),
        "ht_phasor": ht_phasor("Close"),
        "ht_sine": ht_sine("Close"),
        "ht_trendline": ht_trendline("Close"),
        "ht_trendmode": ht_trendmode("Close"),
        "msw": msw("Close"),
    },
    "momentum": {
        "adx": adx("High", "Low", "Close"),
        "adxr": adxr("High", "Low", "Close"),
        "ao": ao("High", "Low"),
        "apo": apo("Close"),
        "aroon": aroon("High", "Low"),
        "aroonosc": aroonosc("High", "Low"),
        "bop": bop("Open", "High", "Low", "Close"),
        "cci": cci("High", "Low", "Close"),
        "cmo": cmo("Close"),
        "dpo": dpo("Close"),
        "dx": dx("High", "Low", "Close"),
        "fisher": fisher("High", "Low"),
        "fosc": fosc("Close"),
        "kst": kst("Close"),
        "macd": macd("Close"),
        "macdext": macdext("Close"),
        "macdfix": macdfix("Close"),
        "mass": mass("High", "Low"),
        "mfi": mfi("High", "Low", "Close", "Volume"),
        "minus_di": minus_di("High", "Low", "Close"),
        "minus_dm": minus_dm("High", "Low"),
        "mom": mom("Close"),
        "plus_di": plus_di("High", "Low", "Close"),
        "plus_dm": plus_dm("High", "Low"),
        "ppo": ppo("Close"),
        "pvo": pvo("Volume"),
        "qstick": qstick("Open", "Close"),
        "roc": roc("Close"),
        "rocp": rocp("Close"),
        "rocr": rocr("Close"),
        "rocr100": rocr100("Close"),
        "rsi": rsi("Close"),
        "stc": stc("Close"),
        "stoch": stoch("High", "Low", "Close"),
        "stochf": stochf("High", "Low", "Close"),
        "stochrsi": stochrsi("Close"),
        "trix": trix("Close"),
        "tsi": tsi("Close"),
        "ultosc": ultosc("High", "Low", "Close"),
        "vhf": vhf("Close"),
        "vortex": vortex("High", "Low", "Close"),
        "willr": willr("High", "Low", "Close"),
    },
    "overlay": {
        "avgprice": avgprice("Open", "High", "Low", "Close"),
        "bbands": bbands("Close"),
        "dema": dema("Close", DEFAULT_WINDOW),
        "donchian": donchian("High", "Low"),
        "ema": ema("Close", DEFAULT_WINDOW),
        "hma": hma("Close"),
        "ichimoku": ichimoku("High", "Low", "Close"),
        "kama": kama("Close"),
        "keltner": keltner("High", "Low", "Close"),
        "ma": ma("Close"),
        "mama": mama("Close"),
        "mavp": mavp("Close", ht_dcperiod("Close")),
        "medprice": medprice("High", "Low"),
        "midpoint": midpoint("Close"),
        "midprice": midprice("High", "Low"),
        "sar": sar("High", "Low"),
        "sarext": sarext("High", "Low"),
        "sma": sma("Close", DEFAULT_WINDOW),
        "supertrend": supertrend("High", "Low", "Close"),
        "t3": t3("Close"),
        "tema": tema("Close", DEFAULT_WINDOW),
        "trima": trima("Close"),
        "typprice": typprice("High", "Low", "Close"),
        "vidya": vidya("Close"),
        "vwma": vwma("Close", "Volume"),
        "wclprice": wclprice("High", "Low", "Close"),
        "wilders": wilders("Close"),
        "wma": wma("Close", DEFAULT_WINDOW),
        "zlema": zlema("Close"),
    },
    "pattern": {
        name: getattr(pattern, name)("Open", "High", "Low", "Close")
        for name in pattern.__all__
    },
    "returns": {
        "cumulative_return": cumulative_return("Close"),
        "daily_log_return": daily_log_return("Close"),
        "daily_return": daily_return("Close"),
    },
    "statistic": {
        "beta": beta("High", "Low"),
        "correl": correl("High", "Low"),
        "linearreg": linearreg("Close"),
        "linearreg_angle": linearreg_angle("Close"),
        "linearreg_intercept": linearreg_intercept("Close"),
        "linearreg_slope": linearreg_slope("Close"),
        "stddev": stddev("Close"),
        "tsf": tsf("Close"),
        "var": var("Close"),
        "zscore": zscore("Close"),
    },
    "volatility": {
        "annualized_volatility": annualized_volatility("Close"),
        "atr": atr("High", "Low", "Close"),
        "cvi": cvi("High", "Low"),
        "natr": natr("High", "Low", "Close"),
        "true_range": true_range("High", "Low", "Close"),
        "ulcer": ulcer("Close"),
    },
    "volume": {
        "ad": ad("High", "Low", "Close", "Volume"),
        "adosc": adosc("High", "Low", "Close", "Volume"),
        "cmf": cmf("High", "Low", "Close", "Volume"),
        "eom": eom("High", "Low", "Volume"),
        "fi": fi("Close", "Volume"),
        "kvo": kvo("High", "Low", "Close", "Volume"),
        "marketfi": marketfi("High", "Low", "Volume"),
        "nvi": nvi("Close", "Volume"),
        "obv": obv("Close", "Volume"),
        "pvi": pvi("Close", "Volume"),
        "vpt": vpt("Close", "Volume"),
        "vwap": vwap("High", "Low", "Close", "Volume"),
        "wad": wad("High", "Low", "Close"),
    },
}


def expected_csv_path(category: str) -> Path:
    return DATA_DIR / f"XAU-1d-data-expect-{category}.csv"


def load_ohlcv(path: Path) -> pl.DataFrame:
    raw_prices = pl.read_csv(path)
    return raw_prices.select(
        pl.col("Date").str.to_date("%Y.%m.%d").alias("Date"),
        pl.col("Open").cast(pl.Float64).alias("Open"),
        pl.col("High").cast(pl.Float64).alias("High"),
        pl.col("Low").cast(pl.Float64).alias("Low"),
        pl.col("Close").cast(pl.Float64).alias("Close"),
        pl.col("Volume").cast(pl.Float64).alias("Volume"),
    )


def evaluate_category(
    prices: pl.DataFrame, indicators: Mapping[str, pl.Expr]
) -> pl.DataFrame:
    expressions: Sequence[pl.Expr] = [
        expr.alias(name) for name, expr in indicators.items()
    ]
    result = prices.with_columns(*expressions)
    struct_columns: list[str] = []
    rename_exprs: list[pl.Expr] = []
    for name, dtype in result.schema.items():
        if isinstance(dtype, pl.Struct):
            struct_columns.append(name)
            rename_exprs.append(
                pl.col(name)
                .struct.rename_fields(
                    [f"{name}_{field.name}" for field in dtype.fields]
                )
                .alias(name)
            )
    if struct_columns:
        result = result.with_columns(*rename_exprs).unnest(*struct_columns)
    return result


def assert_or_create_expected(category: str, actual: pl.DataFrame) -> None:
    path = expected_csv_path(category)
    if not path.exists():
        actual.write_csv(path, date_format="%Y.%m.%d", float_precision=2)

    expected_raw = pl.read_csv(path, infer_schema_length=None)
    expected = expected_raw.select(
        pl.col("Date").str.to_date("%Y.%m.%d").alias("Date"),
        *[
            pl.col(name).cast(dtype).alias(name)
            for name, dtype in actual.schema.items()
            if name != "Date"
        ],
    )
    assert_frame_equal(actual, expected, rel_tol=0.0, abs_tol=0.0051)


def test_all_category_indicators_covered() -> None:
    for category, module in CATEGORY_MODULES.items():
        expected_names = set(module.__all__) - NON_INDICATOR_EXPORTS
        actual_names = set(CATEGORY_INDICATORS[category].keys())
        assert actual_names == expected_names, f"Mismatch in {category}"


@pytest.mark.parametrize("category", tuple(CATEGORY_INDICATORS.keys()))
def test_xau_indicators_match_expected(category: str) -> None:
    prices = load_ohlcv(DATA_PATH)
    actual = evaluate_category(prices, CATEGORY_INDICATORS[category])
    assert_or_create_expected(category, actual)
