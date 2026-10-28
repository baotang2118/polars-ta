"""Compare all indicator categories on XAU daily data with expected CSV fixtures.

Run with:

    uv run pytest tests/uat/test_gold_prices.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

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


class IndicatorSpec(NamedTuple):
    alias: str
    expr: pl.Expr


CATEGORY_INDICATORS: Mapping[str, Mapping[str, IndicatorSpec]] = {
    "cycle": {
        "ht_dcperiod": IndicatorSpec("ht_dcperiod_close", ht_dcperiod("Close")),
        "ht_dcphase": IndicatorSpec("ht_dcphase_close", ht_dcphase("Close")),
        "ht_phasor": IndicatorSpec("ht_phasor_close", ht_phasor("Close")),
        "ht_sine": IndicatorSpec("ht_sine_close", ht_sine("Close")),
        "ht_trendline": IndicatorSpec("ht_trendline_close", ht_trendline("Close")),
        "ht_trendmode": IndicatorSpec("ht_trendmode_close", ht_trendmode("Close")),
        "msw": IndicatorSpec("msw_close_5", msw("Close")),
    },
    "momentum": {
        "adx": IndicatorSpec("adx_high_low_close_14", adx("High", "Low", "Close")),
        "adxr": IndicatorSpec("adxr_high_low_close_14", adxr("High", "Low", "Close")),
        "ao": IndicatorSpec("ao_high_low_5_34", ao("High", "Low")),
        "apo": IndicatorSpec("apo_close_12_26", apo("Close")),
        "aroon": IndicatorSpec("aroon_high_low_14", aroon("High", "Low")),
        "aroonosc": IndicatorSpec("aroonosc_high_low_14", aroonosc("High", "Low")),
        "bop": IndicatorSpec(
            "bop_open_high_low_close", bop("Open", "High", "Low", "Close")
        ),
        "cci": IndicatorSpec("cci_high_low_close_14", cci("High", "Low", "Close")),
        "cmo": IndicatorSpec("cmo_close_14", cmo("Close")),
        "dpo": IndicatorSpec("dpo_close_20", dpo("Close")),
        "dx": IndicatorSpec("dx_high_low_close_14", dx("High", "Low", "Close")),
        "fisher": IndicatorSpec("fisher_high_low_9", fisher("High", "Low")),
        "fosc": IndicatorSpec("fosc_close_14", fosc("Close")),
        "kst": IndicatorSpec("kst_close_10_15_20_30_10_10_10_15_9", kst("Close")),
        "macd": IndicatorSpec("macd_close_12_26_9", macd("Close")),
        "macdext": IndicatorSpec("macdext_close_12_26_9", macdext("Close")),
        "macdfix": IndicatorSpec("macdfix_close_9", macdfix("Close")),
        "mass": IndicatorSpec("mass_high_low_9_25", mass("High", "Low")),
        "mfi": IndicatorSpec(
            "mfi_high_low_close_volume_14", mfi("High", "Low", "Close", "Volume")
        ),
        "minus_di": IndicatorSpec(
            "minus_di_high_low_close_14", minus_di("High", "Low", "Close")
        ),
        "minus_dm": IndicatorSpec("minus_dm_high_low_14", minus_dm("High", "Low")),
        "mom": IndicatorSpec("mom_close_10", mom("Close")),
        "plus_di": IndicatorSpec(
            "plus_di_high_low_close_14", plus_di("High", "Low", "Close")
        ),
        "plus_dm": IndicatorSpec("plus_dm_high_low_14", plus_dm("High", "Low")),
        "ppo": IndicatorSpec("ppo_close_12_26", ppo("Close")),
        "pvo": IndicatorSpec("pvo_volume_12_26", pvo("Volume")),
        "qstick": IndicatorSpec("qstick_open_close_10", qstick("Open", "Close")),
        "roc": IndicatorSpec("roc_close_10", roc("Close")),
        "rocp": IndicatorSpec("rocp_close_10", rocp("Close")),
        "rocr": IndicatorSpec("rocr_close_10", rocr("Close")),
        "rocr100": IndicatorSpec("rocr100_close_10", rocr100("Close")),
        "rsi": IndicatorSpec("rsi_close_14", rsi("Close")),
        "stc": IndicatorSpec("stc_close_23_50_10", stc("Close")),
        "stoch": IndicatorSpec(
            "stoch_high_low_close_5_3_3", stoch("High", "Low", "Close")
        ),
        "stochf": IndicatorSpec(
            "stochf_high_low_close_5_3", stochf("High", "Low", "Close")
        ),
        "stochrsi": IndicatorSpec("stochrsi_close_14_5_3", stochrsi("Close")),
        "trix": IndicatorSpec("trix_close_30", trix("Close")),
        "tsi": IndicatorSpec("tsi_close_13_25", tsi("Close")),
        "ultosc": IndicatorSpec(
            "ultosc_high_low_close_7_14_28", ultosc("High", "Low", "Close")
        ),
        "vhf": IndicatorSpec("vhf_close_28", vhf("Close")),
        "vortex": IndicatorSpec(
            "vortex_high_low_close_14", vortex("High", "Low", "Close")
        ),
        "willr": IndicatorSpec(
            "willr_high_low_close_14", willr("High", "Low", "Close")
        ),
    },
    "overlay": {
        "avgprice": IndicatorSpec(
            "avgprice_open_high_low_close", avgprice("Open", "High", "Low", "Close")
        ),
        "bbands": IndicatorSpec("bbands_close_20_2", bbands("Close")),
        "dema": IndicatorSpec("dema_close_14", dema("Close", DEFAULT_WINDOW)),
        "donchian": IndicatorSpec("donchian_high_low_20", donchian("High", "Low")),
        "ema": IndicatorSpec("ema_close_14", ema("Close", DEFAULT_WINDOW)),
        "hma": IndicatorSpec("hma_close_9", hma("Close")),
        "ichimoku": IndicatorSpec(
            "ichimoku_high_low_close_9_26_52_26", ichimoku("High", "Low", "Close")
        ),
        "kama": IndicatorSpec("kama_close_30_2_30", kama("Close")),
        "keltner": IndicatorSpec(
            "keltner_high_low_close_20_10_2", keltner("High", "Low", "Close")
        ),
        "ma": IndicatorSpec("ma_close_30", ma("Close")),
        "mama": IndicatorSpec("mama_close_0.5_0.05", mama("Close")),
        "mavp": IndicatorSpec("mavp_close", mavp("Close", ht_dcperiod("Close"))),
        "medprice": IndicatorSpec("medprice_high_low", medprice("High", "Low")),
        "midpoint": IndicatorSpec("midpoint_close_14", midpoint("Close")),
        "midprice": IndicatorSpec("midprice_high_low_14", midprice("High", "Low")),
        "sar": IndicatorSpec("sar_high_low_0.02_0.2", sar("High", "Low")),
        "sarext": IndicatorSpec("sarext_high_low", sarext("High", "Low")),
        "sma": IndicatorSpec("sma_close_14", sma("Close", DEFAULT_WINDOW)),
        "supertrend": IndicatorSpec(
            "supertrend_high_low_close_10_3", supertrend("High", "Low", "Close")
        ),
        "t3": IndicatorSpec("t3_close_5_0.7", t3("Close")),
        "tema": IndicatorSpec("tema_close_14", tema("Close", DEFAULT_WINDOW)),
        "trima": IndicatorSpec("trima_close_30", trima("Close")),
        "typprice": IndicatorSpec(
            "typprice_high_low_close", typprice("High", "Low", "Close")
        ),
        "vidya": IndicatorSpec("vidya_close_12_26_0.2", vidya("Close")),
        "vwma": IndicatorSpec("vwma_close_volume_20", vwma("Close", "Volume")),
        "wclprice": IndicatorSpec(
            "wclprice_high_low_close", wclprice("High", "Low", "Close")
        ),
        "wilders": IndicatorSpec("wilders_close_14", wilders("Close")),
        "wma": IndicatorSpec("wma_close_14", wma("Close", DEFAULT_WINDOW)),
        "zlema": IndicatorSpec("zlema_close_10", zlema("Close")),
    },
    "pattern": {
        name: IndicatorSpec(
            name, getattr(pattern, name)("Open", "High", "Low", "Close")
        )
        for name in pattern.__all__
    },
    "returns": {
        "cumulative_return": IndicatorSpec(
            "cumulative_return_close", cumulative_return("Close")
        ),
        "daily_log_return": IndicatorSpec(
            "daily_log_return_close", daily_log_return("Close")
        ),
        "daily_return": IndicatorSpec("daily_return_close", daily_return("Close")),
    },
    "statistic": {
        "beta": IndicatorSpec("beta_high_low_5", beta("High", "Low")),
        "correl": IndicatorSpec("correl_high_low_30", correl("High", "Low")),
        "linearreg": IndicatorSpec("linearreg_close_14", linearreg("Close")),
        "linearreg_angle": IndicatorSpec(
            "linearreg_angle_close_14", linearreg_angle("Close")
        ),
        "linearreg_intercept": IndicatorSpec(
            "linearreg_intercept_close_14", linearreg_intercept("Close")
        ),
        "linearreg_slope": IndicatorSpec(
            "linearreg_slope_close_14", linearreg_slope("Close")
        ),
        "stddev": IndicatorSpec("stddev_close_5", stddev("Close")),
        "tsf": IndicatorSpec("tsf_close_14", tsf("Close")),
        "var": IndicatorSpec("var_close_5", var("Close")),
        "zscore": IndicatorSpec("zscore_close_20", zscore("Close")),
    },
    "volatility": {
        "annualized_volatility": IndicatorSpec(
            "annualized_volatility_close_20", annualized_volatility("Close")
        ),
        "atr": IndicatorSpec("atr_high_low_close_14", atr("High", "Low", "Close")),
        "cvi": IndicatorSpec("cvi_high_low_10", cvi("High", "Low")),
        "natr": IndicatorSpec("natr_high_low_close_14", natr("High", "Low", "Close")),
        "true_range": IndicatorSpec(
            "true_range_high_low_close", true_range("High", "Low", "Close")
        ),
        "ulcer": IndicatorSpec("ulcer_close_14", ulcer("Close")),
    },
    "volume": {
        "ad": IndicatorSpec(
            "ad_high_low_close_volume", ad("High", "Low", "Close", "Volume")
        ),
        "adosc": IndicatorSpec(
            "adosc_high_low_close_volume_3_10",
            adosc("High", "Low", "Close", "Volume"),
        ),
        "cmf": IndicatorSpec(
            "cmf_high_low_close_volume_20", cmf("High", "Low", "Close", "Volume")
        ),
        "eom": IndicatorSpec("eom_high_low_volume_14", eom("High", "Low", "Volume")),
        "fi": IndicatorSpec("fi_close_volume_13", fi("Close", "Volume")),
        "kvo": IndicatorSpec(
            "kvo_high_low_close_volume_34_55",
            kvo("High", "Low", "Close", "Volume"),
        ),
        "marketfi": IndicatorSpec(
            "marketfi_high_low_volume", marketfi("High", "Low", "Volume")
        ),
        "nvi": IndicatorSpec("nvi_close_volume", nvi("Close", "Volume")),
        "obv": IndicatorSpec("obv_close_volume", obv("Close", "Volume")),
        "pvi": IndicatorSpec("pvi_close_volume", pvi("Close", "Volume")),
        "vpt": IndicatorSpec("vpt_close_volume", vpt("Close", "Volume")),
        "vwap": IndicatorSpec(
            "vwap_high_low_close_volume_14",
            vwap("High", "Low", "Close", "Volume"),
        ),
        "wad": IndicatorSpec("wad_high_low_close", wad("High", "Low", "Close")),
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
    prices: pl.DataFrame, indicators: Mapping[str, IndicatorSpec]
) -> pl.DataFrame:
    expressions: Sequence[pl.Expr] = [
        spec.expr.alias(spec.alias) for spec in indicators.values()
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
