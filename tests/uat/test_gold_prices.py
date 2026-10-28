"""Compare the XAU seven-session SMA with its expected fixture.

Run with:

    uv run pytest tests/uat/test_gold_prices.py
"""

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from polars_ta import sma

DATA_DIR = Path(__file__).resolve().parent
DATA_PATH = DATA_DIR / "XAU-1d-data.csv"
EXPECTED_PATH = DATA_DIR / "XAU-1d-data-expect.csv"


def load_closes(path: Path) -> pl.DataFrame:
    raw_prices = pl.read_csv(path)
    return raw_prices.select(
        pl.col("Date").str.to_date("%Y.%m.%d").alias("date"),
        pl.col("Close").cast(pl.Float64).alias("close"),
    )


def test_xau_sma_7_matches_expected() -> None:
    prices = load_closes(DATA_PATH)
    expected_raw = pl.read_csv(EXPECTED_PATH)
    expected_sma = expected_raw.select(
        pl.col("Date").str.to_date("%Y.%m.%d").alias("date"),
        pl.col("SMA_7").cast(pl.Float64).alias("sma_7"),
    )
    actual_sma = prices.with_columns(sma("close", 7).alias("sma_7")).select(
        "date", "sma_7"
    )

    assert_frame_equal(actual_sma, expected_sma, rel_tol=0.0, abs_tol=0.0051)
