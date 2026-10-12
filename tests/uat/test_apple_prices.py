"""Load the Apple daily-price fixture into a typed Polars DataFrame.

Run with:

    uv run pytest tests/uat/test_apple_prices.py
"""

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from polars_ta import sma

DATA_DIR = Path(__file__).resolve().parent
DATA_PATH = DATA_DIR / "apple-1d-data.csv"
EXPECTED_PATH = DATA_DIR / "apple-1d-data-expect.csv"


def load_closes(path: Path) -> pl.DataFrame:
    raw_prices = pl.read_csv(path)
    raw_prices = raw_prices.rename(
        {column: column.strip() for column in raw_prices.columns}
    )
    return raw_prices.select(
        pl.col("Date").str.strip_chars().str.to_date("%m/%d/%Y").alias("date"),
        pl.col("Close")
        .cast(pl.String)
        .str.strip_chars()
        .cast(pl.Float64)
        .alias("close"),
    )


def test_apple_sma_7_matches_expected() -> None:
    prices = load_closes(DATA_PATH)
    expected_raw = pl.read_csv(EXPECTED_PATH)
    expected_raw = expected_raw.rename(
        {column: column.strip() for column in expected_raw.columns}
    )
    expected_sma = expected_raw.select(
        pl.col("Date").str.strip_chars().str.to_date("%m/%d/%Y").alias("date"),
        pl.col("SMA_7").cast(pl.Float64).alias("sma_7"),
    )
    actual_sma = prices.with_columns(sma("close", 7).alias("sma_7")).select(
        "date", "sma_7"
    )

    assert_frame_equal(actual_sma, expected_sma, rel_tol=0.0, abs_tol=0.0051)
