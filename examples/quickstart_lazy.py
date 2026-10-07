"""Quick start, lazily: the same moving averages through a LazyFrame.

Run with:

    uv run python examples/quickstart_lazy.py
"""

import polars as pl

from polars_ta import ema, sma

# 1. Build a lazy frame. In practice use pl.scan_csv(...) or pl.scan_parquet(...).
prices = pl.LazyFrame(
    {
        "date": pl.date_range(
            pl.date(2024, 1, 1), pl.date(2024, 1, 10), "1d", eager=True
        ),
        "close": [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5],
    }
)

# 2. Add indicators. Nothing is computed yet; this only extends the query plan.
query = prices.with_columns(
    sma("close", 3).alias("sma_3"),
    ema("close", 3).alias("ema_3"),
)
print("before collect:", type(query).__name__)

# 3. collect() optimizes and runs the whole plan, returning a pl.DataFrame.
out = query.collect()
print("after collect: ", type(out).__name__)
print(out.tail(3))
