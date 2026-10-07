"""Quick start: add a simple and an exponential moving average to a price frame.

Run with:

    uv run python examples/quickstart.py
"""

import polars as pl

from polars_ta import ema, sma

# 1. Build a frame. In practice use pl.read_csv("prices.csv") or pl.read_parquet(...).
prices = pl.DataFrame(
    {
        "date": pl.date_range(
            pl.date(2024, 1, 1), pl.date(2024, 1, 10), "1d", eager=True
        ),
        "close": [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5],
    }
)

# 2. Add indicators as new columns. Each call is an expression; .alias names the result.
out = prices.with_columns(
    sma("close", 3).alias("sma_3"),
    ema("close", 3).alias("ema_3"),
)

# 3. Print. `out` is an ordinary pl.DataFrame.
print(out)
