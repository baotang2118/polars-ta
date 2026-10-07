"""Quick start from a bare pl.Series: convert it to a frame first.

Indicators are expression-only, so a pl.Series cannot be passed in directly.

Run with:

    uv run python examples/quickstart_series.py
"""

import polars as pl

from polars_ta import sma

close = pl.Series("close", [10.0, 11.0, 12.0, 11.5, 13.0, 14.0])

# 1. Passing the series straight in is rejected at call time, not at collect().
try:
    sma(close, 3)
except TypeError as exc:
    print("TypeError:", exc)

# 2. Give the series a frame to evaluate against. to_frame() reuses the series name.
result = close.to_frame().select(sma("close", 3)).to_series()
print(result)

# 3. An unnamed series has no column name to match, so supply one.
unnamed = pl.Series([10.0, 11.0, 12.0, 11.5, 13.0, 14.0])
print(unnamed.to_frame("close").select(sma("close", 3)).to_series().to_list())
