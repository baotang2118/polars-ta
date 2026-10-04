# polars-ta

A Python technical-indicator library designed around Polars and PyArrow.

Indicators are expression-first: they return a `pl.Expr` that composes inside `select`/`with_columns` and runs lazily, and they also accept a `pl.Series` for eager use.

## Indicators

Indicators are grouped by how they are charted. *Overlay* indicators are drawn on the price axis; *momentum* oscillators occupy a separate pane; *volatility* indicators measure the size of movement rather than its direction. Every public indicator is also re-exported from the package root.

| Group | Module | Function | Description |
| ----- | ------ | -------- | ----------- |
| Overlay | `overlay.ma` | `sma(column, window)` | Simple moving average |
| Overlay | `overlay.ma` | `wma(column, window)` | Weighted moving average (linear weights) |
| Overlay | `overlay.ma` | `ema(column, window, *, alpha=None, mode="talib")` | Exponential moving average |
| Overlay | `overlay.ma` | `dema(column, window, *, alpha=None, mode="talib")` | Double exponential moving average |
| Overlay | `overlay.ma` | `tema(column, window, *, alpha=None, mode="talib")` | Triple exponential moving average |
| Overlay | `overlay.bands` | `bbands(column, window=20, *, num_std=2.0, ddof=0)` | Bollinger Bands (struct of `lower`/`middle`/`upper`) |
| Overlay | `overlay.channels` | `donchian(high, low, window=20)` | Donchian Channels (struct of `lower`/`middle`/`upper`) |
| Overlay | `overlay.supertrend` | `supertrend(high, low, close, window=10, multiplier=3.0)` | Supertrend (struct of `supertrend`/`direction`) |
| Overlay | `overlay.ichimoku` | `ichimoku(high, low, close, 9, 26, 52, 26)` | Ichimoku Cloud (struct of five lines) |
| Momentum | `momentum.rsi` | `rsi(column, window=14)` | Relative Strength Index |
| Momentum | `momentum.mfi` | `mfi(high, low, close, volume, window=14)` | Money Flow Index |
| Momentum | `momentum.stoch` | `stoch(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)` | Stochastic oscillator (struct of `k`/`d`) |
| Momentum | `momentum.cci` | `cci(high, low, close, window=14)` | Commodity Channel Index |
| Momentum | `momentum.macd` | `macd(column, fast_period=12, slow_period=26, signal_period=9, *, mode="talib")` | MACD (struct of `macd`/`signal`/`histogram`) |
| Momentum | `momentum.adx` | `adx(high, low, close, window=14)` | Average Directional Index (struct of `adx`/`plus_di`/`minus_di`) |
| Volatility | `volatility.atr` | `true_range(high, low, close)` | True Range |
| Volatility | `volatility.atr` | `atr(high, low, close, window=14)` | Average True Range |

```python
import polars as pl
from polars_ta import adx, atr, bbands, cci, ema, macd, mfi, rsi, sma, stoch, wma

df = pl.DataFrame({"close": [1.0, 3.0, 2.0, 6.0, 5.0, 9.0]})
df.with_columns(
    sma("close", 3).alias("sma_3"),
    wma("close", 3).alias("wma_3"),
    ema("close", 3).alias("ema_3"),
    rsi("close", 3).alias("rsi_3"),
)
```

`bbands` returns a single struct column, so one call stays one expression. `stoch` and `macd` work the same way:

```python
df.with_columns(bbands("close", 20).alias("bb")).unnest("bb")  # three columns
df.with_columns(bbands("close", 20).struct.field("upper"))  # just one band
df.with_columns(macd("close").alias("m")).unnest("m")  # macd, signal, histogram
```

Indicators needing several price columns take them positionally, in TA-Lib's order:

```python
ohlcv.with_columns(
    mfi("high", "low", "close", "volume", 14).alias("mfi_14"),
    cci("high", "low", "close", 14).alias("cci_14"),
    atr("high", "low", "close", 14).alias("atr_14"),
    stoch("high", "low", "close").alias("st"),
    adx("high", "low", "close").alias("a"),
)
```

Two caveats worth knowing before you reach for them:

- `supertrend`'s band ratchet is a sequential recursion with no Polars primitive, so it runs a Python scan inside `map_batches`. It still returns a `pl.Expr` and still works lazily, but it is much slower than the other indicators.
- `ichimoku`'s `lagging` line is the close shifted *backward*, so each row holds a future close. That is correct for plotting but is a lookahead bug if fed straight into a backtest signal.

Each function takes a column name, a `pl.Expr`, or a `pl.Series`:

```python
sma("close", 3)  # pl.Expr
sma(pl.col("close"), 3)  # pl.Expr
sma(pl.Series("close", [1.0, 2.0]), 3)  # pl.Series
```

`ema` defaults to the TA-Lib convention, seeding the recursion with the simple moving average of the first complete window. Pass `mode="recursive"` or `mode="adjust"` for the pandas `ewm(adjust=False)` and `ewm(adjust=True)` conventions, or `alpha=` to override the default smoothing factor of `2 / (window + 1)`. `dema`, `tema`, and `macd` chain further EMA passes and accept the same `mode`.

Every indicator emits exactly the TA-Lib lookback as leading nulls: `window - 1` for the single-pass moving averages, `bbands`, `cci`, and `donchian`; `2 * (window - 1)` and `3 * (window - 1)` for `dema` and `tema`; `window` for `rsi`, `mfi`, `atr`, and `supertrend`; `2 * window - 1` for `adx` itself; the sum of the three period offsets for `stoch`; and `(slow_period - 1) + (signal_period - 1)` for `macd`. Struct fields start on the same row where TA-Lib emits them together, and keep their own warm-ups where TA-Lib treats them as separate functions. See [docs/indicators.md](docs/indicators.md) for the formulas, null-handling rules, and worked examples.

## Development

Install the package and development tools, then run the unit tests with pytest:

```sh
uv sync --dev
uv run pytest
```

Tests use `unittest.TestCase` assertions and are executed with pytest. Follow [AGENTS.md](AGENTS.md) for the Python linting and formatting workflow.

## Project References

- [docs/indicators.md](docs/indicators.md): indicator formulas, conventions, and null handling.
- [KNOWLEDGE.md](KNOWLEDGE.md): package structure, design principles, and dependency guidance.
- [AGENTS.md](AGENTS.md): repository-wide development and validation requirements.
