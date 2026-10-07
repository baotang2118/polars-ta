# polars-ta

A Python technical-indicator library designed around Polars and PyArrow.

Indicators are expression-only: every function returns a `pl.Expr` that composes inside `select`/`with_columns` and runs lazily. Series and DataFrame inputs are not accepted — convert them outside the library.

## Quick start

### 1. Install

The library requires Python 3.10+ and depends on `polars` and `pyarrow`. From a clone of the repository:

```sh
uv sync        # or: pip install .
```

### 2. A complete example

Indicators read plain columns, so any DataFrame with the right price columns works. The script below is saved as [examples/quickstart.py](examples/quickstart.py) — run it with `uv run python examples/quickstart.py`:

```python
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
```

```text
shape: (10, 4)
┌────────────┬───────┬───────────┬───────────┐
│ date       ┆ close ┆ sma_3     ┆ ema_3     │
│ ---        ┆ ---   ┆ ---       ┆ ---       │
│ date       ┆ f64   ┆ f64       ┆ f64       │
╞════════════╪═══════╪═══════════╪═══════════╡
│ 2024-01-01 ┆ 10.0  ┆ null      ┆ null      │
│ 2024-01-02 ┆ 11.0  ┆ null      ┆ null      │
│ 2024-01-03 ┆ 12.0  ┆ 11.0      ┆ 11.0      │
│ 2024-01-04 ┆ 11.5  ┆ 11.5      ┆ 11.25     │
│ 2024-01-05 ┆ 13.0  ┆ 12.166667 ┆ 12.125    │
│ 2024-01-06 ┆ 14.0  ┆ 12.833333 ┆ 13.0625   │
│ 2024-01-07 ┆ 13.5  ┆ 13.5      ┆ 13.28125  │
│ 2024-01-08 ┆ 15.0  ┆ 14.166667 ┆ 14.140625 │
│ 2024-01-09 ┆ 16.0  ┆ 14.833333 ┆ 15.070312 │
│ 2024-01-10 ┆ 15.5  ┆ 15.5      ┆ 15.285156 │
└────────────┴───────┴───────────┴───────────┘
```

The first rows are `null` while the indicator warms up: a 3-period average needs 3 bars, so rows 1 and 2 are null. See [Missing values](#missing-values-null-and-nan).

### 3. What each call returns

Calling an indicator does **not** compute anything. It builds a `pl.Expr`, a recipe. The frame you pass it to decides when it runs and what you get back:

| Expression | Type | Computed? |
| ---------- | ---- | --------- |
| `sma("close", 3)` | `pl.Expr` | No — just a recipe |
| `df.with_columns(sma("close", 3))` | `pl.DataFrame` | Yes, immediately |
| `lf.with_columns(sma("close", 3))` | `pl.LazyFrame` | No — still a plan |
| `lf.with_columns(...).collect()` | `pl.DataFrame` | Yes, on `collect()` |

So an expression must be evaluated against a frame before it yields values: a `pl.Expr` on its own holds no data, and a `pl.LazyFrame` holds none until you call `.collect()`.

### 4. The same query, lazily

Swap `prices` for `prices.lazy()` and add `.collect()` at the end. Polars then optimizes the whole query before running it. Saved as [examples/quickstart_lazy.py](examples/quickstart_lazy.py) — run it with `uv run python examples/quickstart_lazy.py`:

```python
import polars as pl

from polars_ta import ema, sma

# Build a lazy frame. In practice use pl.scan_csv(...) or pl.scan_parquet(...).
prices = pl.LazyFrame(
    {
        "date": pl.date_range(
            pl.date(2024, 1, 1), pl.date(2024, 1, 10), "1d", eager=True
        ),
        "close": [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5],
    }
)

# Nothing is computed yet; this only extends the query plan.
query = prices.with_columns(
    sma("close", 3).alias("sma_3"),
    ema("close", 3).alias("ema_3"),
)
print("before collect:", type(query).__name__)

# collect() optimizes and runs the whole plan, returning a pl.DataFrame.
out = query.collect()
print("after collect: ", type(out).__name__)
print(out.tail(3))
```

```text
before collect: LazyFrame
after collect:  DataFrame
shape: (3, 4)
┌────────────┬───────┬───────────┬───────────┐
│ date       ┆ close ┆ sma_3     ┆ ema_3     │
│ ---        ┆ ---   ┆ ---       ┆ ---       │
│ date       ┆ f64   ┆ f64       ┆ f64       │
╞════════════╪═══════╪═══════════╪═══════════╡
│ 2024-01-08 ┆ 15.0  ┆ 14.166667 ┆ 14.140625 │
│ 2024-01-09 ┆ 16.0  ┆ 14.833333 ┆ 15.070312 │
│ 2024-01-10 ┆ 15.5  ┆ 15.5      ┆ 15.285156 │
└────────────┴───────┴───────────┴───────────┘
```

The values are identical to the eager version; only the execution strategy differs. Note that the indicator calls are unchanged — the same `sma("close", 3)` expression works in both.

### 5. More indicators, and struct outputs

Indicators needing several price columns take them positionally in TA-Lib's order, and a few return a struct holding more than one line. Use `.unnest()` to spread a struct into separate columns:

```python
import polars as pl
from polars_ta import atr, bbands, macd, rsi, sma

n = 40
close = [100 + i * 0.5 + (i % 5) * 1.3 for i in range(n)]
ohlcv = pl.DataFrame(
    {
        "date": pl.date_range(
            pl.date(2024, 1, 1), pl.date(2024, 2, 9), "1d", eager=True
        ),
        "open": [c - 0.4 for c in close],
        "high": [c + 1.0 for c in close],
        "low": [c - 1.0 for c in close],
        "close": close,
        "volume": [1000.0 + 10 * i for i in range(n)],
    }
)

out = ohlcv.with_columns(
    sma("close", 5).alias("sma_5"),
    rsi("close", 14).alias("rsi_14"),
    atr("high", "low", "close", 14).alias("atr_14"),  # price columns, in TA-Lib order
    bbands("close", 20).alias("bb"),  # struct column
    macd("close").alias("m"),  # struct column
).unnest("bb", "m")  # lower/middle/upper and macd/signal/histogram

print(out.select("date", "close", "sma_5", "rsi_14", "lower", "upper", "macd").tail(3))
```

Output columns are ordinary Polars columns, so filter, join, or group them as usual:

```python
oversold = out.filter((pl.col("rsi_14") < 30) | (pl.col("close") < pl.col("lower")))
```

Find the function you need in the [Indicators](#indicators) table below, and see [docs/indicators.md](docs/indicators.md) for formulas and worked examples.

## Inputs are expressions only

Every indicator accepts a column name or a `pl.Expr`, and nothing else:

```python
sma("close", 3)  # pl.Expr
sma(pl.col("close"), 3)  # pl.Expr
```

Passing a `pl.Series`, a `pl.DataFrame`, or a `pl.LazyFrame` raises `TypeError` at call time, not later at `collect()`:

```python
sma(pl.Series("close", [10.0, 11.0, 12.0]), 3)
# TypeError: pl.Series input is not supported; pass a column name or pl.Expr and
# evaluate the result on a frame, e.g. series.to_frame().select(...)
```

A single expression-only signature keeps the whole indicator inside one Polars query, so the optimizer sees it and nothing is materialized per call.

If you are holding a bare `pl.Series`, give it a frame to be evaluated against. `to_frame()` reuses the series name, so an unnamed series needs one supplied. Saved as [examples/quickstart_series.py](examples/quickstart_series.py):

```python
close = pl.Series("close", [10.0, 11.0, 12.0, 11.5, 13.0, 14.0])
close.to_frame().select(sma("close", 3)).to_series()
# shape: (6,) Series: 'close' [f64] - [null, null, 11.0, 11.5, 12.166667, 12.833333]

unnamed = pl.Series([10.0, 11.0, 12.0, 11.5, 13.0, 14.0])
unnamed.to_frame("close").select(sma("close", 3)).to_series()
```

Because indicators are expressions, they compose: feed one into another, or into any Polars expression, without evaluating in between.

```python
ohlcv.with_columns(
    sma(rsi("close", 14), 5).alias("smoothed_rsi"),  # an indicator of an indicator
    (sma("close", 5) - sma("close", 20)).alias("ma_spread"),
    sma(pl.col("high") - pl.col("low"), 10).alias("avg_range"),
)
```

## Indicators

Indicators are grouped by how they are charted. *Overlay* indicators are drawn on the price axis; *momentum* oscillators occupy a separate pane; *volume* indicators weight movement by how much traded; *volatility* indicators measure the size of movement; *cycle* indicators measure its rhythm; *returns* restate price on a percentage scale. Every public indicator is also re-exported from the package root.

Every indicator in [indicators.md](indicators.md) is implemented, along with a few TA-Lib does not carry: `donchian`, `keltner`, `supertrend`, and `ichimoku`.

| Category | Module | Function | Description |
| ----- | ------ | -------- | ----------- |
| Overlay | `overlay.ma` | `sma(column, window)` | Simple moving average |
| Overlay | `overlay.ma` | `wma(column, window)` | Weighted moving average (linear weights) |
| Overlay | `overlay.ma` | `ema(column, window, *, alpha=None, mode="talib")` | Exponential moving average |
| Overlay | `overlay.ma` | `dema(column, window, ...)` | Double exponential moving average |
| Overlay | `overlay.ma` | `tema(column, window, ...)` | Triple exponential moving average |
| Overlay | `overlay.ma` | `trima(column, window=30)` | Triangular moving average |
| Overlay | `overlay.ma` | `t3(column, window=5, *, vfactor=0.7, ...)` | Tillson T3 |
| Overlay | `overlay.adaptive` | `kama(column, window=30, *, fast_period=2, slow_period=30)` | Kaufman Adaptive Moving Average |
| Overlay | `overlay.adaptive` | `mama(column, *, fast_limit=0.5, slow_limit=0.05)` | MESA Adaptive Moving Average (struct of `mama`/`fama`) |
| Overlay | `overlay.dispatch` | `ma(column, window=30, *, ma_type="sma")` | Moving average of a runtime-chosen kind |
| Overlay | `overlay.dispatch` | `mavp(column, periods, min_period=2, max_period=30, *, ma_type="sma")` | Moving average with a per-row period |
| Overlay | `overlay.midpoint` | `midpoint(column, window=14)` | Midpoint of a rolling range |
| Overlay | `overlay.midpoint` | `midprice(high, low, window=14)` | Midpoint price |
| Overlay | `overlay.transform` | `avgprice(open_, high, low, close)` | Average price |
| Overlay | `overlay.transform` | `medprice(high, low)` | Median price |
| Overlay | `overlay.transform` | `typprice(high, low, close)` | Typical price |
| Overlay | `overlay.transform` | `wclprice(high, low, close)` | Weighted close price |
| Overlay | `overlay.bands` | `bbands(column, window=20, *, num_std=2.0, ddof=0)` | Bollinger Bands (struct of `lower`/`middle`/`upper`) |
| Overlay | `overlay.channels` | `donchian(high, low, window=20)` | Donchian Channels (struct of `lower`/`middle`/`upper`) |
| Overlay | `overlay.channels` | `keltner(high, low, close, window=20, *, atr_window=10, multiplier=2.0)` | Keltner Channels (struct of `lower`/`middle`/`upper`) |
| Overlay | `overlay.sar` | `sar(high, low, acceleration=0.02, maximum=0.2)` | Parabolic SAR |
| Overlay | `overlay.sar` | `sarext(high, low, *, start_value=0.0, ...)` | Parabolic SAR, extended and signed |
| Overlay | `overlay.supertrend` | `supertrend(high, low, close, window=10, multiplier=3.0)` | Supertrend (struct of `supertrend`/`direction`) |
| Overlay | `overlay.ichimoku` | `ichimoku(high, low, close, 9, 26, 52, 26)` | Ichimoku Cloud (struct of five lines) |
| Momentum | `momentum.rsi` | `rsi(column, window=14)` | Relative Strength Index |
| Momentum | `momentum.rsi` | `cmo(column, window=14)` | Chande Momentum Oscillator |
| Momentum | `momentum.mfi` | `mfi(high, low, close, volume, window=14)` | Money Flow Index |
| Momentum | `momentum.stoch` | `stoch(high, low, close, 5, 3, 3)` | Stochastic oscillator (struct of `k`/`d`) |
| Momentum | `momentum.stoch` | `stochf(high, low, close, 5, 3)` | Fast stochastic (struct of `fast_k`/`fast_d`) |
| Momentum | `momentum.stoch` | `stochrsi(column, 14, 5, 3)` | Stochastic RSI (struct of `fast_k`/`fast_d`) |
| Momentum | `momentum.stoch` | `willr(high, low, close, window=14)` | Williams %R |
| Momentum | `momentum.cci` | `cci(high, low, close, window=14)` | Commodity Channel Index |
| Momentum | `momentum.macd` | `macd(column, 12, 26, 9, *, mode="talib")` | MACD (struct of `macd`/`signal`/`histogram`) |
| Momentum | `momentum.macd` | `macdext(column, 12, 26, 9, *, fast_ma_type=..., ...)` | MACD with selectable averages |
| Momentum | `momentum.macd` | `macdfix(column, signal_period=9, *, mode="talib")` | MACD fixed at 12/26 |
| Momentum | `momentum.adx` | `adx(high, low, close, window=14)` | Average Directional Index (struct of `adx`/`plus_di`/`minus_di`) |
| Momentum | `momentum.adx` | `adxr(high, low, close, window=14)` | Average Directional Index Rating |
| Momentum | `momentum.adx` | `dx(high, low, close, window=14)` | Directional Movement Index |
| Momentum | `momentum.adx` | `plus_di`, `minus_di` `(high, low, close, window=14)` | Directional Indicators |
| Momentum | `momentum.adx` | `plus_dm`, `minus_dm` `(high, low, window=14)` | Directional Movement sums |
| Momentum | `momentum.aroon` | `aroon(high, low, window=14)` | Aroon (struct of `down`/`up`) |
| Momentum | `momentum.aroon` | `aroonosc(high, low, window=14)` | Aroon Oscillator |
| Momentum | `momentum.bop` | `bop(open_, high, low, close)` | Balance of Power |
| Momentum | `momentum.roc` | `mom(column, window=10)` | Momentum |
| Momentum | `momentum.roc` | `roc`, `rocp`, `rocr`, `rocr100` `(column, window=10)` | Rate-of-change family |
| Momentum | `momentum.price_oscillator` | `apo(column, 12, 26, *, ma_type="sma")` | Absolute Price Oscillator |
| Momentum | `momentum.price_oscillator` | `ppo(column, 12, 26, *, ma_type="sma")` | Percentage Price Oscillator |
| Momentum | `momentum.price_oscillator` | `pvo(volume, 12, 26, *, ma_type="ema")` | Percentage Volume Oscillator |
| Momentum | `momentum.trix` | `trix(column, window=30, *, mode="talib")` | TRIX |
| Momentum | `momentum.ultosc` | `ultosc(high, low, close, 7, 14, 28)` | Ultimate Oscillator |
| Momentum | `momentum.dpo` | `dpo(column, window=20)` | Detrended Price Oscillator |
| Momentum | `momentum.kst` | `kst(column, roc_periods, sma_periods, 9)` | KST Oscillator (struct of `kst`/`signal`) |
| Momentum | `momentum.stc` | `stc(column, 23, 50, 10, *, smooth_k=3, smooth_d=3)` | Schaff Trend Cycle |
| Momentum | `momentum.tsi` | `tsi(column, 13, 25)` | True Strength Index |
| Momentum | `momentum.awesome` | `ao(high, low, 5, 34)` | Awesome Oscillator |
| Momentum | `momentum.mass` | `mass(high, low, 9, 25)` | Mass Index |
| Momentum | `momentum.vortex` | `vortex(high, low, close, window=14)` | Vortex Indicator (struct of `plus`/`minus`) |
| Volume | `volume.flow` | `ad(high, low, close, volume)` | Chaikin A/D Line |
| Volume | `volume.flow` | `adosc(high, low, close, volume, 3, 10)` | Chaikin A/D Oscillator |
| Volume | `volume.flow` | `obv(close, volume)` | On Balance Volume |
| Volume | `volume.flow` | `cmf(high, low, close, volume, window=20)` | Chaikin Money Flow |
| Volume | `volume.pressure` | `fi(close, volume, window=13)` | Force Index |
| Volume | `volume.pressure` | `eom(high, low, volume, window=14)` | Ease of Movement |
| Volume | `volume.pressure` | `vpt(close, volume)` | Volume-Price Trend |
| Volume | `volume.pressure` | `nvi(close, volume, *, start_value=1000.0)` | Negative Volume Index |
| Volume | `volume.vwap` | `vwap(high, low, close, volume, window=14)` | Volume Weighted Average Price |
| Volatility | `volatility.atr` | `true_range(high, low, close)` | True Range |
| Volatility | `volatility.atr` | `atr(high, low, close, window=14)` | Average True Range |
| Volatility | `volatility.atr` | `natr(high, low, close, window=14)` | Normalized Average True Range |
| Volatility | `volatility.ulcer` | `ulcer(column, window=14)` | Ulcer Index |
| Cycle | `cycle.hilbert` | `ht_dcperiod(column)` | Dominant Cycle Period |
| Cycle | `cycle.hilbert` | `ht_dcphase(column)` | Dominant Cycle Phase |
| Cycle | `cycle.hilbert` | `ht_phasor(column)` | Phasor Components (struct of `in_phase`/`quadrature`) |
| Cycle | `cycle.hilbert` | `ht_sine(column)` | SineWave (struct of `sine`/`lead_sine`) |
| Cycle | `cycle.hilbert` | `ht_trendmode(column)` | Trend versus Cycle Mode (`Int8`) |
| Cycle | `cycle.hilbert` | `ht_trendline(column)` | Instantaneous Trendline |
| Returns | `returns.performance` | `daily_return(column)` | Daily return, in percent |
| Returns | `returns.performance` | `daily_log_return(column)` | Daily log return, in percent |
| Returns | `returns.performance` | `cumulative_return(column)` | Return from the first known value |


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

- `supertrend`, `kama`, `sar`, `sarext`, `mama`, and the `ht_*` cycle indicators are sequential recursions with no Polars primitive, so they run a Python scan inside `map_batches`. They still return a `pl.Expr` and still work lazily, but they are much slower than the other indicators.
- `ichimoku`'s `lagging` line is the close shifted *backward*, so each row holds a future close. That is correct for plotting but is a lookahead bug if fed straight into a backtest signal. `dpo` has the mirror-image property: it compares against a *centred* average, so it describes past cycles rather than the current bar.

`vwap` is the rolling form rather than the session-anchored one, so set `window` to the number of bars in your session if that is what you need:

```python
ohlcv.with_columns(
    vwap("high", "low", "close", "volume", 14).alias("vwap_14"),
    cmf("high", "low", "close", "volume", 20).alias("cmf_20"),
    vortex("high", "low", "close", 14).alias("vi"),
)
```

A few indicators take a runtime-selected average through `ma_type`, which accepts any name in `polars_ta.MA_TYPES` (`"sma"`, `"ema"`, `"wma"`, `"dema"`, `"tema"`, `"trima"`, `"kama"`, `"mama"`, `"t3"`):

```python
ma("close", 20, ma_type="trima")
apo("close", 12, 26, ma_type="ema")
macdext("close", 12, 26, 9, signal_ma_type="wma")
```

`ema` defaults to the TA-Lib convention, seeding the recursion with the simple moving average of the first complete window. Pass `mode="recursive"` or `mode="adjust"` for the pandas `ewm(adjust=False)` and `ewm(adjust=True)` conventions, or `alpha=` to override the default smoothing factor of `2 / (window + 1)`. `dema`, `tema`, and `macd` chain further EMA passes and accept the same `mode`.

Every indicator emits exactly the TA-Lib lookback as leading nulls: `window - 1` for the single-pass moving averages, `bbands`, `cci`, `donchian`, `midpoint`, `midprice`, `trima`, `willr`, `cmf`, and `vwap`; `2 * (window - 1)`, `3 * (window - 1)`, and `6 * (window - 1)` for `dema`, `tema`, and `t3`; `window` for `rsi`, `cmo`, `mfi`, `atr`, `natr`, `kama`, `supertrend`, `aroon`, `dx`, `fi`, `eom`, `vortex`, and the directional indicators; `window - 1` for `plus_dm`/`minus_dm`; `2 * window - 1` for `adx` and `3 * window - 2` for `adxr`; `1` for `sar`, `sarext`, `daily_return`, and `daily_log_return`; `2 * (window - 1)` for `ulcer`; `32` for `ht_dcperiod`, `ht_phasor`, and `mama`; `63` for the remaining `ht_*` indicators; and nothing at all for `bop`, `ad`, `obv`, `vpt`, `nvi`, `cumulative_return`, and the price transforms. Struct fields start on the same row where TA-Lib emits them together, and keep their own warm-ups where TA-Lib treats them as separate functions. `donchian`, `keltner`, `ichimoku`, and `kst` likewise let each field reflect only the inputs it depends on — `keltner`'s `middle` starts after `window - 1` rows while its edges wait for the ATR. See [docs/indicators.md](docs/indicators.md) for the formulas, null-handling rules, and worked examples.

## Missing values: null and NaN

Null is the convention. Warm-up rows, rows whose window contains a null input, and unavailable results are all reported as null.

NaN is a different value in Polars: `is_null()` is false for NaN, and `fill_null()` does not replace it. NaN can still appear in two ways:

- **Input NaN.** The indicators do not convert NaN to null, so a NaN input propagates into the output rows that depend on it. Convert it first if you want it treated as missing:

  ```python
  df = df.with_columns(pl.col(pl.Float64).fill_nan(None))
  ```

- **Undefined results.** `daily_log_return` yields NaN where the price changes sign, because the logarithm is undefined there.

## Development

Install the package and development tools, then run the unit tests with pytest:

```sh
uv sync --dev
uv run pytest
```

Tests use `unittest.TestCase` assertions and are executed with pytest. Module-level fixtures such as `LENGTH` and `BARS` are explicitly annotated. Follow [AGENTS.md](AGENTS.md) for the Python linting and formatting workflow.

## Project References

- [docs/indicators.md](docs/indicators.md): shared conventions, the indicator index, and the overlay formulas.
- [docs/indicators-oscillators.md](docs/indicators-oscillators.md): momentum, volume, volatility, cycle, and returns formulas.
- [KNOWLEDGE.md](KNOWLEDGE.md): package structure, design principles, and dependency guidance.
- [AGENTS.md](AGENTS.md): repository-wide development and validation requirements.
