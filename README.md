# polars-ta

A Python technical-indicator library designed around Polars and PyArrow.

Indicators are expression-first: they return a `pl.Expr` that composes inside `select`/`with_columns` and runs lazily, and they also accept a `pl.Series` for eager use.

## Indicators

Indicators are grouped by how they are charted. *Overlay* indicators are drawn on the price axis; *momentum* oscillators occupy a separate pane; *volume* indicators weight movement by how much traded; *volatility* indicators measure the size of movement; *cycle* indicators measure its rhythm. Every public indicator is also re-exported from the package root.

Every TA-Lib function in [indicators.md](indicators.md) is implemented.

| Group | Module | Function | Description |
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
| Overlay | `overlay.bands` | `bbands(column, window=20, *, num_std=2.0, ddof=0)` | Bollinger Bands (struct of `lower`/`middle`/`upper`) |
| Overlay | `overlay.channels` | `donchian(high, low, window=20)` | Donchian Channels (struct of `lower`/`middle`/`upper`) |
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
| Momentum | `momentum.trix` | `trix(column, window=30, *, mode="talib")` | TRIX |
| Momentum | `momentum.ultosc` | `ultosc(high, low, close, 7, 14, 28)` | Ultimate Oscillator |
| Volume | `volume.flow` | `ad(high, low, close, volume)` | Chaikin A/D Line |
| Volume | `volume.flow` | `adosc(high, low, close, volume, 3, 10)` | Chaikin A/D Oscillator |
| Volume | `volume.flow` | `obv(close, volume)` | On Balance Volume |
| Volatility | `volatility.atr` | `true_range(high, low, close)` | True Range |
| Volatility | `volatility.atr` | `atr(high, low, close, window=14)` | Average True Range |
| Volatility | `volatility.atr` | `natr(high, low, close, window=14)` | Normalized Average True Range |
| Cycle | `cycle.hilbert` | `ht_dcperiod(column)` | Dominant Cycle Period |
| Cycle | `cycle.hilbert` | `ht_dcphase(column)` | Dominant Cycle Phase |
| Cycle | `cycle.hilbert` | `ht_phasor(column)` | Phasor Components (struct of `in_phase`/`quadrature`) |
| Cycle | `cycle.hilbert` | `ht_sine(column)` | SineWave (struct of `sine`/`lead_sine`) |
| Cycle | `cycle.hilbert` | `ht_trendmode(column)` | Trend versus Cycle Mode (`Int8`) |
| Cycle | `cycle.hilbert` | `ht_trendline(column)` | Instantaneous Trendline |


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
- `ichimoku`'s `lagging` line is the close shifted *backward*, so each row holds a future close. That is correct for plotting but is a lookahead bug if fed straight into a backtest signal.

A few indicators take a runtime-selected average through `ma_type`, which accepts any name in `polars_ta.MA_TYPES` (`"sma"`, `"ema"`, `"wma"`, `"dema"`, `"tema"`, `"trima"`, `"kama"`, `"mama"`, `"t3"`):

```python
ma("close", 20, ma_type="trima")
apo("close", 12, 26, ma_type="ema")
macdext("close", 12, 26, 9, signal_ma_type="wma")
```

Each function takes a column name, a `pl.Expr`, or a `pl.Series`:

```python
sma("close", 3)  # pl.Expr
sma(pl.col("close"), 3)  # pl.Expr
sma(pl.Series("close", [1.0, 2.0]), 3)  # pl.Series
```

`ema` defaults to the TA-Lib convention, seeding the recursion with the simple moving average of the first complete window. Pass `mode="recursive"` or `mode="adjust"` for the pandas `ewm(adjust=False)` and `ewm(adjust=True)` conventions, or `alpha=` to override the default smoothing factor of `2 / (window + 1)`. `dema`, `tema`, and `macd` chain further EMA passes and accept the same `mode`.

Every indicator emits exactly the TA-Lib lookback as leading nulls: `window - 1` for the single-pass moving averages, `bbands`, `cci`, `donchian`, `midpoint`, `midprice`, `trima`, and `willr`; `2 * (window - 1)`, `3 * (window - 1)`, and `6 * (window - 1)` for `dema`, `tema`, and `t3`; `window` for `rsi`, `cmo`, `mfi`, `atr`, `natr`, `kama`, `supertrend`, `aroon`, `dx`, and the directional indicators; `window - 1` for `plus_dm`/`minus_dm`; `2 * window - 1` for `adx` and `3 * window - 2` for `adxr`; `1` for `sar` and `sarext`; `32` for `ht_dcperiod`, `ht_phasor`, and `mama`; `63` for the remaining `ht_*` indicators; and nothing at all for `bop`, `ad`, and `obv`. Struct fields start on the same row where TA-Lib emits them together, and keep their own warm-ups where TA-Lib treats them as separate functions. See [docs/indicators.md](docs/indicators.md) for the formulas, null-handling rules, and worked examples.

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
