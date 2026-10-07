# Indicator Reference: Conventions and Overlays

Formulas and conventions for every indicator implemented in `polars_ta`.
Indicators are grouped by how they are charted: *overlay* indicators share the
price axis, *momentum* oscillators occupy a separate pane, *volume* indicators
weight movement by how much traded, *volatility* indicators measure the size of
movement, *cycle* indicators measure its rhythm, and *returns* restate price on
a percentage scale. Each group is a subpackage (`polars_ta.overlay`,
`polars_ta.momentum`, `polars_ta.volume`, `polars_ta.volatility`,
`polars_ta.cycle`, `polars_ta.returns`), and every public indicator is also
re-exported from the package root.

The reference is split across two pages:

| Page | Covers |
| ---- | ------ |
| **This page** | The shared conventions, the index below, and every **overlay** indicator |
| [indicators-oscillators.md](indicators-oscillators.md) | **Momentum**, **volume**, **volatility**, **cycle**, and **returns** indicators |

The conventions below apply to both pages.

Every indicator listed in `indicators.md` at the repository root is implemented.

| Indicator | Module | Inputs | Output |
| --------- | ------ | ------ | ------ |
| `sma`, `wma`, `ema`, `dema`, `tema`, `trima`, `t3` | `overlay.ma` | one column | one `Float64` column |
| `kama` | `overlay.adaptive` | one column | one `Float64` column |
| `mama` | `overlay.adaptive` | one column | struct of two `Float64` fields |
| `ma`, `mavp` | `overlay.dispatch` | one or two columns | one `Float64` column |
| `midpoint` | `overlay.midpoint` | one column | one `Float64` column |
| `midprice` | `overlay.midpoint` | high, low | one `Float64` column |
| `avgprice` | `overlay.transform` | open, high, low, close | one `Float64` column |
| `medprice` | `overlay.transform` | high, low | one `Float64` column |
| `typprice`, `wclprice` | `overlay.transform` | high, low, close | one `Float64` column |
| `bbands` | `overlay.bands` | one column | struct of three `Float64` fields |
| `donchian` | `overlay.channels` | high, low | struct of three `Float64` fields |
| `keltner` | `overlay.channels` | high, low, close | struct of three `Float64` fields |
| `sar`, `sarext` | `overlay.sar` | high, low | one `Float64` column |
| `supertrend` | `overlay.supertrend` | high, low, close | struct of `Float64` and `Int8` |
| `ichimoku` | `overlay.ichimoku` | high, low, close | struct of five `Float64` fields |
| `rsi`, `cmo` | `momentum.rsi` | one column | one `Float64` column |
| `mfi` | `momentum.mfi` | high, low, close, volume | one `Float64` column |
| `stoch`, `stochf` | `momentum.stoch` | high, low, close | struct of two `Float64` fields |
| `stochrsi` | `momentum.stoch` | one column | struct of two `Float64` fields |
| `willr` | `momentum.stoch` | high, low, close | one `Float64` column |
| `cci` | `momentum.cci` | high, low, close | one `Float64` column |
| `macd`, `macdext`, `macdfix` | `momentum.macd` | one column | struct of three `Float64` fields |
| `adx` | `momentum.adx` | high, low, close | struct of three `Float64` fields |
| `adxr`, `dx`, `plus_di`, `minus_di` | `momentum.adx` | high, low, close | one `Float64` column |
| `plus_dm`, `minus_dm` | `momentum.adx` | high, low | one `Float64` column |
| `aroon` | `momentum.aroon` | high, low | struct of two `Float64` fields |
| `aroonosc` | `momentum.aroon` | high, low | one `Float64` column |
| `bop` | `momentum.bop` | open, high, low, close | one `Float64` column |
| `mom`, `roc`, `rocp`, `rocr`, `rocr100` | `momentum.roc` | one column | one `Float64` column |
| `apo`, `ppo`, `pvo` | `momentum.price_oscillator` | one column | one `Float64` column |
| `trix` | `momentum.trix` | one column | one `Float64` column |
| `ultosc` | `momentum.ultosc` | high, low, close | one `Float64` column |
| `dpo` | `momentum.dpo` | one column | one `Float64` column |
| `kst` | `momentum.kst` | one column | struct of two `Float64` fields |
| `stc` | `momentum.stc` | one column | one `Float64` column |
| `tsi` | `momentum.tsi` | one column | one `Float64` column |
| `ao` | `momentum.awesome` | high, low | one `Float64` column |
| `mass` | `momentum.mass` | high, low | one `Float64` column |
| `vortex` | `momentum.vortex` | high, low, close | struct of two `Float64` fields |
| `ad`, `adosc`, `cmf` | `volume.flow` | high, low, close, volume | one `Float64` column |
| `obv` | `volume.flow` | close, volume | one `Float64` column |
| `fi`, `vpt`, `nvi` | `volume.pressure` | close, volume | one `Float64` column |
| `eom` | `volume.pressure` | high, low, volume | one `Float64` column |
| `vwap` | `volume.vwap` | high, low, close, volume | one `Float64` column |
| `true_range`, `atr`, `natr` | `volatility.atr` | high, low, close | one `Float64` column |
| `ulcer` | `volatility.ulcer` | one column | one `Float64` column |
| `ht_dcperiod`, `ht_dcphase`, `ht_trendline` | `cycle.hilbert` | one column | one `Float64` column |
| `ht_phasor`, `ht_sine` | `cycle.hilbert` | one column | struct of two `Float64` fields |
| `ht_trendmode` | `cycle.hilbert` | one column | one `Int8` column |
| `daily_return`, `daily_log_return`, `cumulative_return` | `returns.performance` | one column | one `Float64` column |

Everything in the `overlay.*` modules is documented on this page; the rest is in
[indicators-oscillators.md](indicators-oscillators.md).

## Shared Conventions

- **Input forms.** Each indicator accepts a column name (`str`) or a `pl.Expr`,
  and always returns a `pl.Expr`, so it composes inside `select`/`with_columns`
  and runs lazily. Series and frame inputs are not supported: passing a
  `pl.Series`, `pl.DataFrame`, or `pl.LazyFrame` raises `TypeError`. Evaluate
  the expression on a frame when an eager result is needed, for example
  `values.to_frame("close").select(sma("close", 3)).to_series()`.
- **Warm-up.** Every indicator emits exactly the TA-Lib lookback as leading
  nulls:

  | Indicator | Leading nulls |
  | --------- | ------------- |
  | `bop`, `ad`, `obv`, `vpt`, `nvi`, `cumulative_return` | none |
  | `avgprice`, `medprice`, `typprice`, `wclprice` | none |
  | `sma`, `wma`, `ema`, `bbands`, `cci`, `donchian`, `midpoint`, `midprice`, `trima`, `willr` | `window - 1` |
  | `plus_dm`, `minus_dm` | `window - 1` |
  | `cmf`, `vwap` | `window - 1` |
  | `ao` | `slow_period - 1` |
  | `dema` | `2 * (window - 1)` |
  | `tema` | `3 * (window - 1)` |
  | `t3` | `6 * (window - 1)` |
  | `trix` | `3 * (window - 1) + 1` |
  | `true_range` | `1` |
  | `sar`, `sarext` | `1` |
  | `daily_return`, `daily_log_return` | `1` |
  | `rsi`, `cmo`, `mfi`, `atr`, `natr`, `supertrend`, `kama`, `mom`, `roc`, `rocp`, `rocr`, `rocr100` | `window` |
  | `aroon`, `aroonosc`, `dx` | `window` |
  | `fi`, `eom`, `vortex` | `window` |
  | `ulcer` | `2 * (window - 1)` |
  | `dpo` | `max(window - 1, window // 2 + 1)` |
  | `tsi` | `slow_period + fast_period - 1` |
  | `mass` | `2 * (fast_period - 1) + slow_period - 1` |
  | `adx` (`plus_di`, `minus_di`) | `window` |
  | `adx` (`adx`) | `2 * window - 1` |
  | `adxr` | `3 * window - 2` |
  | `stoch` | `(fastk_period - 1) + (slowk_period - 1) + (slowd_period - 1)` |
  | `stochf` | `(fastk_period - 1) + (fastd_period - 1)` |
  | `stochrsi` | `window + (fastk_period - 1) + (fastd_period - 1)` |
  | `macd`, `macdfix` | `(slow_period - 1) + (signal_period - 1)` |
  | `macdext`, `apo`, `ppo`, `pvo` | the chosen averages' own lookbacks |
  | `adosc` | `slow_period - 1` |
  | `ultosc` | `max(short, medium, long)` |
  | `mavp` | `max_period - 1` |
  | `ht_dcperiod`, `ht_phasor`, `mama` | `32` |
  | `ht_dcphase`, `ht_sine`, `ht_trendmode`, `ht_trendline` | `63` |
  | `keltner` | per field; see below |
  | `ichimoku` | per field; see below |
  | `kst` | per field; see below |
  | `stc` | the chained averages' own lookbacks |

  There is no `min_periods` parameter, so a simple and an exponential moving
  average of the same window line up row for row.
- **Multi-field outputs follow TA-Lib's emission.** Where TA-Lib produces a
  struct's fields from one function, they start on the same row: this holds back
  `stoch`'s `k` until `d` exists, and `macd`'s `macd` line until its `signal`
  exists. Where the fields correspond to *separate* TA-Lib functions with
  different lookbacks, they keep their own warm-ups rather than discarding good
  data — `adx`'s `plus_di` and `minus_di` start `window - 1` rows before `adx`.
  `donchian` and `ichimoku` likewise let each field reflect only the inputs it
  actually depends on.
- **Nulls.** A null input never silently disappears. See the null handling notes
  under each indicator for the exact rule.
- **Output dtype.** Always `Float64`, including for integer inputs.
- **Insufficient data.** If the input is shorter than the warm-up length, the
  whole result is null.

Throughout, $P_t$ is the input value at row $t$ (zero-indexed) and $n$ is
`window`.

## Overlay Indicators

Everything from here on is drawn on the price axis. For the oscillators and the
volume, volatility, cycle, and returns indicators, see
[indicators-oscillators.md](indicators-oscillators.md).

## SMA — Simple Moving Average

The unweighted arithmetic mean of the most recent $n$ values:

$$\mathrm{SMA}_t = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}, \qquad t \ge n-1$$

Every value in the window carries the same weight $1/n$, so an SMA reacts to a
price change only as fast as that change enters and then leaves the window.

```python
import polars as pl
from polars_ta import sma

df = pl.DataFrame({"close": [1.0, 3.0, 2.0, 6.0, 5.0]})
df.with_columns(sma("close", 3).alias("sma_3"))
```

| `close` | `sma_3` | Calculation |
| ------- | ------- | ----------- |
| 1.0 | `null` | warm-up |
| 3.0 | `null` | warm-up |
| 2.0 | 2.000000 | $(1+3+2)/3$ |
| 6.0 | 3.666667 | $(3+2+6)/3$ |
| 5.0 | 4.333333 | $(2+6+5)/3$ |

**Null handling.** $\mathrm{SMA}_t$ is null whenever any of the $n$ values in its
window is null. A single null therefore blanks exactly $n$ output rows; the
series recovers as soon as the null leaves the window.

**Implementation.** `pl.Expr.rolling_mean(window_size=n, min_samples=n)`. The
`min_samples=n` setting produces both the warm-up nulls and the null
propagation, because a null does not count as an observed sample.

## WMA — Weighted Moving Average

A linearly weighted average: the most recent value carries weight $n$, the one
before it $n-1$, down to weight $1$ at the oldest value in the window.

$$\mathrm{WMA}_t = \frac{\sum_{i=0}^{n-1} (n-i)\,P_{t-i}}{\sum_{k=1}^{n} k}
= \frac{2}{n(n+1)} \sum_{i=0}^{n-1} (n-i)\,P_{t-i}$$

The linear ramp makes a WMA respond faster than an SMA of the same window while
still forgetting a value completely once it leaves the window — unlike an EMA,
whose weights never reach zero.

```python
from polars_ta import wma

df.with_columns(wma("close", 3).alias("wma_3"))
```

| `close` | `wma_3` | Calculation |
| ------- | ------- | ----------- |
| 1.0 | `null` | warm-up |
| 3.0 | `null` | warm-up |
| 2.0 | 2.166667 | $(1\cdot1 + 2\cdot3 + 3\cdot2)/6$ |
| 6.0 | 4.166667 | $(1\cdot3 + 2\cdot2 + 3\cdot6)/6$ |
| 5.0 | 4.833333 | $(1\cdot2 + 2\cdot6 + 3\cdot5)/6$ |

**Null handling.** Identical to the SMA: a row is null whenever any of the $n$
values in its window is null.

**Implementation.** `pl.Expr.rolling_mean(window_size=n, weights=[1..n])`, which
normalizes by the weight sum. Polars panics rather than raising when weighted
rolling aggregations meet nulls, so the input is cast to `Float64` and
null-filled first, and rows whose window contained a null are masked back to
null afterwards using a rolling count of nulls. The mask reproduces exactly the
SMA null rule, so no otherwise-valid row is discarded.

## EMA — Exponential Moving Average

A recursive average whose weights decay geometrically into the past:

$$\mathrm{EMA}_t = \alpha P_t + (1-\alpha)\,\mathrm{EMA}_{t-1}$$

The smoothing factor defaults to the standard window-derived value

$$\alpha = \frac{2}{n+1}$$

and can be overridden with the `alpha` argument, which must satisfy
$0 < \alpha \le 1$. Larger $\alpha$ means faster response to new values;
$\alpha = 1$ reduces the EMA to the input itself.

Because the recursion has no natural starting point, implementations differ in
how they seed it. `polars_ta` exposes all three common conventions through the
`mode` argument.

### `mode="talib"` (default)

Seeds the recursion with the simple moving average of the first complete
window, then recurses:

$$\mathrm{EMA}_{n-1} = \frac{1}{n}\sum_{i=0}^{n-1} P_i, \qquad
\mathrm{EMA}_t = \alpha P_t + (1-\alpha)\,\mathrm{EMA}_{t-1} \;\; (t \ge n)$$

This matches TA-Lib's `EMA` and the default of the pandas-based indicator
libraries listed in `KNOWLEDGE.md`.

```python
from polars_ta import ema

df.with_columns(ema("close", 3).alias("ema_3"))
```

With $n = 3$, $\alpha = 0.5$ and `close = [1, 3, 2, 6, 5]`:

| `close` | `ema_3` | Calculation |
| ------- | ------- | ----------- |
| 1.0 | `null` | warm-up |
| 3.0 | `null` | warm-up |
| 2.0 | 2.0 | seed $(1+3+2)/3$ |
| 6.0 | 4.0 | $0.5 \cdot 6 + 0.5 \cdot 2$ |
| 5.0 | 4.5 | $0.5 \cdot 5 + 0.5 \cdot 4$ |

### `mode="recursive"`

Seeds with the first value, $\mathrm{EMA}_0 = P_0$, and applies the same
recursion from there. Equivalent to pandas `ewm(alpha=..., adjust=False)`. The
first $n-1$ rows are still suppressed, so the visible output starts once the
seed has been smoothed $n-1$ times.

### `mode="adjust"`

Uses the finite closed form, which needs no seed at all:

$$\mathrm{EMA}_t = \frac{\sum_{i=0}^{t} (1-\alpha)^i P_{t-i}}
{\sum_{i=0}^{t} (1-\alpha)^i}$$

Equivalent to pandas `ewm(alpha=..., adjust=True)`. Each output is a true
weighted average of all available history, which removes the start-up bias of
the recursive forms at the cost of depending on the full history.

### Mode equivalences

| `mode` | Seed | TA-Lib | pandas |
| ------ | ---- | ------ | ------ |
| `"talib"` | SMA of first $n$ values | `EMA` | `ewm(adjust=False)` applied after an SMA seed |
| `"recursive"` | $P_0$ | — | `ewm(adjust=False)` |
| `"adjust"` | none (closed form) | — | `ewm(adjust=True)` |

All three modes emit `window - 1` leading nulls, which is stricter than the
pandas default of `min_periods=0`.

**Null handling.** A null input produces a null output at that row and is
treated as a missing observation rather than a zero: the weighting of later
rows accounts for the gap, matching pandas' default `ignore_na=False`. In
`"talib"` mode a null inside the seed window delays the seed until the first
complete, null-free window is available.

**Implementation.** `pl.Expr.ewm_mean` with `ignore_nulls=False`. For
`"recursive"` and `"adjust"` the expression is applied directly with
`min_samples=window`. For `"talib"` the input is first rewritten so that rows
before the seed are null and the seed row holds the rolling mean, after which
`ewm_mean(adjust=False, min_samples=1)` reproduces the TA-Lib recursion without
a Python-level loop.

## DEMA — Double Exponential Moving Average

An EMA lags the price; a second EMA applied to the first lags it by roughly the
same amount again. Subtracting that second-order lag cancels most of the first:

$$\mathrm{DEMA}_t = 2\,\mathrm{EMA}^{(1)}_t - \mathrm{EMA}^{(2)}_t,
\qquad \mathrm{EMA}^{(2)} = \mathrm{EMA}\!\left(\mathrm{EMA}^{(1)}\right)$$

Both passes use the same $n$ and the same $\alpha$. The result is faster than a
plain EMA at the cost of overshooting sharp moves.

```python
from polars_ta import dema

df.with_columns(dema("close", 3).alias("dema_3"))
```

With $n = 3$, $\alpha = 0.5$ and `close = [1, 3, 2, 6, 5]`:

| row | $\mathrm{EMA}^{(1)}$ | $\mathrm{EMA}^{(2)}$ | `dema_3` |
| --- | -------------------- | -------------------- | -------- |
| 0-3 | — | `null` | `null` |
| 4 | 4.5 | 3.5 | 5.5 |

$\mathrm{EMA}^{(1)}$ starts at row 2 with the seed $(1+3+2)/3 = 2$, so
$\mathrm{EMA}^{(2)}$ cannot seed until row 4, where it averages
$(2 + 4 + 4.5)/3 = 3.5$. The DEMA is then $2 \cdot 4.5 - 3.5 = 5.5$.

**Warm-up.** $2(n-1)$ leading nulls, matching the TA-Lib `DEMA` lookback.

**Null handling.** Inherited from `ema`. Because the second pass sees the first
pass's nulls, an interior null delays the chain rather than corrupting it.

**Implementation.** `ema` is applied twice; the nested call receives the first
pass as an expression, and the `"talib"` seeding rule — seed at the first
complete, null-free window — automatically produces the correct $2(n-1)$
warm-up without any explicit offset bookkeeping.

## TEMA — Triple Exponential Moving Average

Extends the same lag-cancellation idea to a third pass:

$$\mathrm{TEMA}_t = 3\,\mathrm{EMA}^{(1)}_t - 3\,\mathrm{EMA}^{(2)}_t
+ \mathrm{EMA}^{(3)}_t$$

The coefficients $3, -3, 1$ come from expanding $1 - (1 - E)^3$, where $E$ is
the EMA operator; DEMA is the same expansion truncated at two terms.

```python
from polars_ta import tema

df.with_columns(tema("close", 3).alias("tema_3"))
```

**Warm-up.** $3(n-1)$ leading nulls, matching the TA-Lib `TEMA` lookback.

**Null handling and implementation.** As for DEMA, with three chained `ema`
passes.

### Choosing between them

| Indicator | Lag | Overshoot | Smoothness |
| --------- | --- | --------- | ---------- |
| SMA | highest | none | highest |
| WMA | medium | none | high |
| EMA | medium | none | high |
| DEMA | low | some | medium |
| TEMA | lowest | most | lowest |

`dema` and `tema` accept the same `alpha` and `mode` arguments as `ema`, and
every pass uses the chosen convention.

## BBANDS — Bollinger Bands

A moving average with a volatility envelope: the band width expands when the
recent standard deviation rises and contracts when the market is quiet.

$$\mathrm{middle}_t = \mathrm{SMA}_t, \qquad
\sigma_t = \sqrt{\frac{1}{n - \mathrm{ddof}} \sum_{i=0}^{n-1}
\left(P_{t-i} - \mathrm{middle}_t\right)^2}$$

$$\mathrm{upper}_t = \mathrm{middle}_t + k\,\sigma_t, \qquad
\mathrm{lower}_t = \mathrm{middle}_t - k\,\sigma_t$$

where $k$ is `num_std`, defaulting to `2.0`.

```python
import polars as pl
from polars_ta import bbands

df = pl.DataFrame({"close": [1.0, 3.0, 2.0, 6.0, 5.0]})
df.with_columns(bbands("close", 3).alias("bb")).unnest("bb")
```

`bbands` returns a **struct** with fields `lower`, `middle`, and `upper`, so one
call stays one expression. Unnest it as above, or pull a single band out:

```python
df.with_columns(bbands("close", 20).struct.field("upper").alias("bb_upper"))
```

**Degrees of freedom.** `ddof=0` (the default) is the population deviation used
by TA-Lib. Pass `ddof=1` for the sample deviation, which is Polars' own default
for `rolling_std`. `ddof` must satisfy `0 <= ddof < window`.

**Warm-up.** $n-1$ leading nulls, as for the SMA.

**Null handling.** All three fields are null whenever any of the $n$ values in
the window is null.

**Implementation.** `rolling_mean` and `rolling_std`, both with
`min_samples=window`, combined into a `pl.struct`. A constant window gives
$\sigma = 0$, collapsing all three bands onto the same value rather than
producing a null.

## DONCHIAN — Donchian Channels

The simplest channel there is: the extremes of the last $n$ bars.

$$\mathrm{upper}_t = \max(H_{t-n+1 \ldots t}), \qquad
\mathrm{lower}_t = \min(L_{t-n+1 \ldots t}), \qquad
\mathrm{middle}_t = \frac{\mathrm{upper}_t + \mathrm{lower}_t}{2}$$

A close at the upper edge is a new $n$-bar high — the classic breakout entry of
the Turtle trading system.

```python
from polars_ta import donchian

ohlc.with_columns(donchian("high", "low", 20).alias("dc")).unnest("dc")
```

Returns a **struct** with fields `lower`, `middle`, `upper`, matching the
`bbands` field naming.

**Warm-up.** $n - 1$ leading nulls. Unlike Bollinger Bands, the two edges come
from different input columns, so each blanks independently: a null in `high`
nulls `upper` and `middle` but leaves `lower` intact.

**Implementation.** `rolling_max` and `rolling_min` with `min_samples=window`.
This is not a TA-Lib function; the TA-Lib equivalents are the separate `MAX`
and `MIN`.

## KELTNER — Keltner Channels

An exponential average with an envelope scaled by the Average True Range:

$$\mathrm{middle}_t = \mathrm{EMA}_n(C)_t, \qquad
\mathrm{upper}_t = \mathrm{middle}_t + k \cdot \mathrm{ATR}_m(H, L, C)_t, \qquad
\mathrm{lower}_t = \mathrm{middle}_t - k \cdot \mathrm{ATR}_m(H, L, C)_t$$

with $n$ = `window` (default 20), $m$ = `atr_window` (default 10), and $k$ =
`multiplier` (default 2.0).

```python
from polars_ta import keltner

ohlc.with_columns(keltner("high", "low", "close", 20).alias("kc")).unnest("kc")
```

Where Bollinger Bands scale their envelope by the standard deviation of the
**close**, Keltner scales it by the true range, which includes gaps and the
full bar. The channel is therefore steadier — it does not pinch shut during a
run of small closing changes that nonetheless saw wide bars — so a close
outside it is a stronger breakout signal than a Bollinger touch.

| | Bollinger Bands | Keltner Channels |
| --- | --- | --- |
| Centre | SMA of close | EMA of close |
| Width from | standard deviation of close | Average True Range |
| Reacts to gaps | no | yes |

**Separate periods.** The centre line and the envelope take independent
periods, because the smoothing that suits a trend line rarely suits a
volatility estimate. The defaults follow the common 20/10 convention.

**Warm-up.** Per field, as for `donchian`. `middle` depends only on the close
and starts after $n - 1$ rows; the two edges also need the ATR and so start
after $\max(n - 1, m)$ rows.

**Null handling.** Inherited from `ema` and `atr`. A null close blanks all
three fields; a null high or low blanks only the two edges.

**Implementation.** `_ema_expr` for the centre and the shared `_atr_expr` for
the width, so Wilder's smoothing stays a single implementation. This is not a
TA-Lib function.

## SUPERTREND

An ATR band that sits below price in an uptrend and above it in a downtrend,
flipping when price closes through it. Start from bands around the bar's median
price:

$$\mathrm{basic}^{\pm}_t = \frac{H_t + L_t}{2} \pm m \cdot \mathrm{ATR}_t$$

with $m$ = `multiplier`. The bands are then **ratcheted**: while the trend
holds, each band may only move toward price, never away from it. The trend
flips to up when the close exceeds the previous upper band, and to down when it
falls below the previous lower band. `supertrend` reports whichever band is
currently active.

```python
from polars_ta import supertrend

ohlc.with_columns(supertrend("high", "low", "close", 10, 3.0).alias("st")).unnest("st")
```

Returns a **struct** with `supertrend` (`Float64`, the active band in price
units) and `direction` (`Int8`, `1` while rising and `-1` while falling).

**Warm-up.** $n$ leading nulls, inherited from the ATR. The first valid bar is
seeded as an uptrend by convention.

**Implementation — note the cost.** The ratchet is a genuine sequential
recursion: each band depends on the previous band *and* on the previous
direction, which itself depends on the previous band. Unlike the EMA, which maps
onto Polars' native `ewm_mean`, there is no vectorized primitive for this. It is
therefore implemented with a Python scan inside `map_batches`. The function
still returns a `pl.Expr` and still composes inside `with_columns` and
`LazyFrame`, but the scan runs in Python, so `supertrend` is substantially
slower than the vectorized indicators. It shares that cost with `kama`, `mama`,
`sar`, `sarext`, and the `ht_*` cycle indicators, which are recursions for the
same reason. Avoid calling any of them in a tight loop over many groups.

## ICHIMOKU — Ichimoku Kinko Hyo

Five lines, built from midpoints of rolling high/low ranges rather than from
averages of closes:

| Field | Japanese name | Definition |
| ----- | ------------- | ---------- |
| `conversion` | Tenkan-sen | midpoint of the last `conversion_period` bars |
| `base` | Kijun-sen | midpoint of the last `base_period` bars |
| `span_a` | Senkou Span A | $(\text{conversion} + \text{base})/2$, shifted **forward** |
| `span_b` | Senkou Span B | midpoint of the last `span_b_period` bars, shifted **forward** |
| `lagging` | Chikou Span | the close, shifted **backward** |

where the midpoint of a window is $\bigl(\max H + \min L\bigr)/2$ and both
shifts are by `displacement` bars. The region between `span_a` and `span_b` is
the *cloud* (kumo); price above the cloud is bullish, below is bearish.

```python
from polars_ta import ichimoku

ohlc.with_columns(ichimoku("high", "low", "close").alias("ich")).unnest("ich")
```

Defaults are the classic $9, 26, 52$ with a displacement of $26$.

> **Lookahead warning.** `lagging` is the close shifted *backward*, so the value
> on row $t$ is the close from row $t + \texttt{displacement}$ — data from the
> future relative to that row. This is correct for plotting, which is what the
> line is for, but feeding it into a backtest signal without re-shifting it is a
> lookahead bug that will manufacture profits.

**Warm-up.** Each field carries its own, since they depend on different
lookbacks: `conversion_period - 1` for `conversion`, `base_period - 1` for
`base`, and those plus `displacement` for the two spans. `lagging` instead has
`displacement` **trailing** nulls.

**Truncation.** On a chart the leading spans project `displacement` bars past
the last candle. A column cannot be longer than its frame, so that projection is
simply absent here — to see it, extend the frame with empty rows before calling.

**Null handling.** As for any rolling extreme: a null blanks the windows
overlapping it, per field.

## TRIMA — Triangular Moving Average

An SMA of an SMA, which convolves two rectangular weightings into a triangular
one that peaks at the centre of the window:

$$\mathrm{TRIMA}_t = \mathrm{SMA}_{m}\!\left(\mathrm{SMA}_{k}(P)\right)_t$$

For an odd $n$ both passes use $k = m = (n+1)/2$; for an even $n$ they use
$k = n/2 + 1$ and $m = n/2$. Either way the combined lookback is $n - 1$ and the
weights rise $1, 2, \dots$ to the middle and fall away again, matching TA-Lib's
normalizing factor of $(k)^2$ and $k(k+1)$ respectively.

**Null handling and implementation.** Two chained `sma` passes, so the SMA null
rule applies twice.

## T3 — Tillson Moving Average

A weighted blend of the third through sixth passes of a repeated EMA:

$$\mathrm{T3}_t = c_1 e^{(6)}_t + c_2 e^{(5)}_t + c_3 e^{(4)}_t + c_4 e^{(3)}_t$$

$$c_1 = -v^3,\quad c_2 = 3v^2 + 3v^3,\quad
c_3 = -6v^2 - 3v - 3v^3,\quad c_4 = 1 + 3v + 3v^2 + v^3$$

where $v$ is `vfactor`, which must lie in $[0, 1]$. At $v = 0$ the coefficients
collapse to $c_4 = 1$ and T3 equals the plain triple-smoothed EMA; at $v = 1$ it
is the most aggressive lag cancellation.

**Warm-up.** $6(n-1)$, from the six chained passes — including at $v = 0$, where
the sixth pass still gates the output even though its coefficient is zero. This
matches the TA-Lib lookback, which likewise ignores `vfactor`.

## KAMA — Kaufman Adaptive Moving Average

An EMA whose smoothing factor is chosen bar by bar from an *efficiency ratio*:
the net move over the window divided by the total distance travelled to achieve
it.

$$\mathrm{ER}_t = \frac{\left|P_t - P_{t-n}\right|}
{\sum_{i=0}^{n-1}\left|P_{t-i} - P_{t-i-1}\right|}$$

$$\alpha_t = \left(\mathrm{ER}_t\left(\tfrac{2}{f+1} - \tfrac{2}{s+1}\right)
+ \tfrac{2}{s+1}\right)^{2}, \qquad
\mathrm{KAMA}_t = \mathrm{KAMA}_{t-1} + \alpha_t\left(P_t - \mathrm{KAMA}_{t-1}\right)$$

with $f$ = `fast_period` (default 2) and $s$ = `slow_period` (default 30). A
straight-line move scores $\mathrm{ER} = 1$ and smooths at the fast rate; a
directionless one scores $0$ and nearly freezes. Squaring $\alpha$ biases the
average strongly toward the slow end unless the move is genuinely efficient.

**Degenerate windows.** TA-Lib treats a move at least as large as the path that
produced it as perfectly efficient, which also covers zero volatility; both
report $\mathrm{ER} = 1$.

**Warm-up.** $n$ leading nulls. The recursion is seeded with $P_{n-1}$, the
value immediately before the first output.

**Implementation.** The efficiency ratio and $\alpha$ are plain expressions, but
the recursion itself has a *varying* $\alpha$ and so cannot use `ewm_mean`. It
runs a Python scan inside `map_batches`. A null restarts the recursion rather
than corrupting it.

## MAMA — MESA Adaptive Moving Average

Ehlers' adaptive average, driven by the Hilbert transform (see the cycle
section). The smoothing factor comes from how fast the measured phase is
turning:

$$\alpha_t = \max\!\left(\frac{\texttt{fast\_limit}}{\Delta\phi_t},\,
\texttt{slow\_limit}\right), \qquad \Delta\phi_t = \max(\phi_{t-1} - \phi_t,\, 1)$$

$$\mathrm{MAMA}_t = \alpha_t P_t + (1-\alpha_t)\mathrm{MAMA}_{t-1}, \qquad
\mathrm{FAMA}_t = \tfrac{\alpha_t}{2}\mathrm{MAMA}_t
+ \left(1 - \tfrac{\alpha_t}{2}\right)\mathrm{FAMA}_{t-1}$$

A sharp turn in phase marks a new trend and lets the average jump; a steady
phase slows it to `slow_limit`. `fama` is the half-speed follower whose
crossings with `mama` are the usual signal. Returns a struct of `mama`/`fama`.

**Warm-up.** 32 leading nulls, as for every short-lookback Hilbert indicator.

## MA and MAVP — Dispatched Moving Averages

`ma(column, window, ma_type=...)` selects one of `polars_ta.MA_TYPES` at
runtime: `"sma"`, `"ema"`, `"wma"`, `"dema"`, `"tema"`, `"trima"`, `"kama"`,
`"mama"`, `"t3"`. Each keeps its own defaults and its own warm-up; `"mama"`
returns the MAMA line and ignores `window`, as TA-Lib does.

`mavp(column, periods, min_period, max_period, ma_type=...)` reads the period
from a second column. Each row's period is truncated to an integer and clamped
to $[\texttt{min\_period}, \texttt{max\_period}]$. One average is built per
candidate period and selected row by row, so the expression grows linearly in
$\texttt{max\_period} - \texttt{min\_period}$ — keep that span small, especially
for the chained averages. Output starts only once the `max_period` average is
available, so the warm-up does not change from row to row.

## MIDPOINT and MIDPRICE

$$\mathrm{MIDPOINT}_t = \frac{\max_{i<n} P_{t-i} + \min_{i<n} P_{t-i}}{2},
\qquad \mathrm{MIDPRICE}_t = \frac{\max_{i<n} H_{t-i} + \min_{i<n} L_{t-i}}{2}$$

Unlike a moving average these ignore everything between the two extremes, so
they move only when a new extreme enters or an old one leaves the window.

## SAR and SAREXT — Parabolic SAR

A trailing stop that accelerates toward price. While long, the stop rises each
bar toward the highest high seen in the trade:

$$\mathrm{SAR}_{t+1} = \mathrm{SAR}_t + \mathrm{AF}_t\left(\mathrm{EP}_t - \mathrm{SAR}_t\right)$$

where $\mathrm{EP}$ is the extreme point of the current trade and $\mathrm{AF}$
starts at `acceleration` and grows by `acceleration` at every new extreme, up to
`maximum`. The stop is then clamped inside the current and previous bar's range
so it can never sit inside the bar it is protecting. When price touches the
stop, the trade flips: the stop jumps to the old extreme point and the
acceleration resets.

The opening direction comes from the sign of the first $-\mathrm{DM}$.

`sarext` separates every acceleration parameter for the long and short sides,
accepts an explicit `start_value` (positive forces a long start, negative a
short one), widens the stop on a reversal by `offset_on_reverse`, and reports
short readings **negated** so the sign carries the current side.

**Warm-up.** One leading null; a null high or low ends the scan, leaving every
later row null.

**Implementation.** A genuine sequential recursion with no Polars primitive, so
it runs a Python scan inside `map_batches`.

## AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE — Price Transforms

Four ways of collapsing a bar into a single number. They are not signals; they
are the price series other indicators consume in place of the raw close.

$$\mathrm{AVGPRICE}_t = \frac{O_t + H_t + L_t + C_t}{4}, \qquad
\mathrm{MEDPRICE}_t = \frac{H_t + L_t}{2}$$

$$\mathrm{TYPPRICE}_t = \frac{H_t + L_t + C_t}{3}, \qquad
\mathrm{WCLPRICE}_t = \frac{H_t + L_t + 2C_t}{4}$$

`typprice` is what `cci`, `mfi`, and `vwap` are built on; `medprice` is what
`ao` and `midprice` use. There is no warm-up, and any null input nulls the row.

---

Momentum, volume, volatility, cycle, and returns indicators continue in
[indicators-oscillators.md](indicators-oscillators.md).

