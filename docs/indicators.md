# Indicator Reference

Formulas and conventions for every indicator implemented in `polars_ta`.
Indicators are grouped by how they are charted: *overlay* indicators share the
price axis, *momentum* oscillators occupy a separate pane, *volume* indicators
weight movement by how much traded, *volatility* indicators measure the size of
movement, and *cycle* indicators measure its rhythm. Each group is a subpackage
(`polars_ta.overlay`, `polars_ta.momentum`, `polars_ta.volume`,
`polars_ta.volatility`, `polars_ta.cycle`), and every public indicator is also
re-exported from the package root.

Every TA-Lib function listed in `indicators.md` at the repository root is
implemented.

| Indicator | Module | Inputs | Output |
| --------- | ------ | ------ | ------ |
| `sma`, `wma`, `ema`, `dema`, `tema`, `trima`, `t3` | `overlay.ma` | one column | one `Float64` column |
| `kama` | `overlay.adaptive` | one column | one `Float64` column |
| `mama` | `overlay.adaptive` | one column | struct of two `Float64` fields |
| `ma`, `mavp` | `overlay.dispatch` | one or two columns | one `Float64` column |
| `midpoint` | `overlay.midpoint` | one column | one `Float64` column |
| `midprice` | `overlay.midpoint` | high, low | one `Float64` column |
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
| `apo`, `ppo` | `momentum.price_oscillator` | one column | one `Float64` column |
| `trix` | `momentum.trix` | one column | one `Float64` column |
| `ultosc` | `momentum.ultosc` | high, low, close | one `Float64` column |
| `ad`, `adosc` | `volume.flow` | high, low, close, volume | one `Float64` column |
| `obv` | `volume.flow` | close, volume | one `Float64` column |
| `true_range`, `atr`, `natr` | `volatility.atr` | high, low, close | one `Float64` column |
| `ht_dcperiod`, `ht_dcphase`, `ht_trendline` | `cycle.hilbert` | one column | one `Float64` column |
| `ht_phasor`, `ht_sine` | `cycle.hilbert` | one column | struct of two `Float64` fields |
| `ht_trendmode` | `cycle.hilbert` | one column | one `Int8` column |


## Shared Conventions

- **Input forms.** Each indicator accepts a column name (`str`), a `pl.Expr`, or
  a `pl.Series`. Name and expression inputs return a `pl.Expr`, so they compose
  inside `select`/`with_columns` and run lazily. A series input is evaluated
  eagerly and returns a `pl.Series` carrying the input series name. For
  multi-input indicators every argument must be the same kind — mixing a
  `pl.Series` with a column name raises `TypeError`.
- **Warm-up.** Every indicator emits exactly the TA-Lib lookback as leading
  nulls:

  | Indicator | Leading nulls |
  | --------- | ------------- |
  | `bop`, `ad`, `obv` | none |
  | `sma`, `wma`, `ema`, `bbands`, `cci`, `donchian`, `midpoint`, `midprice`, `trima`, `willr` | `window - 1` |
  | `plus_dm`, `minus_dm` | `window - 1` |
  | `dema` | `2 * (window - 1)` |
  | `tema` | `3 * (window - 1)` |
  | `t3` | `6 * (window - 1)` |
  | `trix` | `3 * (window - 1) + 1` |
  | `true_range` | `1` |
  | `sar`, `sarext` | `1` |
  | `rsi`, `cmo`, `mfi`, `atr`, `natr`, `supertrend`, `kama`, `mom`, `roc`, `rocp`, `rocr`, `rocr100` | `window` |
  | `aroon`, `aroonosc`, `dx` | `window` |
  | `adx` (`plus_di`, `minus_di`) | `window` |
  | `adx` (`adx`) | `2 * window - 1` |
  | `adxr` | `3 * window - 2` |
  | `stoch` | `(fastk_period - 1) + (slowk_period - 1) + (slowd_period - 1)` |
  | `stochf` | `(fastk_period - 1) + (fastd_period - 1)` |
  | `stochrsi` | `window + (fastk_period - 1) + (fastd_period - 1)` |
  | `macd`, `macdfix` | `(slow_period - 1) + (signal_period - 1)` |
  | `macdext`, `apo`, `ppo` | the chosen averages' own lookbacks |
  | `adosc` | `slow_period - 1` |
  | `ultosc` | `max(short, medium, long)` |
  | `mavp` | `max_period - 1` |
  | `ht_dcperiod`, `ht_phasor`, `mama` | `32` |
  | `ht_dcphase`, `ht_sine`, `ht_trendmode`, `ht_trendline` | `63` |
  | `keltner` | per field; see below |
  | `ichimoku` | per field; see below |

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

## RSI — Relative Strength Index

The share of recent price movement that was upward, scaled to $[0, 100]$.
Each bar's change is split into a gain and a loss:

$$G_t = \max(P_t - P_{t-1},\, 0), \qquad L_t = \max(P_{t-1} - P_t,\, 0)$$

Both are smoothed with **Wilder's moving average** — an EMA with
$\alpha = 1/n$ — seeded with the mean of the first $n$ changes:

$$\bar{G}_t = \frac{(n-1)\,\bar{G}_{t-1} + G_t}{n}, \qquad
\mathrm{RSI}_t = 100 \cdot \frac{\bar{G}_t}{\bar{G}_t + \bar{L}_t}$$

That last form is algebraically identical to the more familiar
$100 - \dfrac{100}{1 + \bar{G}_t/\bar{L}_t}$ but has no division by zero when
there were no losses.

```python
from polars_ta import rsi

df.with_columns(rsi("close", 14).alias("rsi_14"))
```

On Wilder's own worked example the first output is **70.4641**, matching the
published value and TA-Lib.

**Warm-up.** $n$ leading nulls — one row lost to the difference, then $n$
changes needed for the seed. This matches the TA-Lib `RSI` lookback.

**Degenerate windows.** When neither a gain nor a loss has been seen, the ratio
is $0/0$ and `rsi` reports the neutral **50.0**. TA-Lib changed this from `0.0`
to `50.0` precisely because zero reads as "extremely oversold".

**Null handling.** Inherited from `ema`: a null delays the seed to the first
complete, null-free window.

**Implementation.** `diff` then `clip` to split gains from losses — `clip`
preserves nulls, whereas a `when/otherwise` chain would turn the leading null
into `0.0` and shorten the warm-up. Each side is smoothed by calling the public
`ema` with `alpha=1/window`, which reuses the existing TA-Lib seeding rule
rather than reimplementing Wilder's recursion.

## MFI — Money Flow Index

A volume-weighted RSI. Each bar gets a typical price and a money flow:

$$\mathrm{TP}_t = \frac{H_t + L_t + C_t}{3}, \qquad
\mathrm{MF}_t = \mathrm{TP}_t \cdot V_t$$

The flow counts as positive or negative according to the direction the typical
price moved, and a bar with an unchanged typical price contributes nothing:

$$\mathrm{MF}^{+}_t = \begin{cases} \mathrm{MF}_t & \mathrm{TP}_t > \mathrm{TP}_{t-1} \\ 0 & \text{otherwise} \end{cases}
\qquad
\mathrm{MF}^{-}_t = \begin{cases} \mathrm{MF}_t & \mathrm{TP}_t < \mathrm{TP}_{t-1} \\ 0 & \text{otherwise} \end{cases}$$

$$\mathrm{MFI}_t = 100 \cdot \frac{\sum_{i=0}^{n-1} \mathrm{MF}^{+}_{t-i}}
{\sum_{i=0}^{n-1} \mathrm{MF}^{+}_{t-i} + \sum_{i=0}^{n-1} \mathrm{MF}^{-}_{t-i}}$$

Note that unlike RSI, MFI uses plain **rolling sums**, not Wilder smoothing.

```python
from polars_ta import mfi

df.with_columns(mfi("high", "low", "close", "volume", 14).alias("mfi_14"))
```

The four inputs are positional, in TA-Lib's order. They may be column names,
expressions, or series, but not a mixture of series and the other two.

**Warm-up.** $n$ leading nulls, matching the TA-Lib `MFI` lookback.

**Degenerate windows.** A window that received no money flow at all — flat
typical prices, or zero volume throughout — reports **0.0**, following TA-Lib.
This differs from `rsi`, which reports `50.0`; the two libraries' conventions
genuinely diverge here, and `polars_ta` matches each one.

**Null handling.** A null in any of the four inputs makes that bar's flow
unknown, which blanks the $n$ windows overlapping it.

**Difference from TA-Lib.** TA-Lib treats a typical-price move as "no movement"
when it falls inside a relative epsilon dead-zone. `polars_ta` compares against
exact zero instead, so the two can differ only for moves at float-rounding
scale.

## STOCH — Stochastic Oscillator

Where the close sits inside its recent high/low range, as a percentage. Raw
fast %K is

$$\mathrm{FastK}_t = 100 \cdot
\frac{C_t - \min(L_{t-n+1 \ldots t})}{\max(H_{t-n+1 \ldots t}) - \min(L_{t-n+1 \ldots t})}$$

with $n$ = `fastk_period`. Raw fast %K is noisy, so it is smoothed twice, each
time with a simple moving average:

$$\%K = \mathrm{SMA}(\mathrm{FastK},\ \texttt{slowk\_period}), \qquad
\%D = \mathrm{SMA}(\%K,\ \texttt{slowd\_period})$$

`stoch` returns these *slow* lines, which is what TA-Lib's `STOCH` returns and
what charting packages normally plot. %D is the signal line, drawn over %K.

```python
import polars as pl
from polars_ta import stoch

ohlc.with_columns(stoch("high", "low", "close", 5, 3, 3).alias("st")).unnest("st")
```

The result is a **struct** with fields `k` and `d`, both in $[0, 100]$.

**Warm-up.** $(\texttt{fastk\_period} - 1) + (\texttt{slowk\_period} - 1) +
(\texttt{slowd\_period} - 1)$ leading nulls, matching the TA-Lib lookback. Both
fields start on that same row: `k` is held back until `d` exists, because
TA-Lib emits the two lines aligned.

**Degenerate windows.** A window whose high equals its low has no range to
divide by; that bar's raw %K is **0.0**, following TA-Lib.

**Null handling.** A null in any input blanks the windows overlapping it, and
that gap then propagates through both smoothing passes.

**Implementation.** `rolling_min` and `rolling_max` with `min_samples`, then two
`rolling_mean` passes. Because `min_samples` does not count nulls, the warm-up
and the null propagation both fall out of the rolling calls.

## CCI — Commodity Channel Index

How far the typical price has strayed from its own recent mean, measured in
units of mean absolute deviation:

$$\mathrm{TP}_t = \frac{H_t + L_t + C_t}{3}, \qquad
M_t = \frac{1}{n}\sum_{i=0}^{n-1} \mathrm{TP}_{t-i}$$

$$D_t = \frac{1}{n}\sum_{i=0}^{n-1} \left| \mathrm{TP}_{t-i} - M_t \right|,
\qquad \mathrm{CCI}_t = \frac{\mathrm{TP}_t - M_t}{0.015 \cdot D_t}$$

The constant $0.015$ is Lambert's, chosen so that roughly 70–80% of readings
land in $[-100, 100]$; readings outside that band are the conventional
overbought/oversold signals. Unlike RSI or MFI, CCI is **unbounded**.

```python
from polars_ta import cci

ohlc.with_columns(cci("high", "low", "close", 14).alias("cci_14"))
```

**Warm-up.** $n - 1$ leading nulls, matching the TA-Lib `CCI` lookback.

**Degenerate windows.** A window with zero deviation reports **0.0** rather than
dividing by zero.

**Null handling.** Null whenever any of the $n$ values in the window is null,
exactly as for the SMA.

**Implementation.** Note that $D_t$ measures each value against the *current*
window's mean $M_t$, which changes every row — so it is **not** a rolling
aggregate and `rolling_std` is not a substitute (that would be a root-mean-square
deviation, not a mean-absolute one). It expands instead into one shifted term
per period: $n$ `shift` expressions summed together. This is $O(n)$ expressions,
which is fine for conventional periods but worth knowing before passing a very
large `window`.

## MACD — Moving Average Convergence/Divergence

The gap between a fast and a slow EMA, compared against a smoothed copy of
itself:

$$\mathrm{MACD}_t = \mathrm{EMA}(P, \texttt{fast})_t -
\mathrm{EMA}(P, \texttt{slow})_t$$

$$\mathrm{signal}_t = \mathrm{EMA}(\mathrm{MACD}, \texttt{signal})_t, \qquad
\mathrm{histogram}_t = \mathrm{MACD}_t - \mathrm{signal}_t$$

The MACD line crossing its signal line is the classic trade trigger; the
histogram makes the size and direction of that gap visible.

```python
from polars_ta import macd

df.with_columns(macd("close", 12, 26, 9).alias("m")).unnest("m")
```

The result is a **struct** with fields `macd`, `signal`, and `histogram`.

**Warm-up.** $(\texttt{slow} - 1) + (\texttt{signal} - 1)$ leading nulls — 33
rows for the default $12, 26, 9$ — matching the TA-Lib `MACD` lookback.

**All three fields start together.** The MACD line alone would be available
$\texttt{signal} - 1$ rows earlier, but TA-Lib emits the three series aligned,
so `macd` and `histogram` are held back until `signal` exists.

**Modes.** `mode` is passed through to every EMA pass, so `"recursive"` and
`"adjust"` give the pandas conventions described under EMA above.

**Null handling and implementation.** Three calls to the public `ema`. The
TA-Lib seeding rule — seed at the first complete, null-free window — produces
the correct combined warm-up with no explicit offset arithmetic.

## TRANGE / ATR — True Range and Average True Range

True Range is the widest of three spans, so that an overnight gap counts as
movement even though no trading happened across it:

$$\mathrm{TR}_t = \max\left(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|\right)$$

ATR is Wilder's smoothing of that series — an EMA with $\alpha = 1/n$, seeded
with the mean of the first $n$ true ranges:

$$\mathrm{ATR}_t = \frac{(n-1)\,\mathrm{ATR}_{t-1} + \mathrm{TR}_t}{n}$$

ATR is a pure *volatility* measure: it is always non-negative and says nothing
about direction. It is reported in price units, so it is not comparable across
instruments without normalizing.

```python
import polars as pl
from polars_ta import atr, true_range

ohlc.with_columns(
    true_range("high", "low", "close").alias("tr"),
    atr("high", "low", "close", 14).alias("atr_14"),
)
```

**Warm-up.** `true_range` emits 1 leading null — there is no previous close for
the first bar — and `atr` emits $n$, both matching the TA-Lib lookbacks.

**Null handling.** A bar is null unless its high, low, and *previous* close are
all known. Note that `pl.max_horizontal` **ignores** nulls rather than
propagating them, so the inputs are guarded explicitly; relying on the default
would silently emit $H_t - L_t$ for the very first bar.

## ADX / DI — Average Directional Index

Measures *trend strength* without regard to direction. Each bar's move is
assigned to at most one side, whichever is larger:

$$\mathrm{+DM}_t = \begin{cases} H_t - H_{t-1} & \text{if } H_t - H_{t-1} > L_{t-1} - L_t \text{ and } > 0 \\ 0 & \text{otherwise}\end{cases}$$

$$\mathrm{-DM}_t = \begin{cases} L_{t-1} - L_t & \text{if } L_{t-1} - L_t > H_t - H_{t-1} \text{ and } > 0 \\ 0 & \text{otherwise}\end{cases}$$

Wilder-smoothing both, and normalizing by smoothed true range, gives the two
directional indicators:

$$\mathrm{+DI}_t = 100 \cdot \frac{\overline{\mathrm{+DM}}_t}{\overline{\mathrm{TR}}_t},
\qquad \mathrm{-DI}_t = 100 \cdot \frac{\overline{\mathrm{-DM}}_t}{\overline{\mathrm{TR}}_t}$$

Their normalized gap is smoothed once more into ADX itself:

$$\mathrm{DX}_t = 100 \cdot \frac{|\mathrm{+DI}_t - \mathrm{-DI}_t|}{\mathrm{+DI}_t + \mathrm{-DI}_t},
\qquad \mathrm{ADX}_t = \mathrm{Wilder}(\mathrm{DX},\ n)_t$$

Conventionally, ADX above 25 means a trend worth following and below 20 means a
range; the crossing of `plus_di` and `minus_di` gives the direction.

```python
from polars_ta import adx

ohlc.with_columns(adx("high", "low", "close", 14).alias("a")).unnest("a")
```

The result is a **struct** with fields `adx`, `plus_di`, and `minus_di`.

**Warm-up.** `plus_di` and `minus_di` start after $n$ rows; `adx` needs a
further $n-1$ rows to seed its own smoothing, so it starts at $2n - 1$. These
are the TA-Lib `PLUS_DI`, `MINUS_DI`, and `ADX` lookbacks respectively, and the
fields are deliberately **not** aligned to the latest of them — doing so would
throw away $n-1$ rows of perfectly good indicator data.

**Degenerate windows.** Zero smoothed true range gives indicators of `0.0`;
zero $\mathrm{+DI} + \mathrm{-DI}$ gives a DX of `0.0`.

**Implementation.** Wilder's running *sum* (as TA-Lib keeps it) and Wilder's
running *average* differ by a constant factor of $n$, which cancels in the
$\mathrm{+DM}/\mathrm{TR}$ ratio — so the public `ema` with `alpha=1/n` can be
used for all three smoothing passes and still match TA-Lib exactly.

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
slower than every other indicator here. Avoid calling it in a tight loop over
many groups.

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
weights rise $1, 2, \dots$ to the middle and fall away again, matching TA-Lib''s
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

Ehlers'' adaptive average, driven by the Hilbert transform (see the cycle
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
from a second column. Each row''s period is truncated to an integer and clamped
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
`maximum`. The stop is then clamped inside the current and previous bar''s range
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

## CMO — Chande Momentum Oscillator

The same Wilder-smoothed gains and losses as the RSI, but centred on zero:

$$\mathrm{CMO}_t = 100 \cdot \frac{\bar{G}_t - \bar{L}_t}{\bar{G}_t + \bar{L}_t}$$

Away from a flat window this is exactly $2\,\mathrm{RSI}_t - 100$. A window with
neither a gain nor a loss reports **0.0** — note that the RSI reports `50.0` in
the same situation; both match TA-Lib and should not be harmonized.

## STOCHF, STOCHRSI, and WILLR

`stochf` is the unsmoothed pair: raw fast %K, and a single SMA pass for fast %D.
`stoch` smooths %K once more before reporting it.

`stochrsi` applies `stochf` to the RSI instead of to price, using the RSI as
high, low, and close alike. The RSI rarely reaches its own extremes, so reading
where it sits inside its recent range is far more sensitive than its level.

`willr` is the mirror of raw fast %K on a $[-100, 0]$ scale:

$$\mathrm{WILLR}_t = -100 \cdot
\frac{\max_{i<n} H_{t-i} - C_t}{\max_{i<n} H_{t-i} - \min_{i<n} L_{t-i}}$$

A flat range reports `0.0`, as in TA-Lib.

## AROON and AROONOSC

$$\mathrm{up}_t = \frac{100}{n}\left(n - \text{bars since the window high}\right),
\qquad \mathrm{down}_t = \frac{100}{n}\left(n - \text{bars since the window low}\right)$$

The window spans the last $n$ bars **plus** today, so $n+1$ rows. `up` reaches
`100` on the bar that sets a new high and decays by $100/n$ per bar since. A
reading says nothing about the size of a move, only its freshness.
`aroonosc` is `up - down`.

Repeated extremes count from the **most recent** occurrence, matching TA-Lib''s
`<=`/`>=` tie handling. The implementation finds the distance with a chain of
$n+1$ shifted equality tests against the rolling extreme, which is exact because
the rolling extreme is literally one of the values in the window.

## BOP — Balance Of Power

$$\mathrm{BOP}_t = \frac{C_t - O_t}{H_t - L_t}$$

A single-bar measure with no lookback at all: `+1` means the bar opened at its
low and closed at its high, `-1` the reverse. A bar with no range reports `0.0`.

## MOM, ROC, ROCP, ROCR, ROCR100

All five compare the current value against the one $n$ bars back:

| Function | Formula |
| -------- | ------- |
| `mom` | $P_t - P_{t-n}$ |
| `roc` | $\left(P_t / P_{t-n} - 1\right) \cdot 100$ |
| `rocp` | $\left(P_t - P_{t-n}\right) / P_{t-n}$ |
| `rocr` | $P_t / P_{t-n}$ |
| `rocr100` | $P_t / P_{t-n} \cdot 100$ |

The four ratio forms report **0.0** when the reference value is zero, rather
than dividing — TA-Lib''s convention. `mom` has no such case.

## APO and PPO — Price Oscillators

$$\mathrm{APO}_t = \mathrm{MA}^{\text{fast}}_t - \mathrm{MA}^{\text{slow}}_t,
\qquad \mathrm{PPO}_t = 100 \cdot \frac{\mathrm{APO}_t}{\mathrm{MA}^{\text{slow}}_t}$$

The same construction as the MACD line, but with a selectable `ma_type` and no
signal line. The periods are ordered before use, so swapping them changes
nothing. Expressing the gap as a percentage makes PPO comparable across
instruments, which a raw price difference is not. A zero slow average makes PPO
report `0.0`.

## TRIX

The one-period rate of change of a triple exponential average:

$$\mathrm{TRIX}_t = 100\left(\frac{e^{(3)}_t}{e^{(3)}_{t-1}} - 1\right)$$

Three smoothing passes strip out cycles shorter than `window`, so the remaining
slope isolates the dominant trend and crosses zero when it turns.

**Warm-up.** $3(n-1) + 1$ — the three EMA passes plus the one-bar difference.

## ULTOSC — Ultimate Oscillator

Buying pressure is measured against the lower of today''s low and yesterday''s
close, so a gap counts as part of the bar''s range:

$$\mathrm{BP}_t = C_t - \min(L_t, C_{t-1}), \qquad
\mathrm{TR}_t = \max(H_t, C_{t-1}) - \min(L_t, C_{t-1})$$

$$\mathrm{ULTOSC}_t = \frac{100}{7}\left(
4\frac{\sum_{i<n_1}\mathrm{BP}_{t-i}}{\sum_{i<n_1}\mathrm{TR}_{t-i}} +
2\frac{\sum_{i<n_2}\mathrm{BP}_{t-i}}{\sum_{i<n_2}\mathrm{TR}_{t-i}} +
\frac{\sum_{i<n_3}\mathrm{BP}_{t-i}}{\sum_{i<n_3}\mathrm{TR}_{t-i}}\right)$$

Williams combined three lookbacks precisely to avoid the false divergences a
single-period oscillator produces. A term whose range summed to zero contributes
nothing rather than dividing.

## PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI, DX, ADXR

The directional movement family shares one engine with `adx`. Raw directional
movement counts a bar toward only the larger outward move:

$$+\mathrm{DM}_t = \begin{cases} H_t - H_{t-1} & \text{if it exceeds } L_{t-1}-L_t \text{ and } 0 \\ 0 & \text{otherwise}\end{cases}$$

Both are accumulated with **Wilder''s running sum**, whose seed deliberately
holds $n-1$ terms rather than $n$:

$$S_{n-1} = \sum_{i=1}^{n-1} x_i, \qquad S_t = S_{t-1} - \frac{S_{t-1}}{n} + x_t$$

That is what gives `plus_dm` its $n-1$ lookback while `plus_di` has $n$: TA-Lib
performs one more smoothing step before emitting the first indicator.

$$\pm\mathrm{DI}_t = 100 \cdot \frac{S^{\pm\mathrm{DM}}_t}{S^{\mathrm{TR}}_t},
\qquad
\mathrm{DX}_t = 100 \cdot \frac{\left|+\mathrm{DI}_t - -\mathrm{DI}_t\right|}
{+\mathrm{DI}_t + -\mathrm{DI}_t}$$

`adx` is Wilder''s average of `dx`, and

$$\mathrm{ADXR}_t = \frac{\mathrm{ADX}_t + \mathrm{ADX}_{t-(n-1)}}{2}$$

with a lookback of $3n - 2$.

**Degenerate windows.** A zero range sum makes the indicators report `0.0`.
TA-Lib instead holds the previous ADX unchanged in that case; this is the one
deliberate divergence in the family.

**Implementation.** `_wilder_sum` reproduces the $n-1$ seed by rewriting the
input so that the seed row carries `rolling_sum(n - 1) / n` and then running
`ewm_mean(alpha=1/n)`, multiplying the result back by $n$. `window == 1` short
-circuits to the raw movement, matching TA-Lib''s unsmoothed special case.

## MACDEXT and MACDFIX

`macdext` is the MACD with a separately selectable average for each of its three
passes, through `fast_ma_type`, `slow_ma_type`, and `signal_ma_type` (all
defaulting to `"sma"`, as in TA-Lib). `macdfix` fixes the periods at the classic
12 and 26, leaving only the signal period tunable. Both return the same
`macd`/`signal`/`histogram` struct as `macd`, with all three fields starting on
the same row.

## AD, ADOSC, and OBV — Volume

Chaikin''s Accumulation/Distribution Line signs each bar''s volume by where the
close finished inside the bar''s range, then accumulates:

$$\mathrm{AD}_t = \mathrm{AD}_{t-1} +
\frac{(C_t - L_t) - (H_t - C_t)}{H_t - L_t} V_t$$

A bar with no range contributes nothing. `adosc` is the gap between two
exponential averages of that line, which turns an open-ended total into a
bounded oscillation:

$$\mathrm{ADOSC}_t = \mathrm{EMA}^{\text{fast}}(\mathrm{AD})_t
- \mathrm{EMA}^{\text{slow}}(\mathrm{AD})_t$$

TA-Lib seeds **both** averages with the first A/D reading rather than with an
SMA, which is `mode="recursive"` here; the warm-up is `slow_period - 1`.

On Balance Volume adds volume on up bars and subtracts it on down bars,
disregarding the size of the move:

$$\mathrm{OBV}_t = \mathrm{OBV}_{t-1} + \mathrm{sgn}(C_t - C_{t-1})\,V_t$$

The first bar seeds the total with its own volume, as TA-Lib does, and an
unchanged close contributes nothing.

## NATR — Normalized Average True Range

$$\mathrm{NATR}_t = 100 \cdot \frac{\mathrm{ATR}_t}{C_t}$$

Expressing volatility relative to price makes readings comparable across
instruments and across long stretches of history, which the raw ATR is not. A
zero close reports `0.0`. The warm-up matches the ATR''s $n$.

## HT_* — The Hilbert Transform Cycle Indicators

TA-Lib derives `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`,
`ht_trendmode`, `ht_trendline`, and `mama` from a single recursion, and so does
`polars_ta` — `polars_ta/_hilbert.py` runs it once and each public indicator
selects the series it needs.

The chain is: a four-period weighted smoother of price; a six-tap Hilbert
transform (coefficients $0.0962$ and $0.5769$, scaled by $0.075\,\text{period}
+ 0.54$) producing the in-phase and quadrature components; a complex
multiplication against the previous bar''s components to recover the dominant
cycle period; and a clamp of that period to $[6, 50]$ bars with a maximum
bar-to-bar change of $\pm 50\%$ before smoothing.

| Indicator | Reports |
| --------- | ------- |
| `ht_dcperiod` | the smoothed dominant cycle period, in bars |
| `ht_phasor` | the `in_phase` and `quadrature` components |
| `ht_dcphase` | the dominant cycle phase, in degrees, wrapped to end at `315` |
| `ht_sine` | `sine` and `lead_sine`, the phase and a 45° lead |
| `ht_trendline` | price averaged over exactly one dominant cycle |
| `ht_trendmode` | `1` while trending, `0` while cycling |

`ht_trendline` adapts its averaging length to the measured cycle, which is what
lets it cancel the cycle and leave the trend as the rhythm stretches and
compresses. `ht_trendmode` combines three tests — a sine/lead-sine crossing, how
long the current mode has held, and how fast the phase is turning — and then
overrides them to `1` whenever price has separated from the trendline by more
than 1.5%.

**Warm-up.** 32 leading nulls for `ht_dcperiod`, `ht_phasor`, and `mama`; 63 for
the rest, which also need the dominant cycle phase. The scan primes the price
smoother over the first 12 bars either way.

**Null handling.** The recursion has no way to skip a bar, so a null input ends
the scan: every row from there on is null.

**Implementation.** A Python scan inside `map_batches`, like `supertrend` and
`sar`. It is substantially slower than the expression-based indicators.
