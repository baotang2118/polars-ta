# Indicator Reference

Formulas and conventions for every indicator implemented in `polars_ta`.
Indicators are grouped by how they are charted: *overlay* indicators share the
price axis and are drawn on top of the price series, *momentum* oscillators
occupy a separate pane, and *volatility* indicators measure the size of
movement rather than its direction. Each group is a subpackage
(`polars_ta.overlay`, `polars_ta.momentum`, `polars_ta.volatility`), and every
public indicator is also re-exported from the package root.

| Indicator | Module | Inputs | Output |
| --------- | ------ | ------ | ------ |
| `sma`, `wma`, `ema`, `dema`, `tema` | `overlay.ma` | one column | one `Float64` column |
| `bbands` | `overlay.bands` | one column | struct of three `Float64` fields |
| `donchian` | `overlay.channels` | high, low | struct of three `Float64` fields |
| `supertrend` | `overlay.supertrend` | high, low, close | struct of `Float64` and `Int8` |
| `ichimoku` | `overlay.ichimoku` | high, low, close | struct of five `Float64` fields |
| `rsi` | `momentum.rsi` | one column | one `Float64` column |
| `mfi` | `momentum.mfi` | high, low, close, volume | one `Float64` column |
| `stoch` | `momentum.stoch` | high, low, close | struct of two `Float64` fields |
| `cci` | `momentum.cci` | high, low, close | one `Float64` column |
| `macd` | `momentum.macd` | one column | struct of three `Float64` fields |
| `adx` | `momentum.adx` | high, low, close | struct of three `Float64` fields |
| `true_range`, `atr` | `volatility.atr` | high, low, close | one `Float64` column |

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
  | `sma`, `wma`, `ema`, `bbands`, `cci`, `donchian` | `window - 1` |
  | `dema` | `2 * (window - 1)` |
  | `tema` | `3 * (window - 1)` |
  | `true_range` | `1` |
  | `rsi`, `mfi`, `atr`, `supertrend` | `window` |
  | `adx` (`plus_di`, `minus_di`) | `window` |
  | `adx` (`adx`) | `2 * window - 1` |
  | `stoch` | `(fastk_period - 1) + (slowk_period - 1) + (slowd_period - 1)` |
  | `macd` | `(slow_period - 1) + (signal_period - 1)` |
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
