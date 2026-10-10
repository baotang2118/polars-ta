# Indicator Reference: Moving Averages
 averages, adaptive averages, and runtime-selected averages.

[Full index and shared conventions](indicators.md) · [All detailed reference pages](indicators.md#detailed-reference-pages)

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

$$
\mathrm{TEMA}_t = 3\,\mathrm{EMA}^{(1)}_t - 3\,\mathrm{EMA}^{(2)}_t + \mathrm{EMA}^{(3)}_t
$$

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

$$
\alpha_t = \max\!\left(\frac{\texttt{fast\_limit}}{\Delta\phi_t},\,
	exttt{slow\_limit}\right), \qquad \Delta\phi_t = \max(\phi_{t-1} - \phi_t,\, 1)
$$

$$
\mathrm{MAMA}_t = \alpha_t P_t + (1-\alpha_t)\mathrm{MAMA}_{t-1}, \qquad
\mathrm{FAMA}_t = \tfrac{\alpha_t}{2}\mathrm{MAMA}_t
+ \left(1 - \tfrac{\alpha_t}{2}\right)\mathrm{FAMA}_{t-1}
$$

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
