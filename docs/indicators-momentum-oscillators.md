# Indicator Reference: Momentum Oscillators

Oscillators built from price changes, ranges, or bar-level price movement.

[Full index and shared conventions](indicators.md) · [All detailed reference pages](indicators.md#detailed-reference-pages)

## Momentum

### RSI — Relative Strength Index

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

### CMO — Chande Momentum Oscillator

The same Wilder-smoothed gains and losses as the RSI, but centred on zero:

$$\mathrm{CMO}_t = 100 \cdot \frac{\bar{G}_t - \bar{L}_t}{\bar{G}_t + \bar{L}_t}$$

Away from a flat window this is exactly $2\,\mathrm{RSI}_t - 100$. A window with
neither a gain nor a loss reports **0.0** — note that the RSI reports `50.0` in
the same situation; both match TA-Lib and should not be harmonized.

### MFI — Money Flow Index

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

The four inputs are positional, in TA-Lib's order. They may be column names or
expressions.

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

### STOCH — Stochastic Oscillator

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

### STOCHF, STOCHRSI, and WILLR

`stochf` is the unsmoothed pair: raw fast %K, and a single SMA pass for fast %D.
`stoch` smooths %K once more before reporting it.

`stochrsi` applies `stochf` to the RSI instead of to price, using the RSI as
high, low, and close alike. The RSI rarely reaches its own extremes, so reading
where it sits inside its recent range is far more sensitive than its level.

`willr` is the mirror of raw fast %K on a $[-100, 0]$ scale:

$$\mathrm{WILLR}_t = -100 \cdot
\frac{\max_{i<n} H_{t-i} - C_t}{\max_{i<n} H_{t-i} - \min_{i<n} L_{t-i}}$$

A flat range reports `0.0`, as in TA-Lib.

### CCI — Commodity Channel Index

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

### BOP — Balance Of Power

$$\mathrm{BOP}_t = \frac{C_t - O_t}{H_t - L_t}$$

A single-bar measure with no lookback at all: `+1` means the bar opened at its
low and closed at its high, `-1` the reverse. A bar with no range reports `0.0`.

### AO — Awesome Oscillator

$$\mathrm{AO}_t = \mathrm{SMA}_{5}(\mathrm{MEDPRICE})_t
- \mathrm{SMA}_{34}(\mathrm{MEDPRICE})_t$$

Building on the bar's midpoint rather than its close keeps the reading out of
the hands of a single print. The warm-up is $n_{\text{slow}} - 1$.
