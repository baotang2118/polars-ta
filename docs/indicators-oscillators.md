# Indicator Reference: Momentum, Volume, Volatility, Cycle, Returns

Formulas and conventions for the indicators that occupy their own pane rather
than the price axis. Overlays live in [indicators.md](indicators.md), which also
holds the full index and the **shared conventions** — input forms, warm-up
table, null rules, and output dtypes — that apply to everything here.

On this page:

- [Momentum](#momentum) — `rsi`, `cmo`, `mfi`, `stoch`, `stochf`, `stochrsi`,
  `willr`, `cci`, `macd`, `macdext`, `macdfix`, `adx`, `adxr`, `dx`,
  `plus_di`, `minus_di`, `plus_dm`, `minus_dm`, `aroon`, `aroonosc`, `bop`,
  `mom`, `roc`, `rocp`, `rocr`, `rocr100`, `apo`, `ppo`, `pvo`, `trix`,
  `ultosc`, `vortex`, `mass`, `dpo`, `kst`, `stc`, `tsi`, `ao`
- [Volume](#volume) — `ad`, `adosc`, `obv`, `cmf`, `fi`, `eom`, `vpt`, `nvi`,
  `vwap`
- [Volatility](#volatility) — `true_range`, `atr`, `natr`, `ulcer`
- [Cycle](#cycle) — `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`,
  `ht_trendmode`, `ht_trendline`
- [Returns](#returns) — `daily_return`, `daily_log_return`, `cumulative_return`

Throughout, $P_t$ is the input value at row $t$ (zero-indexed) and $n$ is
`window`.

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

### MACD — Moving Average Convergence/Divergence

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
`"adjust"` give the pandas conventions described under
[EMA](indicators.md#ema--exponential-moving-average).

**Null handling and implementation.** Three calls to the public `ema`. The
TA-Lib seeding rule — seed at the first complete, null-free window — produces
the correct combined warm-up with no explicit offset arithmetic.

### MACDEXT and MACDFIX

`macdext` is the MACD with a separately selectable average for each of its three
passes, through `fast_ma_type`, `slow_ma_type`, and `signal_ma_type` (all
defaulting to `"sma"`, as in TA-Lib). `macdfix` fixes the periods at the classic
12 and 26, leaving only the signal period tunable. Both return the same
`macd`/`signal`/`histogram` struct as `macd`, with all three fields starting on
the same row.

### ADX / DI — Average Directional Index

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

### PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI, DX, ADXR

The directional movement family shares one engine with `adx`. Raw directional
movement counts a bar toward only the larger outward move:

$$+\mathrm{DM}_t = \begin{cases} H_t - H_{t-1} & \text{if it exceeds } L_{t-1}-L_t \text{ and } 0 \\ 0 & \text{otherwise}\end{cases}$$

Both are accumulated with **Wilder's running sum**, whose seed deliberately
holds $n-1$ terms rather than $n$:

$$S_{n-1} = \sum_{i=1}^{n-1} x_i, \qquad S_t = S_{t-1} - \frac{S_{t-1}}{n} + x_t$$

That is what gives `plus_dm` its $n-1$ lookback while `plus_di` has $n$: TA-Lib
performs one more smoothing step before emitting the first indicator.

$$\pm\mathrm{DI}_t = 100 \cdot \frac{S^{\pm\mathrm{DM}}_t}{S^{\mathrm{TR}}_t},
\qquad
\mathrm{DX}_t = 100 \cdot \frac{\left|+\mathrm{DI}_t - -\mathrm{DI}_t\right|}
{+\mathrm{DI}_t + -\mathrm{DI}_t}$$

`adx` is Wilder's average of `dx`, and

$$\mathrm{ADXR}_t = \frac{\mathrm{ADX}_t + \mathrm{ADX}_{t-(n-1)}}{2}$$

with a lookback of $3n - 2$.

**Degenerate windows.** A zero range sum makes the indicators report `0.0`.
TA-Lib instead holds the previous ADX unchanged in that case; this is the one
deliberate divergence in the family.

**Implementation.** `_wilder_sum` reproduces the $n-1$ seed by rewriting the
input so that the seed row carries `rolling_sum(n - 1) / n` and then running
`ewm_mean(alpha=1/n)`, multiplying the result back by $n$. `window == 1`
short-circuits to the raw movement, matching TA-Lib's unsmoothed special case.

### AROON and AROONOSC

$$\mathrm{up}_t = \frac{100}{n}\left(n - \text{bars since the window high}\right),
\qquad \mathrm{down}_t = \frac{100}{n}\left(n - \text{bars since the window low}\right)$$

The window spans the last $n$ bars **plus** today, so $n+1$ rows. `up` reaches
`100` on the bar that sets a new high and decays by $100/n$ per bar since. A
reading says nothing about the size of a move, only its freshness.
`aroonosc` is `up - down`.

Repeated extremes count from the **most recent** occurrence, matching TA-Lib's
`<=`/`>=` tie handling. The implementation finds the distance with a chain of
$n+1$ shifted equality tests against the rolling extreme, which is exact because
the rolling extreme is literally one of the values in the window.

### BOP — Balance Of Power

$$\mathrm{BOP}_t = \frac{C_t - O_t}{H_t - L_t}$$

A single-bar measure with no lookback at all: `+1` means the bar opened at its
low and closed at its high, `-1` the reverse. A bar with no range reports `0.0`.

### MOM, ROC, ROCP, ROCR, ROCR100

All five compare the current value against the one $n$ bars back:

| Function | Formula |
| -------- | ------- |
| `mom` | $P_t - P_{t-n}$ |
| `roc` | $\left(P_t / P_{t-n} - 1\right) \cdot 100$ |
| `rocp` | $\left(P_t - P_{t-n}\right) / P_{t-n}$ |
| `rocr` | $P_t / P_{t-n}$ |
| `rocr100` | $P_t / P_{t-n} \cdot 100$ |

The four ratio forms report **0.0** when the reference value is zero, rather
than dividing — TA-Lib's convention. `mom` has no such case.

### APO and PPO — Price Oscillators

$$\mathrm{APO}_t = \mathrm{MA}^{\text{fast}}_t - \mathrm{MA}^{\text{slow}}_t,
\qquad \mathrm{PPO}_t = 100 \cdot \frac{\mathrm{APO}_t}{\mathrm{MA}^{\text{slow}}_t}$$

The same construction as the MACD line, but with a selectable `ma_type` and no
signal line. The periods are ordered before use, so swapping them changes
nothing. Expressing the gap as a percentage makes PPO comparable across
instruments, which a raw price difference is not. A zero slow average makes PPO
report `0.0`.

### PVO — Percentage Volume Oscillator

`ppo` read on volume instead of price, so it says whether participation is
picking up or draining away independently of direction:

$$\mathrm{PVO}_t = 100 \cdot
\frac{\mathrm{MA}_{\text{fast}}(V)_t - \mathrm{MA}_{\text{slow}}(V)_t}
{\mathrm{MA}_{\text{slow}}(V)_t}$$

It defaults to `ma_type="ema"`, unlike `ppo`, whose `"sma"` default comes from
TA-Lib. A zero slow average reports `0.0`.

### TRIX

The one-period rate of change of a triple exponential average:

$$\mathrm{TRIX}_t = 100\left(\frac{e^{(3)}_t}{e^{(3)}_{t-1}} - 1\right)$$

Three smoothing passes strip out cycles shorter than `window`, so the remaining
slope isolates the dominant trend and crosses zero when it turns.

**Warm-up.** $3(n-1) + 1$ — the three EMA passes plus the one-bar difference.

### ULTOSC — Ultimate Oscillator

Buying pressure is measured against the lower of today's low and yesterday's
close, so a gap counts as part of the bar's range:

$$\mathrm{BP}_t = C_t - \min(L_t, C_{t-1}), \qquad
\mathrm{TR}_t = \max(H_t, C_{t-1}) - \min(L_t, C_{t-1})$$

$$\mathrm{ULTOSC}_t = \frac{100}{7}\left(
4\frac{\sum_{i<n_1}\mathrm{BP}_{t-i}}{\sum_{i<n_1}\mathrm{TR}_{t-i}} +
2\frac{\sum_{i<n_2}\mathrm{BP}_{t-i}}{\sum_{i<n_2}\mathrm{TR}_{t-i}} +
\frac{\sum_{i<n_3}\mathrm{BP}_{t-i}}{\sum_{i<n_3}\mathrm{TR}_{t-i}}\right)$$

Williams combined three lookbacks precisely to avoid the false divergences a
single-period oscillator produces. A term whose range summed to zero contributes
nothing rather than dividing.

### VORTEX — Vortex Indicator

$$\mathrm{VI}^{+}_t = \frac{\sum_{i<n}\left|H_{t-i} - L_{t-i-1}\right|}
{\sum_{i<n}\mathrm{TR}_{t-i}}, \qquad
\mathrm{VI}^{-}_t = \frac{\sum_{i<n}\left|L_{t-i} - H_{t-i-1}\right|}
{\sum_{i<n}\mathrm{TR}_{t-i}}$$

`plus` measures the ground covered from the previous low up to today's high and
`minus` the reverse, each against true range, so the lines cross when one
direction starts covering more ground than the other. Both fields share the
warm-up of $n$, and a window with no range reports `0.0`.

### MASS — Mass Index

$$\mathrm{MI}_t = \sum_{i<n_{\text{slow}}}
\frac{\mathrm{EMA}_{n_{\text{fast}}}(H - L)_{t-i}}
{\mathrm{EMA}_{n_{\text{fast}}}\!\left(\mathrm{EMA}_{n_{\text{fast}}}(H - L)\right)_{t-i}}$$

The ratio rises above one when bars start spanning more than they recently did,
so the sum looks for a range bulge — often a reversal warning — without caring
which way price is going. Both averages use `mode="recursive"`, as the common
reference implementations do, which makes the warm-up
$2(n_{\text{fast}} - 1) + n_{\text{slow}} - 1$. Defaults are 9 and 25.

### DPO — Detrended Price Oscillator

$$\mathrm{DPO}_t = P_{t - (\lfloor n/2 \rfloor + 1)} - \mathrm{SMA}_n(P)_t$$

Shifting the comparison back half a window *centres* the average rather than
lagging behind it, which removes the trend and leaves the shorter cycles it was
hiding. The centring reads a past bar, so the line is a study of past cycle
length, not a real-time signal.

### KST — Know Sure Thing

$$\mathrm{KST}_t = 100 \sum_{k=1}^{4} k \cdot
\mathrm{SMA}_{m_k}\!\left(\frac{P_t - P_{t-r_k}}{P_{t-r_k}}\right)_t,
\qquad \mathrm{signal}_t = \mathrm{SMA}_{n_{\text{sig}}}(\mathrm{KST})_t$$

One rate of change only sees one cycle length. Stacking four, with the slowest
weighted four times the fastest, gives a reading that turns on short-term
momentum but stays anchored to the long term. Defaults are $r = (10, 15, 20,
30)$, $m = (10, 10, 10, 15)$, and a 9-period signal. The two fields keep
**separate** warm-ups, following `donchian` rather than `macd`, because the
signal is derived from the line and aligning them would discard good data. A
zero reference price contributes `0.0`.

### STC — Schaff Trend Cycle

The MACD line put through two stochastic passes:

$$K_t = 100 \cdot \frac{M_t - \min_{i<c} M_{t-i}}
{\max_{i<c} M_{t-i} - \min_{i<c} M_{t-i}},
\qquad D_t = \mathrm{EMA}_{k}(K)_t$$

$$\mathrm{STC}_t = \mathrm{EMA}_{d}\!\left(
100 \cdot \frac{D_t - \min_{i<c} D_{t-i}}{\max_{i<c} D_{t-i} - \min_{i<c} D_{t-i}}
\right)_t$$

where $M$ is $\mathrm{EMA}_{\text{fast}}(P) - \mathrm{EMA}_{\text{slow}}(P)$.
The MACD line is unbounded and slow to turn; measuring where it sits inside its
own recent range, twice over, bounds it to $[0, 100]$ and sharpens the turns
enough to read as overbought and oversold. Defaults are 23, 50, a cycle of 10,
and two 3-period smoothings. Every average uses `mode="recursive"`, and a flat
range reports `0.0`.

### TSI — True Strength Index

$$\mathrm{TSI}_t = 100 \cdot
\frac{\mathrm{EMA}_{f}\!\left(\mathrm{EMA}_{s}(\Delta P)\right)_t}
{\mathrm{EMA}_{f}\!\left(\mathrm{EMA}_{s}(\left|\Delta P\right|)\right)_t}$$

Smoothing the raw change twice strips the noise that makes momentum unreadable,
and dividing by the same smoothing of its absolute value rescales the result to
$[-100, 100]$ regardless of the instrument's volatility. Defaults are a slow 25
and a fast 13, giving a warm-up of $s + f - 1$. A window with no movement reports
`0.0`.

### AO — Awesome Oscillator

$$\mathrm{AO}_t = \mathrm{SMA}_{5}(\mathrm{MEDPRICE})_t
- \mathrm{SMA}_{34}(\mathrm{MEDPRICE})_t$$

Building on the bar's midpoint rather than its close keeps the reading out of
the hands of a single print. The warm-up is $n_{\text{slow}} - 1$.

## Volume

### AD, ADOSC, and OBV

Chaikin's Accumulation/Distribution Line signs each bar's volume by where the
close finished inside the bar's range, then accumulates:

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

### CMF — Chaikin Money Flow

The same money flow multiplier as `ad`, summed over a window and divided by the
window's volume:

$$\mathrm{CMF}_t = \frac{\sum_{i<n}
\frac{(C_{t-i} - L_{t-i}) - (H_{t-i} - C_{t-i})}{H_{t-i} - L_{t-i}} V_{t-i}}
{\sum_{i<n} V_{t-i}}$$

Normalising by volume bounds the reading to $[-1, 1]$, so it says what
*fraction* of recent trade was accumulation rather than how much of it there
was. A window that traded nothing reports `0.0`; a bar with no range contributes
nothing. The warm-up is $n - 1$; the default $n$ is 20.

### FI, EOM, VPT, NVI — Volume Pressure

**Force Index** multiplies the close-to-close move by the volume behind it, then
smooths the product, which is far too noisy raw:

$$\mathrm{FI}_t = \mathrm{EMA}_n\left((C_t - C_{t-1}) V_t\right)$$

**Ease of Movement** asks how far the bar's midpoint travelled per unit of
volume:

$$\mathrm{EMV}_t = \frac{\left(\frac{H_t + L_t}{2} -
\frac{H_{t-1} + L_{t-1}}{2}\right)(H_t - L_t)}{V_t} \times 10^8,
\qquad \mathrm{EOM}_t = \mathrm{SMA}_n(\mathrm{EMV})_t$$

The $10^8$ factor is the usual box-ratio convention, which keeps the reading
legible against raw share volume. `window=1` leaves it unsmoothed, and a bar
with no volume reports `0.0`.

**Volume-Price Trend** accumulates volume weighted by each bar's *return*, where
`obv` would add the whole of it:

$$\mathrm{VPT}_t = \mathrm{VPT}_{t-1} + \frac{C_t - C_{t-1}}{C_{t-1}} V_t$$

The first bar has no return and so seeds the total at `0.0`.

**Negative Volume Index** compounds only on bars whose volume fell, on the
premise that informed money moves quietly:

$$\mathrm{NVI}_t = \begin{cases}
\mathrm{NVI}_{t-1}\left(1 + \frac{C_t - C_{t-1}}{C_{t-1}}\right)
& \text{if } V_t < V_{t-1} \\
\mathrm{NVI}_{t-1} & \text{otherwise}\end{cases}$$

It starts at `start_value` (conventionally 1000) and has no warm-up. A bar whose
return cannot be computed carries the level forward rather than nulling the rest
of the index, which a cumulative product would otherwise do.

### VWAP — Volume Weighted Average Price

$$\mathrm{VWAP}_t = \frac{\sum_{i<n} \mathrm{TYPPRICE}_{t-i} V_{t-i}}
{\sum_{i<n} V_{t-i}}$$

Weighting each bar's typical price by its volume puts the line where most of the
trade actually happened. This is the **rolling** form, not the session-anchored
one; set `window` to the number of bars in your session to approximate the
latter. A window that traded no volume is **null**, not `0.0` — unlike the
oscillators, a zero here would be a nonsense price. The warm-up is $n - 1$.

## Volatility

### TRANGE / ATR — True Range and Average True Range

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

### NATR — Normalized Average True Range

$$\mathrm{NATR}_t = 100 \cdot \frac{\mathrm{ATR}_t}{C_t}$$

Expressing volatility relative to price makes readings comparable across
instruments and across long stretches of history, which the raw ATR is not. A
zero close reports `0.0`. The warm-up matches the ATR's $n$.

### ULCER — Ulcer Index

$$R_t = 100 \cdot \frac{C_t - \max_{i<n} C_{t-i}}{\max_{i<n} C_{t-i}},
\qquad \mathrm{UI}_t = \sqrt{\frac{1}{n}\sum_{i<n} R_{t-i}^2}$$

Standard deviation punishes upside and downside alike. This only accumulates
while price sits below its recent high, so it measures how deep and how long the
pain was. Because the drawdown series needs a full window before the averaging
starts, the warm-up is $2(n-1)$ rather than $n-1$.

## Cycle

### HT_* — The Hilbert Transform Cycle Indicators

TA-Lib derives `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`,
`ht_trendmode`, `ht_trendline`, and `mama` from a single recursion, and so does
`polars_ta` — `polars_ta/_hilbert.py` runs it once and each public indicator
selects the series it needs.

The chain is: a four-period weighted smoother of price; a six-tap Hilbert
transform (coefficients $0.0962$ and $0.5769$, scaled by $0.075\,\text{period}
+ 0.54$) producing the in-phase and quadrature components; a complex
multiplication against the previous bar's components to recover the dominant
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

## Returns

### DAILY_RETURN, DAILY_LOG_RETURN, CUMULATIVE_RETURN

$$\mathrm{DR}_t = 100\left(\frac{P_t}{P_{t-1}} - 1\right), \qquad
\mathrm{DLR}_t = 100 \ln\frac{P_t}{P_{t-1}}, \qquad
\mathrm{CR}_t = 100\left(\frac{P_t}{P_{0}} - 1\right)$$

`daily_return` is `roc` with a window of 1, named for discoverability. Log
returns add across time, which simple returns do not, so a sum over a period is
that period's return rather than an approximation of it; they assume positive
prices, since the logarithm of a sign change is `NaN`. `cumulative_return`
measures from the first **known** value, so leading nulls do not set the base,
and has no warm-up. A zero reference price reports `0.0`.
