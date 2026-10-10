# Indicator Reference: Price Overlays
 bands, channels, stops, rolling midpoints, and price transforms.

[Full index and shared conventions](indicators.md) · [All detailed reference pages](indicators.md#detailed-reference-pages)

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
