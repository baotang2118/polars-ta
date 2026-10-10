# Indicator Reference: Volatility and Cycle

Volatility and cycle indicators.

[Full index and shared conventions](indicators.md) · [All detailed reference pages](indicators.md#detailed-reference-pages)

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
