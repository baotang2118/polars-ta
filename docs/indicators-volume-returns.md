# Indicator Reference: Volume and Returns

Volume-derived indicators and price-return measures.

[Full index and shared conventions](indicators.md) · [All detailed reference pages](indicators.md#detailed-reference-pages)

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
