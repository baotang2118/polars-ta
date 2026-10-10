# Indicator Reference: Other Momentum Indicators

Rate-of-change, price oscillators, and other momentum measures.

[Full index and shared conventions](indicators.md) · [All detailed reference pages](indicators.md#detailed-reference-pages)

## Momentum

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

$$
K_t = 100 \cdot
\left(M_t - \min_{i<c} M_{t-i}\right)
/ \left(\max_{i<c} M_{t-i} - \min_{i<c} M_{t-i}\right),
\qquad D_t = \mathrm{EMA}_{k}(K)_t
$$

$$
\mathrm{STC}_t = \mathrm{EMA}_{d}\left(
	100 \cdot
	\frac{
		D_t - \min_{i<c}(D_{t-i})
	}{
		\max_{i<c}(D_{t-i}) - \min_{i<c}(D_{t-i})
	}
\right)_t
$$

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
