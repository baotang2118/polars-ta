# Indicator Reference: Trend and Direction

Trend-following and directional momentum indicators.

[Full index and shared conventions](indicators.md) · [All detailed reference pages](indicators.md#detailed-reference-pages)

## Momentum

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

### VORTEX — Vortex Indicator

$$\mathrm{VI}^{+}_t = \frac{\sum_{i<n}\left|H_{t-i} - L_{t-i-1}\right|}
{\sum_{i<n}\mathrm{TR}_{t-i}}, \qquad
\mathrm{VI}^{-}_t = \frac{\sum_{i<n}\left|L_{t-i} - H_{t-i-1}\right|}
{\sum_{i<n}\mathrm{TR}_{t-i}}$$

`plus` measures the ground covered from the previous low up to today's high and
`minus` the reverse, each against true range, so the lines cross when one
direction starts covering more ground than the other. Both fields share the
warm-up of $n$, and a window with no range reports `0.0`.
