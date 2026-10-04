# Indicator Reference

Formulas and conventions for every indicator implemented in `polars_ta`.
Indicators are grouped by how they are charted: *overlay* indicators share the
price axis and are drawn on top of the price series. Moving averages live in
`polars_ta.overlay.ma` and are re-exported from `polars_ta.overlay` and the
package root.

## Shared Conventions

- **Input forms.** Each indicator accepts a column name (`str`), a `pl.Expr`, or
  a `pl.Series`. Name and expression inputs return a `pl.Expr`, so they compose
  inside `select`/`with_columns` and run lazily. A series input is evaluated
  eagerly and returns a `pl.Series` carrying the input series name.
- **Warm-up.** The first `window - 1` rows are always null, for every indicator
  and every mode. There is no `min_periods` parameter, so a simple and an
  exponential moving average of the same window line up row for row.
- **Nulls.** A null input never silently disappears. See the null handling notes
  under each indicator for the exact rule.
- **Output dtype.** Always `Float64`, including for integer inputs.
- **Insufficient data.** If the input is shorter than `window`, the whole result
  is null.

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
