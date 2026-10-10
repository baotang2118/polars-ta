# Indicator Reference: Statistics

Statistical measures summarise a window of values rather than reading price
action: they fit a line, measure a spread, or compare two series. They live in
`polars_ta.statistic` and are re-exported from the package root. See
[indicators.md](indicators.md) for the conventions every function here shares.

## Dispersion

`polars_ta.statistic.dispersion`.

### `var(column, window=5)`

The population variance of the window, the window divided by in full rather
than by `window - 1`:

```
mean    = rolling_mean(x, window)
squares = rolling_mean(x * x, window)
var     = squares - mean^2
```

Computing it as the mean of squares minus the square of the mean is TA-Lib's
form and needs one pass, but it can land a hair below zero when every value in
the window is nearly identical; the result is clipped at zero.

Leading nulls: `window - 1`.

TA-Lib's `VAR` accepts an `optInNbDev` argument and ignores it. Rather than
carry a parameter that does nothing, `var` omits it; use `stddev` when a
multiple is wanted.

### `stddev(column, window=5, nbdev=1.0)`

`sqrt(var) * nbdev`. `nbdev` is the band width Bollinger Bands and similar
envelopes ask for: `2.0` returns two standard deviations.

Leading nulls: `window - 1`.

### `zscore(column, window=20, ddof=0)`

How far the current value sits from its own rolling mean, measured in standard
deviations:

```
zscore = (x - rolling_mean(x, window)) / rolling_stddev(x, window)
```

Both the mean and the standard deviation are taken over the same window, so the
score says where the latest bar falls within its recent history: `0` is the
average, `2` is two standard deviations above it.

`ddof` chooses the divisor behind the standard deviation. `0`, the default,
divides by `window` for the population figure, matching `var` and `stddev`; `1`
divides by `window - 1` for the sample figure, which scales every score by
`sqrt((window - 1) / window)`.

A window whose values never move has no spread and no meaningful score, so the
row is null rather than `0.0`. This differs from `correl` and `beta`, which
report `0.0` in the same situation: there a flat input still says something
about the relationship between two series, whereas here the division simply has
no answer. A `window` of `1` is flat by definition and yields nulls throughout.

Leading nulls: `window - 1`.

TA-Lib has no `ZSCORE`.

## Correlation

`polars_ta.statistic.correlation`.

### `correl(first, second, window=30)`

Pearson's correlation coefficient over the window:

```
cov    = Sxy - Sx * Sy / n
spread = (Sxx - Sx^2 / n) * (Syy - Sy^2 / n)
r      = cov / sqrt(spread)
```

where each `S` is a rolling sum over `n = window` rows. It measures the
strength of a straight-line relationship only, not its slope: `1` and `-1` mean
the points fall exactly on a rising or falling line, however steep. A window in
which either series never moves collapses `spread` to zero and reports `0.0`.

Leading nulls: `window - 1`.

### `beta(asset, market, window=5)`

Both series are reduced to simple returns, `(x[t] - x[t-1]) / x[t-1]`, with a
zero previous value yielding a zero return rather than a division. The slope of
a line fitted through the resulting pairs is reported, with the *reference*
series on the x axis:

```
beta = (n * Sxy - Sx * Sy) / (n * Sxx - Sx^2)
```

where `x` holds the reference returns and `y` the asset's. `1` means the asset
matched the reference move for move; above `1` it amplified it. A reference
that never moves reports `0.0`.

Leading nulls: `window`, since one bar is consumed turning prices into returns.

TA-Lib's `BETA` puts its *first* argument on the x axis while documenting that
argument as the stock, which inverts the conventional definition. `beta` keeps
the conventional one: the denominator is always the spread of `market`.

## Linear regression

`polars_ta.statistic.regression`. All five fit the same least-squares line
through the window, with the oldest bar at `x = 0` and the newest at
`x = window - 1`, and differ only in what they read off it. `window` must be at
least `2`; one point does not determine a line.

With `n = window`, `Sy` the rolling sum of the values and `Sjy` the rolling sum
weighted by position within the window:

```
slope     = 12 * (Sjy - Sy * (n - 1) / 2) / (n * (n - 1) * (n + 1))
intercept = Sy / n - slope * (n - 1) / 2
```

| Function | Reads | Value |
| -------- | ----- | ----- |
| `linearreg_slope(column, window=14)` | the gradient | `slope` |
| `linearreg_intercept(column, window=14)` | the line at the window's first bar | `intercept` |
| `linearreg(column, window=14)` | the line at the current bar | `intercept + slope * (window - 1)` |
| `tsf(column, window=14)` | the line one bar past the window | `intercept + slope * window` |
| `linearreg_angle(column, window=14)` | the gradient in degrees | `atan(slope) * 180 / pi` |

`tsf` therefore leads `linearreg` by exactly one slope step.

`linearreg_angle` depends on the input's scale, since one bar on the horizontal
axis is compared against one price unit on the vertical: the same series quoted
in cents tilts far more steeply than quoted in dollars.

Leading nulls: `window - 1` for all five.
