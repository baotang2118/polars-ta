# Indicator Reference: Conventions and Overlays

Formulas and conventions for every indicator implemented in `polars_ta`.
Indicators are grouped by how they are charted: *overlay* indicators share the
price axis, *momentum* oscillators occupy a separate pane, *volume* indicators
weight movement by how much traded, *volatility* indicators measure the size of
movement, *cycle* indicators measure its rhythm, and *returns* restate price on
a percentage scale. Each group is a subpackage (`polars_ta.overlay`,
`polars_ta.momentum`, `polars_ta.volume`, `polars_ta.volatility`,
`polars_ta.cycle`, `polars_ta.returns`), and every public indicator is also
re-exported from the package root.

The reference is organized as an index and shared conventions, followed by seven focused formula pages. The detail pages are grouped to keep related indicators together and generally cover 10–15 functions.

## Detailed Reference Pages

| Page | Covers |
| ---- | ------ |
| [Moving averages](indicators-overlays-averages.md) | Moving, adaptive, and dispatched averages (11) |
| [Price overlays](indicators-overlays-price.md) | Bands, channels, stops, and price transforms (13) |
| [Momentum oscillators](indicators-momentum-oscillators.md) | RSI-style, range, and bar oscillators (10) |
| [Trend and direction](indicators-momentum-trend.md) | MACD, directional movement, Aroon, and Vortex (13) |
| [Other momentum](indicators-momentum-other.md) | Rate-of-change, price oscillators, and related measures (15) |
| [Volume and returns](indicators-volume-returns.md) | Volume indicators and return measures (12) |
| [Volatility and cycle](indicators-volatility-cycle.md) | Volatility and Hilbert cycle indicators (10) |

The conventions below apply to every page.

Every indicator listed in the table below is implemented.

| Indicator | Module | Inputs | Output |
| --------- | ------ | ------ | ------ |
| `sma`, `wma`, `ema`, `dema`, `tema`, `trima`, `t3` | `overlay.ma` | one column | one `Float64` column |
| `kama` | `overlay.adaptive` | one column | one `Float64` column |
| `mama` | `overlay.adaptive` | one column | struct of two `Float64` fields |
| `ma`, `mavp` | `overlay.dispatch` | one or two columns | one `Float64` column |
| `midpoint` | `overlay.midpoint` | one column | one `Float64` column |
| `midprice` | `overlay.midpoint` | high, low | one `Float64` column |
| `avgprice` | `overlay.transform` | open, high, low, close | one `Float64` column |
| `medprice` | `overlay.transform` | high, low | one `Float64` column |
| `typprice`, `wclprice` | `overlay.transform` | high, low, close | one `Float64` column |
| `bbands` | `overlay.bands` | one column | struct of three `Float64` fields |
| `donchian` | `overlay.channels` | high, low | struct of three `Float64` fields |
| `keltner` | `overlay.channels` | high, low, close | struct of three `Float64` fields |
| `sar`, `sarext` | `overlay.sar` | high, low | one `Float64` column |
| `supertrend` | `overlay.supertrend` | high, low, close | struct of `Float64` and `Int8` |
| `ichimoku` | `overlay.ichimoku` | high, low, close | struct of five `Float64` fields |
| `rsi`, `cmo` | `momentum.rsi` | one column | one `Float64` column |
| `mfi` | `momentum.mfi` | high, low, close, volume | one `Float64` column |
| `stoch`, `stochf` | `momentum.stoch` | high, low, close | struct of two `Float64` fields |
| `stochrsi` | `momentum.stoch` | one column | struct of two `Float64` fields |
| `willr` | `momentum.stoch` | high, low, close | one `Float64` column |
| `cci` | `momentum.cci` | high, low, close | one `Float64` column |
| `macd`, `macdext`, `macdfix` | `momentum.macd` | one column | struct of three `Float64` fields |
| `adx` | `momentum.adx` | high, low, close | struct of three `Float64` fields |
| `adxr`, `dx`, `plus_di`, `minus_di` | `momentum.adx` | high, low, close | one `Float64` column |
| `plus_dm`, `minus_dm` | `momentum.adx` | high, low | one `Float64` column |
| `aroon` | `momentum.aroon` | high, low | struct of two `Float64` fields |
| `aroonosc` | `momentum.aroon` | high, low | one `Float64` column |
| `bop` | `momentum.bop` | open, high, low, close | one `Float64` column |
| `mom`, `roc`, `rocp`, `rocr`, `rocr100` | `momentum.roc` | one column | one `Float64` column |
| `apo`, `ppo`, `pvo` | `momentum.price_oscillator` | one column | one `Float64` column |
| `trix` | `momentum.trix` | one column | one `Float64` column |
| `ultosc` | `momentum.ultosc` | high, low, close | one `Float64` column |
| `dpo` | `momentum.dpo` | one column | one `Float64` column |
| `kst` | `momentum.kst` | one column | struct of two `Float64` fields |
| `stc` | `momentum.stc` | one column | one `Float64` column |
| `tsi` | `momentum.tsi` | one column | one `Float64` column |
| `ao` | `momentum.awesome` | high, low | one `Float64` column |
| `mass` | `momentum.mass` | high, low | one `Float64` column |
| `vortex` | `momentum.vortex` | high, low, close | struct of two `Float64` fields |
| `ad`, `adosc`, `cmf` | `volume.flow` | high, low, close, volume | one `Float64` column |
| `obv` | `volume.flow` | close, volume | one `Float64` column |
| `fi`, `vpt`, `nvi` | `volume.pressure` | close, volume | one `Float64` column |
| `eom` | `volume.pressure` | high, low, volume | one `Float64` column |
| `vwap` | `volume.vwap` | high, low, close, volume | one `Float64` column |
| `true_range`, `atr`, `natr` | `volatility.atr` | high, low, close | one `Float64` column |
| `ulcer` | `volatility.ulcer` | one column | one `Float64` column |
| `ht_dcperiod`, `ht_dcphase`, `ht_trendline` | `cycle.hilbert` | one column | one `Float64` column |
| `ht_phasor`, `ht_sine` | `cycle.hilbert` | one column | struct of two `Float64` fields |
| `ht_trendmode` | `cycle.hilbert` | one column | one `Int8` column |
| `daily_return`, `daily_log_return`, `cumulative_return` | `returns.performance` | one column | one `Float64` column |

Detailed formulas are in the topic pages listed above.

## Shared Conventions

- **Input forms.** Each indicator accepts a column name (`str`) or a `pl.Expr`,
  and always returns a `pl.Expr`, so it composes inside `select`/`with_columns`
  and runs lazily. Series and frame inputs are not supported: passing a
  `pl.Series`, `pl.DataFrame`, or `pl.LazyFrame` raises `TypeError`. Evaluate
  the expression on a frame when an eager result is needed, for example
  `values.to_frame("close").select(sma("close", 3)).to_series()`.
- **Warm-up.** Every indicator emits exactly the TA-Lib lookback as leading
  nulls:

  | Indicator | Leading nulls |
  | --------- | ------------- |
  | `bop`, `ad`, `obv`, `vpt`, `nvi`, `cumulative_return` | none |
  | `avgprice`, `medprice`, `typprice`, `wclprice` | none |
  | `sma`, `wma`, `ema`, `bbands`, `cci`, `donchian`, `midpoint`, `midprice`, `trima`, `willr` | `window - 1` |
  | `plus_dm`, `minus_dm` | `window - 1` |
  | `cmf`, `vwap` | `window - 1` |
  | `ao` | `slow_period - 1` |
  | `dema` | `2 * (window - 1)` |
  | `tema` | `3 * (window - 1)` |
  | `t3` | `6 * (window - 1)` |
  | `trix` | `3 * (window - 1) + 1` |
  | `true_range` | `1` |
  | `sar`, `sarext` | `1` |
  | `daily_return`, `daily_log_return` | `1` |
  | `rsi`, `cmo`, `mfi`, `atr`, `natr`, `supertrend`, `kama`, `mom`, `roc`, `rocp`, `rocr`, `rocr100` | `window` |
  | `aroon`, `aroonosc`, `dx` | `window` |
  | `fi`, `eom`, `vortex` | `window` |
  | `ulcer` | `2 * (window - 1)` |
  | `dpo` | `max(window - 1, window // 2 + 1)` |
  | `tsi` | `slow_period + fast_period - 1` |
  | `mass` | `2 * (fast_period - 1) + slow_period - 1` |
  | `adx` (`plus_di`, `minus_di`) | `window` |
  | `adx` (`adx`) | `2 * window - 1` |
  | `adxr` | `3 * window - 2` |
  | `stoch` | `(fastk_period - 1) + (slowk_period - 1) + (slowd_period - 1)` |
  | `stochf` | `(fastk_period - 1) + (fastd_period - 1)` |
  | `stochrsi` | `window + (fastk_period - 1) + (fastd_period - 1)` |
  | `macd`, `macdfix` | `(slow_period - 1) + (signal_period - 1)` |
  | `macdext`, `apo`, `ppo`, `pvo` | the chosen averages' own lookbacks |
  | `adosc` | `slow_period - 1` |
  | `ultosc` | `max(short, medium, long)` |
  | `mavp` | `max_period - 1` |
  | `ht_dcperiod`, `ht_phasor`, `mama` | `32` |
  | `ht_dcphase`, `ht_sine`, `ht_trendmode`, `ht_trendline` | `63` |
  | `keltner` | per field; see [price overlays](indicators-overlays-price.md) |
  | `ichimoku` | per field; see [price overlays](indicators-overlays-price.md) |
  | `kst` | per field; see [other momentum](indicators-momentum-other.md) |
  | `stc` | the chained averages' own lookbacks |

  There is no `min_periods` parameter, so a simple and an exponential moving
  average of the same window line up row for row.
- **Multi-field outputs follow TA-Lib's emission.** Where TA-Lib produces a
  struct's fields from one function, they start on the same row: this holds back
  `stoch`'s `k` until `d` exists, and `macd`'s `macd` line until its `signal`
  exists. Where the fields correspond to *separate* TA-Lib functions with
  different lookbacks, they keep their own warm-ups rather than discarding good
  data — `adx`'s `plus_di` and `minus_di` start `window - 1` rows before `adx`.
  `donchian` and `ichimoku` likewise let each field reflect only the inputs it
  actually depends on.
- **Nulls.** A null input never silently disappears. See the null handling notes
  under each indicator for the exact rule.
- **Output dtype.** Always `Float64`, including for integer inputs.
- **Insufficient data.** If the input is shorter than the warm-up length, the
  whole result is null.

Throughout, $P_t$ is the input value at row $t$ (zero-indexed) and $n$ is
`window`.
