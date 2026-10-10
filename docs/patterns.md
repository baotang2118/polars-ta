# Candlestick Pattern Reference

Candlestick recognisers read the shape of a bar and the bars before it and
report whether a named formation has just completed. They live in
`polars_ta.pattern` and are re-exported from the package root. Every one takes
`(open_, high, low, close)` as column names or expressions and returns a
`pl.Expr`, like any other indicator; see [indicators.md](indicators.md) for the
conventions they share with the rest of the library.

## Output

A recogniser returns one `Int32` column:

| Value | Meaning |
| ----- | ------- |
| `0` | The pattern did not complete on this bar. |
| `100` | A bullish reading completed on this bar. |
| `-100` | A bearish reading completed on this bar. |
| `±200` | `cdlhikkake` and `cdlhikkakemod` only: a later bar confirmed an earlier reading, carrying its sign. |
| `null` | Warm-up, or a window containing a null input. |

The sign is the formation's own orientation, not a forecast. Many of these
patterns only carry their traditional meaning when they appear in a trend, and
none of the functions look at the trend: a Hammer and a Hanging Man are the
same shape, told apart by what preceded them, and the library reports both
whenever the shape occurs.

## Candle settings

"Long", "short", "near" and the rest are relative terms, measured against the
bars that came before. Each named setting pairs the quantity measured with the
number of preceding bars it is averaged over and the multiple that counts as a
match. These are TA-Lib's defaults and are not configurable.

| Setting | Measures | Averaged over | Factor |
| ------- | -------- | ------------- | ------ |
| `BODY_LONG` | real body | 10 bars | 1.0 |
| `BODY_VERY_LONG` | real body | 10 bars | 3.0 |
| `BODY_SHORT` | real body | 10 bars | 1.0 |
| `BODY_DOJI` | high-low range | 10 bars | 0.1 |
| `SHADOW_LONG` | real body | the bar itself | 1.0 |
| `SHADOW_VERY_LONG` | real body | the bar itself | 2.0 |
| `SHADOW_SHORT` | both shadows summed | 10 bars | 1.0, then halved |
| `SHADOW_VERY_SHORT` | high-low range | 10 bars | 0.1 |
| `NEAR` | high-low range | 5 bars | 0.2 |
| `FAR` | high-low range | 5 bars | 0.6 |
| `EQUAL` | high-low range | 5 bars | 0.05 |

The average always covers the bars *before* the bar being judged, never that
bar itself, so a formation is measured against history rather than against its
own size. A setting averaged over the bar itself (`SHADOW_LONG`,
`SHADOW_VERY_LONG`) compares a shadow with that same bar's body.

The geometry each setting reads is the usual one: the real body is
`|close - open|`, the upper shadow runs from the body's top to the high, the
lower shadow from the low to the body's bottom, and a bar counts as white when
`close >= open`, so a doji is white.

## Warm-up

A recogniser's leading nulls are the longest averaging period it consults plus
the bars its formation spans, which is TA-Lib's lookback. A null in any input
blanks that bar and the whole window that follows it.

## The patterns

| Function | Pattern | Leading nulls | Reports |
| -------- | ------- | ------------- | ------- |
| `cdl2crows` | Two Crows | 12 | `-100` |
| `cdl3blackcrows` | Three Black Crows | 13 | `-100` |
| `cdl3inside` | Three Inside Up/Down | 12 | `±100` |
| `cdl3linestrike` | Three-Line Strike | 8 | `±100` |
| `cdl3outside` | Three Outside Up/Down | 3 | `±100` |
| `cdl3starsinsouth` | Three Stars In The South | 12 | `100` |
| `cdl3whitesoldiers` | Three Advancing White Soldiers | 12 | `100` |
| `cdlabandonedbaby` | Abandoned Baby | 12 | `±100` |
| `cdladvanceblock` | Advance Block | 12 | `-100` |
| `cdlbelthold` | Belt-hold | 10 | `±100` |
| `cdlbreakaway` | Breakaway | 14 | `±100` |
| `cdlclosingmarubozu` | Closing Marubozu | 10 | `±100` |
| `cdlconcealbabyswall` | Concealing Baby Swallow | 13 | `100` |
| `cdlcounterattack` | Counterattack | 11 | `±100` |
| `cdldarkcloudcover` | Dark Cloud Cover | 11 | `-100` |
| `cdldoji` | Doji | 10 | `100` |
| `cdldojistar` | Doji Star | 11 | `±100` |
| `cdldragonflydoji` | Dragonfly Doji | 10 | `100` |
| `cdlengulfing` | Engulfing Pattern | 2 | `±100` |
| `cdleveningdojistar` | Evening Doji Star | 12 | `-100` |
| `cdleveningstar` | Evening Star | 12 | `-100` |
| `cdlgapsidesidewhite` | Up/Down-gap side-by-side white lines | 7 | `±100` |
| `cdlgravestonedoji` | Gravestone Doji | 10 | `100` |
| `cdlhammer` | Hammer | 11 | `100` |
| `cdlhangingman` | Hanging Man | 11 | `-100` |
| `cdlharami` | Harami Pattern | 11 | `±100` |
| `cdlharamicross` | Harami Cross Pattern | 11 | `±100` |
| `cdlhighwave` | High-Wave Candle | 10 | `±100` |
| `cdlhikkake` | Hikkake Pattern | 5 | `±100`, `±200` |
| `cdlhikkakemod` | Modified Hikkake Pattern | 10 | `±100`, `±200` |
| `cdlhomingpigeon` | Homing Pigeon | 11 | `100` |
| `cdlidentical3crows` | Identical Three Crows | 12 | `-100` |
| `cdlinneck` | In-Neck Pattern | 11 | `-100` |
| `cdlinvertedhammer` | Inverted Hammer | 11 | `100` |
| `cdlkicking` | Kicking | 11 | `±100` |
| `cdlkickingbylength` | Kicking, by the longer marubozu | 11 | `±100` |
| `cdlladderbottom` | Ladder Bottom | 14 | `100` |
| `cdllongleggeddoji` | Long Legged Doji | 10 | `100` |
| `cdllongline` | Long Line Candle | 10 | `±100` |
| `cdlmarubozu` | Marubozu | 10 | `±100` |
| `cdlmatchinglow` | Matching Low | 6 | `100` |
| `cdlmathold` | Mat Hold | 14 | `100` |
| `cdlmorningdojistar` | Morning Doji Star | 12 | `100` |
| `cdlmorningstar` | Morning Star | 12 | `100` |
| `cdlonneck` | On-Neck Pattern | 11 | `-100` |
| `cdlpiercing` | Piercing Pattern | 11 | `100` |
| `cdlrickshawman` | Rickshaw Man | 10 | `100` |
| `cdlrisefall3methods` | Rising/Falling Three Methods | 14 | `±100` |
| `cdlseparatinglines` | Separating Lines | 11 | `±100` |
| `cdlshootingstar` | Shooting Star | 11 | `-100` |
| `cdlshortline` | Short Line Candle | 10 | `±100` |
| `cdlspinningtop` | Spinning Top | 10 | `±100` |
| `cdlstalledpattern` | Stalled Pattern | 12 | `-100` |
| `cdlsticksandwich` | Stick Sandwich | 7 | `100` |
| `cdltakuri` | Takuri | 10 | `100` |
| `cdltasukigap` | Tasuki Gap | 7 | `±100` |
| `cdlthrusting` | Thrusting Pattern | 11 | `-100` |
| `cdltristar` | Tristar Pattern | 12 | `±100` |
| `cdlunique3river` | Unique 3 River | 12 | `100` |
| `cdlupsidegap2crows` | Upside Gap Two Crows | 12 | `-100` |
| `cdlxsidegap3methods` | Upside/Downside Gap Three Methods | 2 | `±100` |

## Penetration

Seven recognisers take an extra `penetration` argument, the fraction of an
earlier body a later close must reach into. TA-Lib's defaults are kept:

| Function | Default |
| -------- | ------- |
| `cdlabandonedbaby` | `0.3` |
| `cdldarkcloudcover` | `0.5` |
| `cdleveningdojistar` | `0.3` |
| `cdleveningstar` | `0.3` |
| `cdlmathold` | `0.5` |
| `cdlmorningdojistar` | `0.3` |
| `cdlmorningstar` | `0.3` |

## Example

```python
import polars as pl

from polars_ta import cdlengulfing, cdlhammer, cdlmorningstar

signals = bars.select(
    cdlengulfing("open", "high", "low", "close").alias("engulfing"),
    cdlhammer("open", "high", "low", "close").alias("hammer"),
    cdlmorningstar("open", "high", "low", "close").alias("morning_star"),
)

# Bars where any of the three fired, in either direction.
fired = pl.sum_horizontal(pl.all().abs()) > 0
hits = signals.filter(fired)
```
