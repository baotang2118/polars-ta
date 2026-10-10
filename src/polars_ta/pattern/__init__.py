"""Candlestick pattern recognisers, which score the shape of recent bars.

Each function reads the open, high, low, and close of a bar and the bars
before it, and reports an ``Int32``: ``0`` when the pattern is absent, ``100``
for a bullish reading, ``-100`` for a bearish one. The hikkake pair adds
``±200`` for a later bar that confirms an earlier reading. "Long", "short",
"near", and the rest are judged against the preceding bars, so the leading
rows of a frame, where there is nothing to compare against, are null.
"""

from polars_ta.pattern.continuation import (
    cdlladderbottom,
    cdlmathold,
    cdlrisefall3methods,
)
from polars_ta.pattern.crows_soldiers import (
    cdl2crows,
    cdl3blackcrows,
    cdl3whitesoldiers,
    cdladvanceblock,
    cdlidentical3crows,
    cdlstalledpattern,
    cdlupsidegap2crows,
)
from polars_ta.pattern.doji import (
    cdldoji,
    cdldojistar,
    cdldragonflydoji,
    cdlgravestonedoji,
    cdllongleggeddoji,
    cdlrickshawman,
    cdltakuri,
)
from polars_ta.pattern.engulfing import (
    cdlcounterattack,
    cdldarkcloudcover,
    cdlengulfing,
    cdlharami,
    cdlharamicross,
    cdlkicking,
    cdlkickingbylength,
    cdlmatchinglow,
    cdlpiercing,
    cdlseparatinglines,
)
from polars_ta.pattern.gaps import (
    cdlbreakaway,
    cdlgapsidesidewhite,
    cdltasukigap,
    cdlxsidegap3methods,
)
from polars_ta.pattern.hikkake import cdlhikkake, cdlhikkakemod
from polars_ta.pattern.necklines import (
    cdlhomingpigeon,
    cdlinneck,
    cdlonneck,
    cdlthrusting,
)
from polars_ta.pattern.single import (
    cdlbelthold,
    cdlclosingmarubozu,
    cdlhammer,
    cdlhangingman,
    cdlhighwave,
    cdlinvertedhammer,
    cdllongline,
    cdlmarubozu,
    cdlshootingstar,
    cdlshortline,
    cdlspinningtop,
)
from polars_ta.pattern.stars import (
    cdl3starsinsouth,
    cdlabandonedbaby,
    cdleveningdojistar,
    cdleveningstar,
    cdlmorningdojistar,
    cdlmorningstar,
    cdltristar,
)
from polars_ta.pattern.three_bar import (
    cdl3inside,
    cdl3linestrike,
    cdl3outside,
    cdlconcealbabyswall,
    cdlsticksandwich,
    cdlunique3river,
)

__all__ = [
    "cdl2crows",
    "cdl3blackcrows",
    "cdl3inside",
    "cdl3linestrike",
    "cdl3outside",
    "cdl3starsinsouth",
    "cdl3whitesoldiers",
    "cdlabandonedbaby",
    "cdladvanceblock",
    "cdlbelthold",
    "cdlbreakaway",
    "cdlclosingmarubozu",
    "cdlconcealbabyswall",
    "cdlcounterattack",
    "cdldarkcloudcover",
    "cdldoji",
    "cdldojistar",
    "cdldragonflydoji",
    "cdlengulfing",
    "cdleveningdojistar",
    "cdleveningstar",
    "cdlgapsidesidewhite",
    "cdlgravestonedoji",
    "cdlhammer",
    "cdlhangingman",
    "cdlharami",
    "cdlharamicross",
    "cdlhighwave",
    "cdlhikkake",
    "cdlhikkakemod",
    "cdlhomingpigeon",
    "cdlidentical3crows",
    "cdlinneck",
    "cdlinvertedhammer",
    "cdlkicking",
    "cdlkickingbylength",
    "cdlladderbottom",
    "cdllongleggeddoji",
    "cdllongline",
    "cdlmarubozu",
    "cdlmatchinglow",
    "cdlmathold",
    "cdlmorningdojistar",
    "cdlmorningstar",
    "cdlonneck",
    "cdlpiercing",
    "cdlrickshawman",
    "cdlrisefall3methods",
    "cdlseparatinglines",
    "cdlshootingstar",
    "cdlshortline",
    "cdlspinningtop",
    "cdlstalledpattern",
    "cdlsticksandwich",
    "cdltakuri",
    "cdltasukigap",
    "cdlthrusting",
    "cdltristar",
    "cdlunique3river",
    "cdlupsidegap2crows",
    "cdlxsidegap3methods",
]
