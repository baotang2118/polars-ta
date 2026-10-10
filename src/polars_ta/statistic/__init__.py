"""Statistical measures, which summarise a window of values rather than price action."""

from polars_ta.statistic.correlation import beta, correl
from polars_ta.statistic.dispersion import stddev, var, zscore
from polars_ta.statistic.regression import (
    linearreg,
    linearreg_angle,
    linearreg_intercept,
    linearreg_slope,
    tsf,
)

__all__ = [
    "beta",
    "correl",
    "linearreg",
    "linearreg_angle",
    "linearreg_intercept",
    "linearreg_slope",
    "stddev",
    "tsf",
    "var",
    "zscore",
]
