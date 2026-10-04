# polars-ta

A Python technical-indicator library designed around Polars and PyArrow.

Indicators are expression-first: they return a `pl.Expr` that composes inside `select`/`with_columns` and runs lazily, and they also accept a `pl.Series` for eager use.

## Indicators

Indicators are grouped by how they are charted. *Overlay* indicators are drawn on the price axis, and moving averages live in `polars_ta.overlay.ma`. Every public indicator is also re-exported from the package root.

| Group | Module | Function | Description |
| ----- | ------ | -------- | ----------- |
| Overlay | `overlay.ma` | `sma(column, window)` | Simple moving average |
| Overlay | `overlay.ma` | `ema(column, window, *, alpha=None, mode="talib")` | Exponential moving average |

```python
import polars as pl
from polars_ta import ema, sma  # or: from polars_ta.overlay.ma import ema, sma

df = pl.DataFrame({"close": [1.0, 3.0, 2.0, 6.0, 5.0, 9.0]})
df.with_columns(
    sma("close", 3).alias("sma_3"),
    ema("close", 3).alias("ema_3"),
)
```

Each function takes a column name, a `pl.Expr`, or a `pl.Series`:

```python
sma("close", 3)  # pl.Expr
sma(pl.col("close"), 3)  # pl.Expr
sma(pl.Series("close", [1.0, 2.0]), 3)  # pl.Series
```

`ema` defaults to the TA-Lib convention, seeding the recursion with the simple moving average of the first complete window. Pass `mode="recursive"` or `mode="adjust"` for the pandas `ewm(adjust=False)` and `ewm(adjust=True)` conventions, or `alpha=` to override the default smoothing factor of `2 / (window + 1)`.

The first `window - 1` rows are null for every indicator and mode. See [docs/indicators.md](docs/indicators.md) for the formulas, null-handling rules, and worked examples.

## Development

Install the package and development tools, then run the unit tests with pytest:

```sh
uv sync --dev
uv run pytest
```

The starter test uses `unittest.TestCase` assertions and is executed with pytest. Follow [AGENTS.md](AGENTS.md) for the Python linting and formatting workflow.

## Project References

- [docs/indicators.md](docs/indicators.md): indicator formulas, conventions, and null handling.
- [KNOWLEDGE.md](KNOWLEDGE.md): package structure, design principles, and dependency guidance.
- [AGENTS.md](AGENTS.md): repository-wide development and validation requirements.
