# Project Knowledge

## Purpose

This project develops a Python data and technical-analysis library centered on Polars.

## Core Libraries

- Prefer Polars for dataframe operations and PyArrow where columnar interchange or Arrow-native functionality is needed.
- Avoid Pandas and NumPy when the requirement can be met with Polars or PyArrow.
- Avoid TA-Lib, its Python wrapper, pandas-ta, and pandas-ta-classic when the required behavior can be implemented with the preferred core libraries.
- Ask the user for explicit approval before using or adding any of those alternative dependencies.

## Behavioral References

Use these projects to check expected behavior and terminology. Do not copy their source code:

- https://github.com/TA-Lib/ta-lib
- https://github.com/ta-lib/ta-lib-python
- https://github.com/xgboosted/pandas-ta-classic
- https://github.com/bukosabino/ta
- https://github.com/aarigs/pandas-ta

## Sources of Project Guidance

- `AGENTS.md` defines repository-wide development, documentation, and validation requirements.
- `.github/instructions/` contains task-relevant Python instructions.
- `.agents/skills/` contains the detailed pytest and Ruff workflows.
- `docs/indicators.md` is the formula reference for implemented indicators; keep it in sync with the code.
- Update this file with implementation, API, algorithm, and maintenance knowledge as the project evolves; keep user-facing behavior documented in `README.md`.

## Package Scaffold

- Installable package source lives in `src/polars_ta/` and is configured through `pyproject.toml`.
- The public API exports `sma()`, `wma()`, `ema()`, `dema()`, and `tema()`.
- Unit tests live in `tests/`; tests use `unittest.TestCase` and are run with pytest as required by the project workflow.
- Python files are linted and formatted with Ruff; follow the workflow in `AGENTS.md` after Python code changes.
- Runtime dependencies are Polars and PyArrow. Development dependencies include pytest and Ruff.

## Module Layout

- Indicators are grouped by chart placement. `src/polars_ta/overlay/` holds indicators drawn on the price axis; within it, `ma.py` holds the moving averages (`sma`, `wma`, `ema`, `dema`, `tema`). Future non-overlay groups (for example oscillators in a separate pane) get their own sibling package.
- `src/polars_ta/overlay/__init__.py` re-exports the group's public names so callers can import from either `polars_ta`, `polars_ta.overlay`, or `polars_ta.overlay.ma`.
- `src/polars_ta/_common.py` holds private shared helpers: the `IntoColumn` type alias, `validate_window`, `validate_alpha`, and `apply_to_column`. New indicator modules should reuse these rather than re-implementing validation or input dispatch.
- `src/polars_ta/__init__.py` re-exports the public indicator names with a sorted `__all__`.

## Indicator API Design

- Indicators are expression-first. A `str` or `pl.Expr` input yields a `pl.Expr`, which keeps the functions composable and lazy-compatible.
- A single function serves both eager and lazy use instead of separate `sma`/`sma_series` names. `apply_to_column` detects a `pl.Series` input, renames it to an internal placeholder, evaluates through a one-column frame, and restores the original name. The placeholder keeps unnamed series (`name == ""`) working.
- Static typing uses `typing.overload` so `Series -> Series` and `str | Expr -> Expr` are both correct.
- Validation happens once in the public function, before any expression is built, so errors surface at call time rather than at `collect()` time. Invalid `window`, `alpha`, or `mode` raise `ValueError`; an unsupported input type raises `TypeError`.
- `_resolve_alpha(window, alpha, mode)` is the single validation entry point shared by `ema`, `dema`, and `tema`; it validates all three arguments and returns the effective smoothing factor.
- There is deliberately no `min_periods` parameter. Warm-up is `window - 1` nulls for the single-pass averages in every EMA mode, so an SMA and an EMA of the same window align row for row. This is stricter than the pandas default. `dema` and `tema` extend it to `2 * (window - 1)` and `3 * (window - 1)`, matching the TA-Lib lookbacks.

## Indicator Algorithms

- `sma` is `rolling_mean(window_size=n, min_samples=n)`. `min_samples=n` yields both the warm-up nulls and null propagation at once, because Polars does not count a null as an observed sample; no extra masking expression is needed.
- `wma` uses `rolling_mean(window_size=n, weights=[1.0 .. n])`, which Polars normalizes by the weight sum, giving the standard linear-weight WMA directly.
- Polars **panics** (`PanicException: weights not yet supported on array with null values`) when a weighted rolling aggregation meets a null, so `wma` cannot rely on `min_samples` for null propagation. The input is cast to `Float64` and `fill_null(0.0)`-ed before the weighted call, then rows whose window contained a null are masked back to null using `is_null().cast(UInt32).rolling_sum(n, min_samples=n) > 0`. The mask covers exactly the rows the fill would have corrupted, so the result still matches the SMA null rule. Re-check this workaround if Polars gains null support for weighted rolling windows.
- `ema` supports three seeding conventions through `mode`:
  - `"talib"` (default) seeds with the SMA of the first complete window, matching TA-Lib's `EMA`.
  - `"recursive"` matches pandas `ewm(adjust=False)`.
  - `"adjust"` matches pandas `ewm(adjust=True)`.
- `mode` is a string literal rather than a raw `adjust: bool` plus a seed flag, because only three of the four flag combinations are meaningful.
- TA-Lib seeding is implemented without a Python loop: the input is rewritten so rows before the seed are null and the seed row carries the rolling mean, then `ewm_mean(adjust=False, min_samples=1)` continues the recursion. The seed row is located with `rolling.is_not_null().cum_sum() == 1`, so a null inside the warm-up window simply delays the seed to the first complete, null-free window.
- All `ewm_mean` calls use `ignore_nulls=False` (the Polars and pandas default) so that a gap caused by a null is reflected in the weighting of later rows.
- Default smoothing factor is `2 / (window + 1)`; an explicit `alpha` must satisfy `0 < alpha <= 1`.
- `dema` is `2 * EMA - EMA(EMA)` and `tema` is `3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))`, built by feeding `_ema_expr` its own output expression. No offset bookkeeping is needed: the `"talib"` seeding rule seeds at the first complete, null-free window, so chaining passes produces the `2 * (window - 1)` and `3 * (window - 1)` TA-Lib lookbacks automatically. The `"recursive"` and `"adjust"` modes reach the same warm-up because `ewm_mean`'s `min_samples` counts non-null observations.
- Chained passes recompute the inner EMA expressions (for example `tema` evaluates the first pass three times). This is correctness-neutral and was left unoptimized; revisit only if profiling shows it matters.

## Testing Notes

- `tests/` mirrors the source grouping; `tests/overlay/test_ma.py` computes expected values with pure-Python reference implementations (`reference_sma`, `reference_wma`, `reference_ema_talib`, `reference_ema_recursive`, `reference_ema_adjust`, `reference_dema`, `reference_tema`) rather than importing pandas or NumPy.
- `reference_ema_talib` accepts nullable input and reproduces the seed-at-first-complete-window rule, which lets `reference_dema` and `reference_tema` simply chain it.
- Float comparisons use `assertAlmostEqual(places=10)` via a shared `assert_values_equal` helper on an `IndicatorAssertions` base class, with `subTest` for per-row and per-mode diagnostics.
- Null and warm-up semantics are asserted explicitly so that a Polars upgrade changing `rolling_mean` or `ewm_mean` behavior surfaces as a test failure.
- Verified against Polars 1.44.2.