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
- The public API exports `hello()`, `sma()`, and `ema()`.
- Unit tests live in `tests/`; tests use `unittest.TestCase` and are run with pytest as required by the project workflow.
- Python files are linted and formatted with Ruff; follow the workflow in `AGENTS.md` after Python code changes.
- Runtime dependencies are Polars and PyArrow. Development dependencies include pytest and Ruff.

## Module Layout

- Indicators are grouped by chart placement. `src/polars_ta/overlay/` holds indicators drawn on the price axis; within it, `ma.py` holds the moving averages (`sma`, `ema`). Future non-overlay groups (for example oscillators in a separate pane) get their own sibling package.
- `src/polars_ta/overlay/__init__.py` re-exports the group's public names so callers can import from either `polars_ta`, `polars_ta.overlay`, or `polars_ta.overlay.ma`.
- `src/polars_ta/_common.py` holds private shared helpers: the `IntoColumn` type alias, `validate_window`, `validate_alpha`, and `apply_to_column`. New indicator modules should reuse these rather than re-implementing validation or input dispatch.
- `src/polars_ta/__init__.py` re-exports the public indicator names with a sorted `__all__`.

## Indicator API Design

- Indicators are expression-first. A `str` or `pl.Expr` input yields a `pl.Expr`, which keeps the functions composable and lazy-compatible.
- A single function serves both eager and lazy use instead of separate `sma`/`sma_series` names. `apply_to_column` detects a `pl.Series` input, renames it to an internal placeholder, evaluates through a one-column frame, and restores the original name. The placeholder keeps unnamed series (`name == ""`) working.
- Static typing uses `typing.overload` so `Series -> Series` and `str | Expr -> Expr` are both correct.
- Validation happens once in the public function, before any expression is built, so errors surface at call time rather than at `collect()` time. Invalid `window`, `alpha`, or `mode` raise `ValueError`; an unsupported input type raises `TypeError`.
- There is deliberately no `min_periods` parameter. Warm-up is always `window - 1` nulls for every indicator and every EMA mode, so an SMA and an EMA of the same window align row for row. This is stricter than the pandas default.

## Indicator Algorithms

- `sma` is `rolling_mean(window_size=n, min_samples=n)`. `min_samples=n` yields both the warm-up nulls and null propagation at once, because Polars does not count a null as an observed sample; no extra masking expression is needed.
- `ema` supports three seeding conventions through `mode`:
  - `"talib"` (default) seeds with the SMA of the first complete window, matching TA-Lib's `EMA`.
  - `"recursive"` matches pandas `ewm(adjust=False)`.
  - `"adjust"` matches pandas `ewm(adjust=True)`.
- `mode` is a string literal rather than a raw `adjust: bool` plus a seed flag, because only three of the four flag combinations are meaningful.
- TA-Lib seeding is implemented without a Python loop: the input is rewritten so rows before the seed are null and the seed row carries the rolling mean, then `ewm_mean(adjust=False, min_samples=1)` continues the recursion. The seed row is located with `rolling.is_not_null().cum_sum() == 1`, so a null inside the warm-up window simply delays the seed to the first complete, null-free window.
- All `ewm_mean` calls use `ignore_nulls=False` (the Polars and pandas default) so that a gap caused by a null is reflected in the weighting of later rows.
- Default smoothing factor is `2 / (window + 1)`; an explicit `alpha` must satisfy `0 < alpha <= 1`.

## Testing Notes

- `tests/` mirrors the source grouping; `tests/overlay/test_ma.py` computes expected values with pure-Python reference implementations (`reference_sma`, `reference_ema_talib`, `reference_ema_recursive`, `reference_ema_adjust`) rather than importing pandas or NumPy.
- Float comparisons use `assertAlmostEqual(places=10)` via a shared `assert_values_equal` helper on an `IndicatorAssertions` base class, with `subTest` for per-row and per-mode diagnostics.
- Null and warm-up semantics are asserted explicitly so that a Polars upgrade changing `rolling_mean` or `ewm_mean` behavior surfaces as a test failure.
- Verified against Polars 1.44.2.