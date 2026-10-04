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
- The public API exports `sma()`, `wma()`, `ema()`, `dema()`, `tema()`, `bbands()`, `rsi()`, `mfi()`, `stoch()`, `cci()`, and `macd()`.
- Unit tests live in `tests/`; tests use `unittest.TestCase` and are run with pytest as required by the project workflow.
- Python files are linted and formatted with Ruff; follow the workflow in `AGENTS.md` after Python code changes.
- Runtime dependencies are Polars and PyArrow. Development dependencies include pytest and Ruff.

## Module Layout

- Indicators are grouped by chart placement. `src/polars_ta/overlay/` holds indicators drawn on the price axis (`ma.py` for the moving averages `sma`, `wma`, `ema`, `dema`, `tema`; `bands.py` for `bbands`). `src/polars_ta/momentum/` holds oscillators drawn in a separate pane (`rsi.py`, `mfi.py`, `stoch.py`, `cci.py`, `macd.py`).
- Within a group, one module per *family* when indicators share machinery (the moving averages), one module per indicator when they are independent (every momentum indicator so far). Split a family module when it approaches ~500 lines.
- `src/polars_ta/overlay/__init__.py` and `src/polars_ta/momentum/__init__.py` re-export the group's public names so callers can import from either `polars_ta`, the group, or the leaf module.
- `momentum/rsi.py` imports the **public** `ema` from `overlay.ma` rather than a private helper, which keeps Wilder smoothing as a single implementation without a cross-group private import or a shared kernels module. `momentum/macd.py` does the same for its three EMA passes.
- `src/polars_ta/_common.py` holds private shared helpers: the `IntoColumn` type alias, `validate_window`, `validate_alpha`, `validate_positive`, `apply_to_column`, and `apply_to_columns`. New indicator modules should reuse these rather than re-implementing validation or input dispatch.
- `src/polars_ta/__init__.py` re-exports the public indicator names with a sorted `__all__`.

## Indicator API Design

- Indicators are expression-first. A `str` or `pl.Expr` input yields a `pl.Expr`, which keeps the functions composable and lazy-compatible.
- A single function serves both eager and lazy use instead of separate `sma`/`sma_series` names. `apply_to_column` detects a `pl.Series` input, renames it to an internal placeholder, evaluates through a one-column frame, and restores the original name. The placeholder keeps unnamed series (`name == ""`) working.
- `apply_to_columns` is the multi-input counterpart, used by `mfi`. Inputs must be uniform: all series (evaluated eagerly through a temporary frame, result named after the first input) or all names/expressions. Mixing raises `TypeError`, because a series carries its own data while a name only refers to a frame that may not exist.
- Multi-output indicators return a single `pl.struct` rather than a tuple, so one call stays one expression and still works inside `with_columns`. `bbands` exposes `lower`, `middle`, `upper`; `stoch` exposes `k`, `d`; `macd` exposes `macd`, `signal`, `histogram`. Callers use `.unnest()` or `.struct.field()`.
- Every field of a struct output starts on the same row, matching TA-Lib's aligned output arrays. `stoch` masks `k` where `d` is null, and `macd` masks `macd` and `histogram` where `signal` is null. Without this the faster field would begin several rows earlier and silently misalign against a TA-Lib comparison.
- Static typing uses `typing.overload` so `Series -> Series` and `str | Expr -> Expr` are both correct.
- Validation happens once in the public function, before any expression is built, so errors surface at call time rather than at `collect()` time. Invalid `window`, `alpha`, or `mode` raise `ValueError`; an unsupported input type raises `TypeError`.
- `_resolve_alpha(window, alpha, mode)` is the single validation entry point shared by `ema`, `dema`, and `tema`; it validates all three arguments and returns the effective smoothing factor.
- There is deliberately no `min_periods` parameter. Warm-up is `window - 1` nulls for the single-pass averages in every EMA mode, so an SMA and an EMA of the same window align row for row. This is stricter than the pandas default. `dema` and `tema` extend it to `2 * (window - 1)` and `3 * (window - 1)`; `rsi` and `mfi` use `window` because they consume one row to a difference. Every figure matches the corresponding TA-Lib lookback.

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
- `bbands` is `rolling_mean` plus `rolling_std`, both with `min_samples=window`, wrapped in `pl.struct`. `ddof` defaults to `0` (population, TA-Lib) because Polars' `rolling_std` defaults to `1`; it is validated against `0 <= ddof < window`.
- `rsi` splits the one-period `diff` with `clip(lower_bound=0.0)` rather than a `when/otherwise` chain, because `clip` preserves the leading null. A `when(delta > 0)` chain would send the null down the `otherwise` branch and emit `0.0`, silently shortening the warm-up by a row. Both sides are then smoothed by the public `ema` with `alpha=1/window`.
- `mfi` builds positive and negative flows with an explicit `known = change.is_not_null() & flow.is_not_null()` guard for the same reason, then uses plain `rolling_sum` — MFI is **not** Wilder-smoothed, unlike RSI.
- Degenerate-window conventions were verified against TA-Lib sources and genuinely differ between the two indicators: `rsi` reports `50.0` when there is neither a gain nor a loss (TA-Lib issue #480 changed this from `0.0`), while `mfi` reports `0.0` when a window received no money flow. Do not "harmonize" these.- `mfi` compares the typical-price change against exact zero; TA-Lib uses a relative epsilon dead-zone, so results can differ only for float-rounding-scale moves. The same applies to the flatness tests in `stoch` and `cci`.
- `stoch` is `rolling_min`/`rolling_max` for the range, then two `rolling_mean` passes (fast %K to %K to %D). TA-Lib's `STOCH` returns the *slow* lines, which is what `stoch` returns; the fast pair (`STOCHF`) is not implemented.
- `cci`'s mean absolute deviation measures each window value against that window's own mean, which changes every row, so it is **not** expressible as a rolling aggregate. `rolling_std` is not a substitute: that is a root-mean-square deviation, not a mean-absolute one. It expands into `window` shifted terms summed together, which is `O(window)` expressions — acceptable for conventional periods, but the reason to be wary of a very large `window`.
- Degenerate-window conventions, all verified against TA-Lib sources: `rsi` reports `50.0`, `mfi` reports `0.0`, `stoch` gives a raw %K of `0.0` for a flat range, and `cci` reports `0.0` for zero deviation. These genuinely differ between indicators; do not "harmonize" them.
- `macd` is three `ema` calls. The TA-Lib seeding rule makes the combined warm-up `(slow - 1) + (signal - 1)` fall out automatically, with no offset arithmetic.

## Testing Notes

- `tests/` mirrors the source grouping: `tests/overlay/test_ma.py`, `tests/overlay/test_bands.py`, and `tests/momentum/test_{rsi,mfi,stoch,cci,macd}.py`. Each computes expected values with pure-Python reference implementations rather than importing pandas or NumPy.
- `tests/_assertions.py` holds the shared `IndicatorAssertions` base class. `pyproject.toml` sets `pythonpath = ["src", "tests"]` so test modules can import it directly.
- `rsi` is pinned to Wilder's published worked example, whose first 14-period output is `70.4641` — an external check that does not depend on the reference implementation in the same file.
- `reference_ema_talib` accepts nullable input and reproduces the seed-at-first-complete-window rule, which lets `reference_dema` and `reference_tema` simply chain it.
- Float comparisons use `assertAlmostEqual(places=10)` via a shared `assert_values_equal` helper on an `IndicatorAssertions` base class, with `subTest` for per-row and per-mode diagnostics.
- Null and warm-up semantics are asserted explicitly so that a Polars upgrade changing `rolling_mean` or `ewm_mean` behavior surfaces as a test failure.
- Verified against Polars 1.44.2.