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
- The public API covers **every TA-Lib function listed in `indicators.md`** at the repository root. That file is a checklist; keep it in sync when adding or removing an indicator.
- Unit tests live in `tests/`; tests use `unittest.TestCase` and are run with pytest as required by the project workflow.
- Python files are linted and formatted with Ruff; follow the workflow in `AGENTS.md` after Python changes.
- Runtime dependencies are Polars and PyArrow. Development dependencies include pytest and Ruff.

## Module Layout

- Indicators are grouped by chart placement. `src/polars_ta/overlay/` holds indicators drawn on the price axis (`ma.py`, `adaptive.py`, `dispatch.py`, `midpoint.py`, `bands.py`, `channels.py`, `sar.py`, `supertrend.py`, `ichimoku.py`). `src/polars_ta/momentum/` holds oscillators drawn in a separate pane (`rsi.py`, `mfi.py`, `stoch.py`, `cci.py`, `macd.py`, `adx.py`, `aroon.py`, `bop.py`, `roc.py`, `price_oscillator.py`, `trix.py`, `ultosc.py`). `src/polars_ta/volume/` holds `flow.py`. `src/polars_ta/volatility/` holds magnitude-only measures (`atr.py`, which also exports `true_range` and `natr`). `src/polars_ta/cycle/` holds `hilbert.py`.
- `adx` lives in `momentum/` because TA-Lib classifies it as a momentum indicator, even though it is usually described as a trend-strength measure.
- Within a group, one module per *family* when indicators share machinery, one module per indicator when they are independent. Split a family module when it approaches ~500 lines. `adx.py` owns the whole directional-movement family (`plus_dm`, `minus_dm`, `plus_di`, `minus_di`, `dx`, `adx`, `adxr`) because they share one Wilder-sum engine; `stoch.py` owns `stoch`, `stochf`, `stochrsi`, and `willr` for the same reason; `rsi.py` owns `cmo`.
- `bands.py` and `channels.py` are named for the *category* rather than the lead indicator, unlike the rest. That is deliberate: they are the growth slots for the two classic envelope families — bands scale around an average, channels bound price between two edges — so `donchian` and `keltner` sit together without either module needing a rename. Public names drop the structural noun for the same reason `bbands` and `ichimoku` do: `donchian`, not `donchian_channel`.
- `momentum/rsi.py` imports the **public** `ema` from `overlay.ma` rather than a private helper, which keeps Wilder smoothing as a single implementation without a cross-group private import or a shared kernels module. `momentum/macd.py` does the same for its three EMA passes.
- `overlay/dispatch.py` holds `MaType`, `MA_TYPES`, `validate_ma_type`, `ma`, and `mavp`. It sits *above* `ma.py` and `adaptive.py` in the import graph so the dispatcher can reach every average without a cycle; indicators needing a runtime-selected average (`apo`, `ppo`, `macdext`) import from it.
- `src/polars_ta/_common.py` holds private shared helpers: the `IntoColumn` type alias, `validate_window`, `validate_alpha`, `validate_positive`, `apply_to_column`, and `apply_to_columns`.
- `src/polars_ta/_hilbert.py` is a second top-level private module, holding Ehlers' Hilbert transform. It is top-level rather than inside `cycle/` because `overlay/adaptive.py` needs it for `mama`, and a `cycle` -> `overlay` edge would be the wrong direction.
- `src/polars_ta/__init__.py` re-exports every public indicator name with a sorted `__all__`.

## Indicator API Design

- Indicators are expression-first. A `str` or `pl.Expr` input yields a `pl.Expr`, which keeps the functions composable and lazy-compatible.
- A single function serves both eager and lazy use instead of separate `sma`/`sma_series` names. `apply_to_column` detects a `pl.Series` input, renames it to an internal placeholder, evaluates through a one-column frame, and restores the original name. The placeholder keeps unnamed series (`name == ""`) working.
- `apply_to_columns` is the multi-input counterpart, used by `mfi`. Inputs must be uniform: all series (evaluated eagerly through a temporary frame, result named after the first input) or all names/expressions. Mixing raises `TypeError`, because a series carries its own data while a name only refers to a frame that may not exist.
- Multi-output indicators return a single `pl.struct` rather than a tuple, so one call stays one expression and still works inside `with_columns`. `bbands` exposes `lower`, `middle`, `upper`; `stoch` exposes `k`, `d`; `macd` exposes `macd`, `signal`, `histogram`. Callers use `.unnest()` or `.struct.field()`.
- Every field of a struct output starts on the same row **when TA-Lib emits them from one function**: `stoch` masks `k` where `d` is null, and `macd` masks `macd` and `histogram` where `signal` is null. Where the fields map to separate TA-Lib functions with different lookbacks, they keep their own warm-ups: `adx`'s `plus_di`/`minus_di` start `window - 1` rows before `adx`, because aligning them would discard good data. `donchian` and `ichimoku` likewise let each field depend only on its own inputs — a null `high` nulls Donchian's `upper` and `middle` but not `lower`.
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
- **`pl.max_horizontal` ignores nulls rather than propagating them.** `true_range` must therefore guard `high`, `low`, and the shifted `close` explicitly; without the guard the first bar silently emits `high - low` instead of null, which shortens every downstream warm-up (ATR, ADX, Supertrend) by one row. This was a real bug caught by the warm-up assertions.
- A second null-swallowing trap: `pl.when(cond).then(x).otherwise(default)` sends **null** conditions to the `otherwise` branch. In `adx` this turned the DI warm-up into `0.0` values. Any `when` chain over a nullable condition needs an explicit `.when(cond.is_null()).then(None)` (or a `~known` guard) first. The same pattern appears in `rsi`, `mfi`, `stoch`, `cci`, and `macd`.
- `adx` can use the public `ema` with `alpha=1/window` for all three smoothing passes even though TA-Lib keeps running *sums*: sum and average differ by a constant factor of `window`, which cancels in the `+DM / TR` ratio.
- `atr` and the ADX smoothing both reuse `ema(..., alpha=1/window)` for Wilder's smoothing, the same trick as `rsi`.
- `donchian` is `rolling_max`/`rolling_min`; it is not a TA-Lib function (the TA-Lib equivalents are the separate `MAX` and `MIN`).
- `keltner` is also outside TA-Lib. It reuses `_ema_expr` for the centre line and `volatility.atr._atr_expr` for the width rather than re-deriving either, the same cross-group private import `supertrend` already uses. Its centre and edges keep **separate** warm-ups (`window - 1` and `max(window - 1, atr_window)`) because they depend on different inputs, following the `donchian` and `ichimoku` precedent rather than the `bbands` one.
- `keltner` takes independent `window` and `atr_window` periods, defaulting to the common 20/10 convention, because the smoothing that suits a trend line rarely suits a volatility estimate.
- **`supertrend` is the only indicator that is not a pure expression.** Its band ratchet is a genuine sequential recursion — each band depends on the previous band *and* the previous direction, which depends on the previous band — with no Polars primitive to map onto (unlike EMA, which has `ewm_mean`). It uses a Python scan inside `map_batches`, which still returns a `pl.Expr` and still works lazily but is substantially slower. If a vectorized formulation is ever found, this is the place to apply it.
- `ichimoku`'s `lagging` field is `close.shift(-displacement)`, which puts **future** data on each row. Correct for plotting, a lookahead bug in a backtest signal; documented prominently in both README and the indicator reference. The forward-shifted spans are truncated at the end of the frame rather than extending past it, since a column cannot outgrow its frame.

## Indicator Algorithms, Continued

- `trima` is the convolution of two SMA passes: `(n+1)//2` twice for odd `n`, and `n//2 + 1` then `n//2` for even `n`. Both give the TA-Lib lookback of `n - 1` and reproduce TA-Lib's normalizing factors ((k)² and k(k+1)) exactly, without materializing a weight vector.
- `t3` chains six `_ema_expr` passes and blends the last four. The warm-up stays `6 * (window - 1)` even at `vfactor=0`, because `0.0 * null` is null in Polars — which also happens to match TA-Lib, whose lookback ignores `vfactor`. The chained passes make the expression tree large; the T3 tests are the slowest in the suite.
- **`_wilder_sum` in `adx.py` is the exact TA-Lib running sum, and its seed deliberately holds `window - 1` terms, not `window`.** This is what makes `plus_dm` emit at `window - 1` while `plus_di` emits at `window`: TA-Lib runs one extra smoothing step before the first indicator. The seed is produced by rewriting the input so the seed row carries `rolling_sum(window - 1) * (1 / window)` and the rest carries the raw value, then running `ewm_mean(alpha=1/window)` and multiplying back by `window`. The one-row DI offset is expressed as `tr_sum.shift(1).is_not_null()`.
- That seeding replaced an earlier `ema(dm, window, alpha=1/window)` in `adx`, which had the right lookbacks but a different seed (mean of `window` terms) and so diverged from TA-Lib on every row. The whole family now shares one engine.
- `adxr` is `(adx + adx.shift(window - 1)) / 2`, which produces the TA-Lib lookback of `3 * window - 2` with no extra bookkeeping.
- `aroon` needs the distance back to the most recent window extreme, which is not a rolling aggregate. It expands into `window + 1` shifted equality tests against the rolling extreme — `O(window)` expressions, the same trade-off as `cci`'s mean absolute deviation. Equality on floats is exact here because `rolling_max` returns a value that is literally in the window, and the `when` chain naturally resolves ties to the most recent bar, matching TA-Lib's `>=`/`<=`.
- `ultosc` must guard `min_horizontal`/`max_horizontal` against nulls for the same reason `true_range` does.
- `adosc` uses `ema(..., mode="recursive")`, not the default `"talib"` mode: TA-Lib seeds both A/D averages with the first A/D reading rather than with an SMA.
- `obv` seeds the running total with the first bar's volume by overriding row 0 with `pl.int_range(pl.len()) == 0`, then `cum_sum`.
- Degenerate-window conventions continue to differ by indicator, all verified against TA-Lib: `cmo` reports `0.0` where `rsi` reports `50.0`; `willr` and `stoch` report `0.0` for a flat range; the `roc` family reports `0.0` for a zero reference price; `natr` and `ppo` report `0.0` for a zero denominator; `bop` and `ad` report `0.0` for a bar with no range. The one deliberate divergence is the directional family, where a zero range sum gives `0.0` here but leaves the previous ADX unchanged in TA-Lib.
- `kama`, `sar`, `sarext`, `mama`, and the `ht_*` indicators join `supertrend` as Python scans inside `map_batches`. `kama`'s recursion has a *varying* alpha, so `ewm_mean` cannot be used even though the efficiency ratio itself is a plain expression; the scan therefore receives the precomputed factor alongside the value and the seed.
- `_hilbert.py` runs the entire TA-Lib Hilbert recursion once and returns all ten derived series in a `HilbertSeries` named tuple. Running one scan for seven indicators keeps the arithmetic in one place, and is correct because every TA-Lib HT function evolves identical state — they differ only in which series they emit and in their lookback (32 or 63). `mask_lookback` applies that difference.
- The Hilbert scan starts at index 0, primes the four-period price smoother over the first 12 bars, then runs the main loop from index 12. That reproduces TA-Lib exactly for the natural case `startIdx == lookbackTotal`, which is the only case a whole-column implementation has.
- `ht_trendline` averages the **raw** price over the dominant cycle, not the smoothed price; TA-Lib issue #88 corrected this. The dominant-cycle-phase loop, by contrast, reads the 50-entry smoothed-price circular buffer.
- `mavp` builds one average per candidate period and selects with a `when` chain, so its expression size is linear in `max_period - min_period`. Output is masked to the `max_period` warm-up so the lookback does not vary row to row, as in TA-Lib.
- `ma(..., ma_type="mama")` returns the MAMA line and ignores `window`, matching TA-Lib's `MA` with `MAType_MAMA`.


## Testing Notes

- `tests/` mirrors the source grouping: `tests/overlay/`, `tests/momentum/`, `tests/volume/`, `tests/volatility/`, and `tests/cycle/`. Test modules are named for the *family* they cover, so one file may exercise several related functions (`test_ma_family.py`, `test_directional.py`, `test_stoch_family.py`, `test_aroon_bop.py`, `test_trix_ultosc_macd_variants.py`). Each computes expected values with pure-Python reference implementations rather than importing pandas or NumPy.
- Warm-up assertions are parameterized over several window sizes rather than hard-coding one. This is what caught the `max_horizontal` null bug, which shifted every ATR-derived warm-up by exactly one row.
- `tests/_assertions.py` holds the shared `IndicatorAssertions` base class. `pyproject.toml` sets `pythonpath = ["src", "tests"]` so test modules can import it directly.
- `tests/_data.py` is the single source of test data; no test module defines price literals of its own. It publishes three datasets, each for a distinct reason:
  - `OPEN`/`CLOSE`/`HIGH`/`LOW`/`VOLUME` — 120 canonical bars, long enough to exercise every indicator's default parameters (the long-lookback Hilbert indicators need 63, Ichimoku needs `52 + 26`). Generated by a small deterministic LCG walk rather than `random`, so the data is stable across runs without hard-coding hundreds of literals.
  - `HAND_CHECKED` — the short series the hand-computed moving-average literals were derived from, kept verbatim so those assertions stay genuine checks rather than change detectors.
  - `WILDER_CLOSE` — Wilder's published worked example, an external anchor for RSI.
- Bars are deliberately symmetric about the close (`high = close + spread`, `low = close - spread`), which keeps the typical price exactly equal to the close and so keeps the MFI and CCI hand-checks tractable. `OPEN` sits inside each bar's range on a repeating 0.5 / 0.0 / -0.5 offset so `bop` has something non-degenerate to measure.
- `_data.py` also owns the scenario builders (`ramp_up`, `ramp_down`, `constant`, `frame_from`, `with_null`) and the `frame()` factory. They are named to avoid colliding with the `rising`/`falling`/`flat` local variables in the tests. `frame()` truncates the canonical columns to match any shorter replacement column, so a test can pass a handful of synthetic bars without supplying every column.
- Tests must assert properties, not incidental geometry. The ADX directional tests originally asserted `plus_di == 100` on a rising series, which only held because that fixture's bar range happened to equal its step size; they now assert the real claim, that the opposite indicator is zero.
- The directional-movement and Hilbert tests are the exceptions that *do* pin values, because both reproduce intricate TA-Lib state machines where a property test would not catch a seeding error. `test_directional.py` carries a step-by-step Wilder-sum reference; `test_hilbert.py` compares each public indicator against one shared `hilbert_transform` call, which checks the wiring rather than the arithmetic, and backs that with property tests (bounded sine, clamped period, flat-series behaviour, a straight line always reading as trending).
- `rsi` is pinned to Wilder's published worked example, whose first 14-period output is `70.4641` — an external check that does not depend on the reference implementation in the same file.
- `reference_ema_talib` accepts nullable input and reproduces the seed-at-first-complete-window rule, which lets `reference_dema` and `reference_tema` simply chain it.
- Float comparisons use `assertAlmostEqual(places=10)` via a shared `assert_values_equal` helper on an `IndicatorAssertions` base class, with `subTest` for per-row and per-mode diagnostics. Bound assertions on composed oscillators (`stochrsi`) must tolerate rounding past the nominal limit; comparing with `assertAlmostEqual` against the clamped value is the pattern used.
- Null and warm-up semantics are asserted explicitly so that a Polars upgrade changing `rolling_mean` or `ewm_mean` behavior surfaces as a test failure.
- Verified against Polars 1.44.2.