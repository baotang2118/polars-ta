# Repository Agent Guidelines

These guidelines apply to all work in this repository.

## Project Principles

- Prefer Polars for dataframe operations and PyArrow where Arrow interoperability or Arrow-native functionality is needed.
- Avoid Pandas, NumPy, TA-Lib, and indicator wrappers when the core libraries can meet the requirement. Ask the user for explicit approval before using or adding these alternatives.

## Project Layout

- `src/polars_ta/`: indicators grouped by chart placement (`overlay/`, `momentum/`, `volume/`, `volatility/`, `cycle/`, `returns/`), plus private `_common.py` and `_hilbert.py`.
- `tests/`: pytest suite mirroring the source groups; shared helpers in `_assertions.py` and `_data.py`.
- `docs/`: indicator reference pages, starting at `indicators.md`.
- `examples/`: runnable quick-start scripts.

## Documentation

- For every code change, update `README.md` and `KNOWLEDGE.md`. Keep the README useful to library users; record implementation, API, algorithm, testing, or maintenance knowledge in KNOWLEDGE. Add a concise note for test-only or internal changes.
- Keep documentation consistent with the implemented behavior; do not invent APIs or capabilities.

## Python Validation

- After every Python code change, run the relevant tests with pytest from the project root. Run the full suite unless the change has a clearly narrower test scope.
- After every Python code change, lint and format the relevant scope with Ruff, using the project configuration and environment.
- Report the validation commands actually run, their results, and any unresolved issues. If tests are missing, cannot run, or fail, or a required tool is unavailable, say so rather than claiming success.