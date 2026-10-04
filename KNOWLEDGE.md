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
- Update this file with implementation, API, algorithm, and maintenance knowledge as the project evolves; keep user-facing behavior documented in `README.md`.

## Package Scaffold

- Installable package source lives in `src/polars_ta/` and is configured through `pyproject.toml`.
- The initial public API exports `hello()`, which returns `"Hello, world!"`. No technical indicators are implemented yet.
- Unit tests live in `tests/`; the starter test uses `unittest.TestCase` and is run with pytest as required by the project workflow.
- Starter Python files are linted and formatted with Ruff; follow the workflow in `AGENTS.md` after Python code changes.
- Runtime dependencies are Polars and PyArrow. Development dependencies include pytest and Ruff.