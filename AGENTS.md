# Repository Agent Guidelines

These guidelines apply to all work in this repository. Follow relevant file-specific instructions and skills for detailed procedures.

## Project Principles

- Prefer Polars for dataframe operations and PyArrow where Arrow interoperability or Arrow-native functionality is needed.
- Avoid Pandas, NumPy, TA-Lib, and indicator wrappers when the core libraries can meet the requirement. Ask the user for explicit approval before using or adding these alternatives.
- Inspect existing code and conventions before making changes.

## Documentation

- For every code change, update both `README.md` and `KNOWLEDGE.md` with accurate, relevant information. Keep the README useful to library users and record implementation, API, algorithm, testing, or maintenance knowledge in KNOWLEDGE. Include a concise note for test-only or internal changes.
- Keep documentation consistent with the implemented behavior; do not invent APIs or capabilities.

## Python Validation

- Always use pytest for Python unit tests. Follow the `pytest-unit-tests` skill.
- After generating or modifying Python code, run the complete lint and formatting workflow in the `lint-format-code` skill before finishing.
- Report the validation commands actually run, their results, and any unresolved issues. If a required tool is unavailable, say so rather than claiming validation passed.