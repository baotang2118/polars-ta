---
name: pytest-unit-tests
description: 'Run Python unit tests with pytest after Python code changes. Use when implementing, refactoring, debugging, or validating Python code and tests in this workspace.'
argument-hint: 'Optional test file, directory, or pytest selection'
---
# Python Unit Tests with Pytest

Always run relevant Python unit tests after a Python code change, even when the user did not explicitly request testing. Use pytest as the runner; do not switch to `unittest` or another test runner.

## Procedure

1. Run tests from the workspace root. Use the full suite unless the request identifies a narrower scope:

   ```sh
   uv run pytest
   ```

2. For a targeted run, pass the relevant test file or directory to pytest:

   ```sh
   uv run pytest path/to/test_file.py
   ```

3. Report the command, outcome, and any failures. If `uv` or pytest is unavailable, report that validation could not run; do not substitute a different test runner.