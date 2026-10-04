---
name: lint-format-code
description: 'Lint, autofix, and format Python code with Ruff. Use whenever writing, editing, checking, or cleaning up Python files in this workspace.'
argument-hint: 'Python files or workspace scope to lint and format'
---

# Lint and Format Python Code

Use this workflow when asked to lint, format, or clean up Python code in this workspace.

## Procedure

1. Run the commands from the workspace root. These commands intentionally check the whole workspace:

	```sh
	uv run ruff check .
	uv run ruff check --fix .
	uv run ruff format .
	uv run ruff check .
	```

2. Review the final lint output. Fix any remaining violations that are in scope, then rerun the relevant Ruff checks.
3. Report which commands ran, whether they passed, and any violations or files changed. If `uv` or Ruff is unavailable, report that instead of claiming validation succeeded.