# polars-ta

A Python technical-indicator library designed around Polars and PyArrow.

The initial scaffold includes a small `hello()` API; technical indicators will be added in subsequent development.

## Development

Install the package and development tools, then run the unit tests with pytest:

```sh
uv sync --dev
uv run pytest
```

The starter test uses `unittest.TestCase` assertions and is executed with pytest. Follow [AGENTS.md](AGENTS.md) for the Python linting and formatting workflow.

## Project References

- [KNOWLEDGE.md](KNOWLEDGE.md): package structure, design principles, and dependency guidance.
- [AGENTS.md](AGENTS.md): repository-wide development and validation requirements.
