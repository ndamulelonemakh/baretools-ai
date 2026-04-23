# baretools-ai — Agent Guide

Minimal Python library for LLM tool calling. Zero runtime dependencies, supports OpenAI, Anthropic, Gemini, and generic JSON Schema.

For deeper conventions (style, provider parity, testing, commits, safety) see [.github/instructions/baretools-conventions.instructions.md](.github/instructions/baretools-conventions.instructions.md).

## Architecture

- [src/baretools/core.py](src/baretools/core.py) — single source module containing `tool`, `ToolRegistry`, `parse_tool_calls`, `format_tool_results` and supporting `TypedDict` shapes. Public API is re-exported from [src/baretools/__init__.py](src/baretools/__init__.py).
- [tests/test_core.py](tests/test_core.py) — exercises only the public API.
- [examples/](examples/) — runnable provider-specific agents (`openai_agent.py`, `anthropic_agent.py`, `gemini_agent.py`). Example-only deps live in `examples/requirements.examples.txt`, never in the package.
- [docs/](docs/) — Jekyll site published to GitHub Pages.

## Hard Invariants

- `dependencies = []` in [pyproject.toml](pyproject.toml) is permanent. New runtime functionality must use stdlib only. Dev tooling goes in `[dependency-groups]`.
- Python 3.10–3.13. Every `.py` file starts with `from __future__ import annotations`.
- Four supported providers — `openai`, `anthropic`, `gemini`, `json_schema`. Any change to schema generation, tool-call parsing, or result formatting must cover all four with tests.

## Build and Test

```bash
uv sync --group dev
uv run ruff check .
uv run pytest -q
uv build
```

CI (`.github/workflows/`) runs `ruff`, `pytest`, and the package build on every push and PR to `main`. All three must pass.

## Conventions

- Conventional Commits with allowed tags: `feat`, `fix`, `perf`, `refactor`, `build`, `chore`, `ci`, `docs`, `style`, `test`, `revert`. `feat` → minor bump, `fix`/`perf` → patch. Releases are automated by `python-semantic-release`; do not hand-edit `version` in pyproject.toml or `CHANGELOG.md`.
- Ruff: line length 100, target `py310`, rules E/F/I/B. No suppressions without a comment justifying them.
- Tests use `pytest` only — no shared base classes, no `conftest.py` fixtures unless reused 3+ times. Test through the public API; do not mock baretools internals.

## Verification Discipline

LLM provider request/response shapes change. Before modifying provider-specific code (schema output, parsing, formatting), fetch the current vendor docs rather than relying on memory. See the workspace conventions file for the per-provider field checklist.

## Pitfalls

- `tool` decorator stores metadata as `__baretools_*` attributes on the function — preserve them when wrapping.
- `ToolRegistry.register` walks the caller's frame to resolve forward references in type hints. Calling it from unusual contexts (exec, eval, compiled code) may break hint resolution.
- LLM-supplied tool arguments are untrusted input. Use `ToolRegistry.execute` for validation/coercion; never hand raw arguments to tool functions.
- Do not log tool argument values above DEBUG — they may contain user data.
