---
description: "Use when writing, reviewing, or modifying Python source, tests, or pyproject.toml for baretools-ai. Covers zero-dependency constraint, ruff config, provider parity, TypedDict patterns, and conventional commits."
applyTo: ["src/**/*.py", "tests/**/*.py", "pyproject.toml"]
---
# baretools-ai Engineering Conventions

## Hard Constraints

- `dependencies = []` in pyproject.toml is a design invariant. Runtime deps must stay zero. Any new functionality must use stdlib only. Dev tools go in `[dependency-groups]`, never `[project.dependencies]`.
- Python 3.10–3.13 support is required. No syntax or stdlib APIs beyond 3.10 unless guarded or polyfilled.
- Every `.py` file starts with `from __future__ import annotations`.

## Code Style

- Line length: 100. Target: `py310`. Ruff rules: E, F, I, B — no suppressions without a comment explaining why.
- Use `TypedDict` for structured dict types passed across module boundaries. Use `dataclass(frozen=True)` for internal value objects.
- `from __future__ import annotations` enables PEP 563 deferred evaluation — use bare string annotations only when you have no other option.
- Prefer `tuple[str, ...]` over `Tuple[str, ...]`, `dict[str, Any]` over `Dict[str, Any]` (PEP 585, safe with the `__future__` import).

## Provider Parity

The four supported providers are `openai`, `anthropic`, `gemini`, `json_schema` (see `_SUPPORTED_PROVIDERS`). Any change that affects schema generation, tool result formatting, or tool call parsing must be implemented and tested for all four. Never add a provider-specific fast path without updating the others.

## Verification Discipline (LLM Provider APIs)

LLM provider APIs change frequently. Before writing any code that targets a provider's request/response shape:

1. Fetch the current API reference using available tools (context7, tavily, or direct fetch).
2. Check the provider's Python SDK changelog for breaking changes since the last release.
3. Confirm the exact field names, types, and required vs optional status — do not rely on training data alone.

Specifically verify:
- OpenAI: `tools[].function.parameters` schema shape, `tool_choice` options, `tool_calls` response field
- Anthropic: `tools[].input_schema` shape, `tool_use` content block structure
- Gemini: `tools[].functionDeclarations` shape, `functionCall` / `functionResponse` part types

## Testing

- Use `pytest`. No test framework abstractions over pytest — no custom base classes, no shared fixtures in `conftest.py` unless they provide genuine reuse across 3+ tests.
- Test function naming: `test_<behaviour>` or `test_<behaviour>_when_<condition>`.
- Do not mock internals of `baretools` itself in tests. Test through the public API (`tool`, `ToolRegistry`, `parse_tool_calls`, `format_tool_results`).
- Each test must be independent — no shared mutable state between tests.

## Conventional Commits

Allowed prefixes: `feat`, `fix`, `perf`, `refactor`, `build`, `chore`, `ci`, `docs`, `style`, `test`, `revert`.
- `feat` → minor version bump
- `fix`, `perf` → patch version bump
- Breaking changes: append `!` after the prefix (e.g., `feat!:`) and include a `BREAKING CHANGE:` footer

## Safety at Boundaries

- LLM-supplied tool arguments arrive as raw JSON and must be validated before use. `ToolRegistry.execute` handles coercion — do not bypass it by calling tool functions directly with raw LLM output.
- When `max_argument_payload_chars` is set, enforce it before deserialization to prevent payload amplification.
- Do not log tool argument values at INFO level or above — they may contain sensitive user data. Use DEBUG only.
