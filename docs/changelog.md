---
title: Changelog — Baretools AI
---

# Changelog

[← Home](.) &nbsp;·&nbsp; [Get Started](getting-started) &nbsp;·&nbsp; [API Reference](api-reference)

---

## v0.3.0 (Current branch target)

- **Zero runtime dependencies**: `pip install baretools-ai` installs nothing but the package itself. Standard-library `dataclasses` work as tool parameter types out-of-the-box; Pydantic is opt-in via `baretools-ai[pydantic]`.
- **Dataclass parameters**: `@dataclass` fields are reflected into the JSON Schema and dict arguments from the LLM are coerced into the dataclass instance before invocation.
- **Pydantic model parameters**: `BaseModel` subclasses are supported as parameter types. Schema comes from `model_json_schema()`; coercion uses `model_validate()`.
- **Streaming**: `execute_stream()` and `execute_stream_async()` yield `ToolResult` values as each call completes.
- **Multi-provider helpers**: `parse_tool_calls(response, provider=...)` and `format_tool_results(results, provider=...)` for OpenAI, Anthropic, and Gemini.
- **Live provider examples**: `examples/openai_agent.py`, `examples/anthropic_agent.py`, `examples/gemini_agent.py` — tested BMI agent loops for each provider.
- **Optional W&B Weave tracing**: gated on `WEAVE_PROJECT` env var; zero overhead when unset.
- **Version from metadata**: `baretools.__version__` reads from `importlib.metadata`; `pyproject.toml` is the single source of truth.

## v0.2.0

- Multi-provider schema support via `ToolRegistry.get_schemas(provider=...)`.
- Gemini output uses the `functionDeclarations` shape with OpenAPI subset compliance.
- `get_schemas()` returns defensive copies.
- `strict=True` for OpenAI structured-outputs mode.
- **Breaking**: `RegisteredTool.schema` renamed to `RegisteredTool.parameters`.

## v0.1.0

- `@tool` decorator and `ToolRegistry`
- Schema generation (OpenAI format)
- Synchronous tool execution with optional parallel mode
- Error capture — exceptions never propagate out of `execute`
- `ToolCall` and `ToolResult` TypedDicts

---

_Full history: [CHANGELOG.md on GitHub](https://github.com/ndamulelonemakh/baretools-ai/blob/main/CHANGELOG.md)_
