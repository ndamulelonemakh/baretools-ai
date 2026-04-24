# CHANGELOG

<!-- version list -->

## v0.4.0 (2026-04-24)

### Continuous Integration

- **release**: Allow manual workflow_dispatch
  ([`85a6ee6`](https://github.com/ndamulelonemakh/baretools-ai/commit/85a6ee6fa3ee17f157a04d3ed5f99613a05de1db))

### Documentation

- **readme**: Clarify contributing guidelines and trim redundant CI note
  ([`651c78b`](https://github.com/ndamulelonemakh/baretools-ai/commit/651c78bef5215b116e806047003b89713caabae7))

### Features

- **core**: Add per-call timeout and strict OpenAI tool-call parsing
  ([`4425426`](https://github.com/ndamulelonemakh/baretools-ai/commit/442542616b4c26cc7b2f5dee81bb38f91b5a7e16))


## v0.3.0 (Current branch target)

- Zero runtime dependencies: baretools ships with no third-party packages on
  the install path. Standard-library `dataclasses` are supported as tool
  parameter types out-of-the-box; `pydantic` is opt-in via the
  `baretools-ai[pydantic]` extra.
- Dataclass parameters: tools may declare a `@dataclass` type as a parameter.
  The dataclass fields are reflected into the JSON Schema and dict arguments
  are coerced into the dataclass instance before the tool is invoked.
- Pydantic model parameters: tools may declare a `pydantic.BaseModel` subclass
  as a parameter type. The model's JSON Schema is embedded in the tool schema,
  and dict arguments are validated/coerced via `model_validate()` before the
  tool is invoked. Pydantic remains an optional dependency — it is only
  required if you use this feature.
- Add `ToolRegistry.execute_stream()` and `execute_stream_async()` that yield
  `ToolResult` values as each call finishes (completion order when parallel,
  input order otherwise).
- Add live OpenAI, Anthropic, and Gemini example agent loops under `examples/`
  showing how baretools turns provider-specific tool calls into the same
  register → schema → execute → return-results cycle.
- Add optional W&B Weave tracing to the provider examples so developers can
  inspect agent-loop and tool-call traces without changing the library itself.
- `parse_tool_calls(message, provider="openai" | "anthropic" | "gemini")` and
  `format_tool_results(results, provider=...)` now cover all three providers.
  Returns plain dicts (no SDK imports in core), so callers can wrap them in
  `types.Part.from_function_response(...)` or an Anthropic user message.

## v0.2.0

- Add multi-provider tool schema support via `ToolRegistry.get_schemas(provider=...)`.
  Supported providers: `openai` (default), `anthropic`, `gemini`, `json_schema`.
- Gemini output uses the `functionDeclarations` shape and conforms to the supported
  OpenAPI subset (no `additionalProperties`; `parameters` and empty `required` are
  omitted when not needed).
- `get_schemas()` returns defensive copies so callers can mutate results without
  affecting the registry.
- Add opt-in `strict=True` for the `openai` provider, which marks every property
  as required, unions optional parameters with `null`, and emits `"strict": true`
  per OpenAI's structured-outputs requirements.
- **Breaking:** `RegisteredTool.schema` renamed to `RegisteredTool.parameters` and
  now stores a canonical JSON Schema rather than the OpenAI-wrapped shape.
  Provider-specific shapes are produced lazily by `get_schemas()`.

## v0.1.0 (2026-04-22)

- Initial Release
