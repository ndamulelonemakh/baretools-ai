# CHANGELOG

<!-- version list -->

## v0.3.0 (Unreleased)

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

## v0.2.0 (Unreleased)

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
