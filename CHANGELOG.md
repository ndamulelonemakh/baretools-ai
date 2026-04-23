# CHANGELOG

<!-- version list -->

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
