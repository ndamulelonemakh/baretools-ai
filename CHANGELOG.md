# CHANGELOG

<!-- version list -->

## v0.4.4 (2026-04-24)

### Bug Fixes

- **readme**: Bypass stale camo cache for pypi+release badges
  ([`97c45a9`](https://github.com/ndamulelonemakh/baretools-ai/commit/97c45a9084e4edccc918faeec4ee5348ce6ee111))


## v0.4.3 (2026-04-24)

### Bug Fixes

- **readme**: Replace ci+python badges with release pipeline badge
  ([`b1ca4aa`](https://github.com/ndamulelonemakh/baretools-ai/commit/b1ca4aa7c57e05a84b32f95b49f031cb6daee5f4))


## v0.4.2 (2026-04-24)

### Bug Fixes

- **ci**: Bump pinned action versions to latest releases
  ([`e1237a6`](https://github.com/ndamulelonemakh/baretools-ai/commit/e1237a6afc4e10840c7b0670bd4cc6f2c55ca0c0))

### Chores

- **lock**: Sync uv.lock to 0.4.1
  ([`38cb66d`](https://github.com/ndamulelonemakh/baretools-ai/commit/38cb66ddd185e8a6d53bd1d0451af54e084e7b09))

### Documentation

- **readme**: Update description to emphasize focus on AI Engineers
  ([`b8fe09a`](https://github.com/ndamulelonemakh/baretools-ai/commit/b8fe09a12ff665f2bb6fe1c0be034b41e69e2c68))

- **readme**: Use shields.io workflow badge for reliable pypi rendering
  ([`2500d26`](https://github.com/ndamulelonemakh/baretools-ai/commit/2500d261c0ec08b9519dba700e3c39e9da50651b))


## v0.4.1 (2026-04-24)

### Bug Fixes

- **release**: Skip when no version bump and guard gh release on dist artifacts
  ([`7ab08f8`](https://github.com/ndamulelonemakh/baretools-ai/commit/7ab08f80bc89b1f10e7bbee1c2eaad7378dc91c9))

### Continuous Integration

- **release**: Bind release job to pypi environment for trusted publishing
  ([`e4fc014`](https://github.com/ndamulelonemakh/baretools-ai/commit/e4fc0142f5f94d7bab0bb2d488125319ec13bf4b))

- **release**: Create github release before pypi publish so pypi failures don't skip it
  ([`f00dff7`](https://github.com/ndamulelonemakh/baretools-ai/commit/f00dff7a6bc11247faa04ed56b958350a4498523))


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
