---
title: Roadmap
---

# Roadmap

Tracks shipped milestones and the loosely-planned next steps. Concrete in-progress work lives in [GitHub issues](https://github.com/ndamulelonemakh/baretools-ai/issues); release-by-release detail is in the [Changelog](changelog).

## Shipped

### v0.1.0 — Core
- `@tool` decorator and registration
- Schema generation (OpenAI format)
- Synchronous tool execution
- Error capture without crashing the loop

### v0.2.0 — Multi-Provider
- Anthropic tool-use schemas
- Google / Gemini function declarations
- Provider-agnostic schema conversion (`get_schemas("openai" | "anthropic" | "gemini" | "json_schema")`)

### v0.3.0 — Developer Experience
- Async tool execution (`execute_async`)
- Built-in logging and `on_event` tracing hooks
- Optional Pydantic model support
- Streaming results (`execute_stream`, `execute_stream_async`)
- Runnable provider examples in [`examples/`](https://github.com/ndamulelonemakh/baretools-ai/tree/main/examples)

## Considering

- **Tool composition** — tools calling tools, programmatic tool search and dispatch
- **Execution sandboxing options** — pluggable isolation for untrusted tool bodies
- **Cost tracking utilities** — token / call accounting that stays out of the request path
- **Rate limiting helpers** — small primitives, not a scheduler

If one of these matters to your use case, open an issue describing the concrete problem you hit. Real use cases beat speculative API design every time.
