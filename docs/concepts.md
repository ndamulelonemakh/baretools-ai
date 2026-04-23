---
title: Core Concepts — Baretools AI
---

# Core Concepts

[← Home](.) &nbsp;·&nbsp; [Get Started](getting-started) &nbsp;·&nbsp; [Why Baretools?](why-baretools) &nbsp;·&nbsp; [Providers](providers) &nbsp;·&nbsp; [API Reference](api-reference)

---

## The `@tool` Decorator

```python
from baretools import tool

@tool
def search(query: str, max_results: int = 5) -> list[str]:
    """Search for documents matching a query."""
    ...
```

`@tool` captures the function's signature, type hints, and docstring. It does not wrap or modify the function's runtime behaviour — calling `search(...)` works exactly as before.

### Type Mapping

| Python type | JSON Schema type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list[T]` | `{"type":"array","items":{...}}` |
| `dict` | `{"type":"object"}` |
| `@dataclass` | `{"type":"object","properties":{...}}` |
| `BaseModel` subclass | `model.model_json_schema()` _(requires pydantic)_ |

Parameters without defaults are added to `required`. Parameters with defaults are optional.

### Structured Parameters (Zero Extra Deps)

Standard-library `@dataclass` types work out-of-the-box:

```python
from dataclasses import dataclass
from baretools import tool

@dataclass
class Address:
    street: str
    city: str
    zip: str

@tool
def create_user(name: str, address: Address) -> dict:
    return {"name": name, "city": address.city}
```

When the LLM sends `{"address": {"street": "...", "city": "...", "zip": "..."}}`, baretools coerces the dict into an `Address` instance before invoking the function. No Pydantic required.

### Async Tools

`@tool` works on `async def` functions transparently. Use `execute_async` / `execute_stream_async` to drive them.

---

## `ToolRegistry`

```python
from baretools import ToolRegistry

registry = ToolRegistry(
    logger=None,    # logging.Logger | None
    on_event=None,  # Callable[[ToolEvent], None] | None
)
registry.register(my_tool)
```

### Schema Output

```python
registry.get_schemas("openai")      # [{"type":"function","function":{...}}]
registry.get_schemas("anthropic")   # [{"name":...,"input_schema":{...}}]
registry.get_schemas("gemini")      # [{"functionDeclarations":[...]}]
registry.get_schemas("json_schema") # [{"name":...,"parameters":{...}}]

# OpenAI structured outputs mode
registry.get_schemas("openai", strict=True)
```

### Execution Options

```python
results = registry.execute(
    tool_calls,
    parallel=True,           # ThreadPoolExecutor
    max_workers=4,
    retries=2,               # retry on exception
    retry_delay_seconds=0.5,
)
```

### Streaming

```python
# Sync — yields in completion order when parallel=True
for result in registry.execute_stream(tool_calls, parallel=True, max_workers=4):
    handle(result)

# Async equivalent
async for result in registry.execute_stream_async(tool_calls, parallel=True, max_concurrency=4):
    await handle(result)
```

---

## `parse_tool_calls`

```python
from baretools import parse_tool_calls

tool_calls = parse_tool_calls(llm_response, provider="openai")
# → list[ToolCall]
```

Converts the provider's raw response into a list of normalised `ToolCall` TypedDicts:

```python
{"id": "call_abc123", "name": "search", "arguments": {"query": "AI news"}}
```

Returns an empty list when the response contains no tool calls. Supported providers: `"openai"`, `"anthropic"`, `"gemini"`.

---

## `format_tool_results`

```python
from baretools import format_tool_results

messages = format_tool_results(results, provider="openai")
```

Converts `list[ToolResult]` into provider-native message shapes ready to append to your conversation history.

| Provider | Output shape |
|---|---|
| `"openai"` | `[{"role":"tool","tool_call_id":...,"content":...}]` — `messages.extend(...)` |
| `"anthropic"` | `[{"type":"tool_result","tool_use_id":...,"content":...}]` — wrap in a `user` message |
| `"gemini"` | `[{"name":...,"response":{...}}]` — `Part.from_function_response(**item)` for each |

---

## Public TypedDicts

All public types are plain `TypedDict`s — no class hierarchies, no serialization magic, no runtime overhead.

```python
class ToolCall(TypedDict, total=False):
    id: str | None
    name: str | None
    arguments: dict[str, Any] | str

class ToolResult(TypedDict):
    tool_call_id: str | None
    tool_name: str | None
    output: Any
    error: str | None       # None on success
    attempts: int           # 1 if no retries configured
    execution_time_ms: int

class ToolEvent(TypedDict, total=False):
    event: Literal["tool_attempt", "tool_retry", "tool_failed"]
    tool_call_id: str | None
    tool_name: str | None
    attempt: int
    error: str
```

`ProviderToolResult` is the union type returned by `format_tool_results`. Its internal shape is provider-specific; pass items directly to your SDK without inspecting them.
