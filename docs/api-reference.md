---
title: API Reference — Baretools AI
---

# API Reference

[← Home](.) &nbsp;·&nbsp; [Get Started](getting-started) &nbsp;·&nbsp; [Concepts](concepts) &nbsp;·&nbsp; [Providers](providers)

---

## `tool`

```python
from baretools import tool

@tool
def my_function(a: int, b: str = "default") -> str:
    """Tool description for the LLM."""
    ...
```

Marks a function as a tool. Works with both sync and `async def` functions. Can also be called directly: `tool(my_fn)`.

---

## `ToolRegistry`

```python
from baretools import ToolRegistry

registry = ToolRegistry(
    logger=None,    # logging.Logger | None
    on_event=None,  # Callable[[ToolEvent], None] | None
)
```

### `register(fn)`

Register a `@tool`-decorated function.

---

### `get_schemas(provider="openai", strict=False) → list`

Return provider-native schemas for all registered tools.

| `provider` | Return shape |
|---|---|
| `"openai"` | `[{"type":"function","function":{"name":...,"parameters":...}}]` |
| `"anthropic"` | `[{"name":...,"description":...,"input_schema":...}]` |
| `"gemini"` | `[{"functionDeclarations":[...]}]` |
| `"json_schema"` | `[{"name":...,"parameters":...}]` |

`strict=True` applies only to `"openai"`. Returns defensive copies — mutations do not affect the registry.

---

### `execute(tool_calls, *, parallel=False, max_workers=4, retries=0, retry_delay_seconds=0.0) → list[ToolResult]`

Execute a list of `ToolCall`s synchronously. Errors are captured in `ToolResult.error`; exceptions never propagate out of `execute`.

| Parameter | Default | Description |
|---|---|---|
| `parallel` | `False` | Run calls concurrently in a `ThreadPoolExecutor` |
| `max_workers` | `4` | Thread pool size when `parallel=True` |
| `retries` | `0` | Additional attempts on exception (0 = no retries) |
| `retry_delay_seconds` | `0.0` | Sleep between retries |

---

### `execute_async(tool_calls, *, parallel=False, max_concurrency=4, retries=0, retry_delay_seconds=0.0) → Coroutine[list[ToolResult]]`

Async variant. Drives `async def` tools natively; wraps sync tools in a thread executor.

---

### `execute_stream(tool_calls, *, parallel=False, max_workers=4, retries=0, retry_delay_seconds=0.0) → Iterator[ToolResult]`

Sync generator. Yields one `ToolResult` per completed call. With `parallel=True`, yields in completion order (fastest first). Serial mode preserves input order.

---

### `execute_stream_async(tool_calls, *, parallel=False, max_concurrency=4, retries=0, retry_delay_seconds=0.0) → AsyncIterator[ToolResult]`

Async generator equivalent of `execute_stream`.

---

## `parse_tool_calls`

```python
from baretools import parse_tool_calls

tool_calls: list[ToolCall] = parse_tool_calls(response, provider="openai")
```

Parse a provider response into normalised `ToolCall` TypedDicts. Returns an empty list when the response contains no tool calls.

| `provider` | Expected `response` type |
|---|---|
| `"openai"` | `openai.types.chat.ChatCompletionMessage` |
| `"anthropic"` | `anthropic.types.Message` |
| `"gemini"` | `google.genai.types.GenerateContentResponse` |

---

## `format_tool_results`

```python
from baretools import format_tool_results

messages: list[ProviderToolResult] = format_tool_results(results, provider="openai")
```

Format `list[ToolResult]` into provider-native message dicts.

| `provider` | Output | How to use |
|---|---|---|
| `"openai"` | `[{"role":"tool","tool_call_id":...,"content":...}]` | `messages.extend(...)` |
| `"anthropic"` | `[{"type":"tool_result","tool_use_id":...,"content":...}]` | `{"role":"user","content": messages}` |
| `"gemini"` | `[{"name":...,"response":{...}}]` | `Part.from_function_response(**item)` for each |

---

## TypedDicts

### `ToolCall`

```python
class ToolCall(TypedDict, total=False):
    id: str | None
    tool_call_id: str | None
    name: str | None
    arguments: dict[str, Any] | str
```

### `ToolResult`

```python
class ToolResult(TypedDict):
    tool_call_id: str | None
    tool_name: str | None
    output: Any
    error: str | None       # None on success; exception message on failure
    attempts: int           # 1 if no retries configured
    execution_time_ms: int
```

### `ToolEvent`

```python
class ToolEvent(TypedDict, total=False):
    event: Literal["tool_attempt", "tool_retry", "tool_failed"]
    tool_call_id: str | None
    tool_name: str | None
    attempt: int
    error: str
```

### `ProviderToolResult`

Union type returned by `format_tool_results`. Internal structure is provider-specific; pass items directly to your SDK.

---

## `__version__`

```python
import baretools
print(baretools.__version__)  # e.g. "0.3.0"
```

Read from package metadata at import time. Matches `version` in `pyproject.toml`.
