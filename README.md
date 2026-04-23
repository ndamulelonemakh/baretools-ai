# [WIP] Baretools AI

**The un-framework for AI Agents**

Just the plumbing. You bring the intelligence.

---

## The Problem

Modern agent frameworks try to do too much:

- **Opinionated orchestration** that doesn't fit your use case
- **Black-box prompt management** hiding your most important logic
- **Over-abstracted context handling** fighting you at every turn
- **Rigid execution flows** that assume one-size-fits-all
- **Cognitive overhead** learning framework patterns instead of building

You end up fighting the framework more than building your application.

**The truth**: With modern LLMs, your prompts *are* your application logic. You need full control over context, orchestration, and state management. These vary wildly by use case and shouldn't be abstracted away.

---

## The Solution

Baretools AI does **one thing well**: handle the boring, error-prone plumbing between your Python functions and LLM tool calls.

**What Baretools provides:**
- Function → LLM tool schema conversion
- Tool call parsing and validation
- Function execution routing
- Result formatting back to LLM format

**What you control:**
- Your prompts and system messages
- Context window engineering
- Orchestration logic (retries, loops, conditionals)
- State management
- Error handling and guardrails

No magic. No opinions. No fighting the framework.

---

## Visual Overview

```mermaid
flowchart TD
    U[User prompt] --> LLM[LLM response]
    LLM -->|tool_calls| P[baretools: parse_tool_calls]
    P --> R[ToolRegistry.execute / execute_async]
    R --> T1[Your Python tool functions]
    R --> F[format_tool_results]
    F --> M[Append tool messages]
    M --> LLM

    subgraph baretools["Baretools (plumbing only)"]
      P
      R
      F
    end

    subgraph you["You Control"]
      U
      LLM
      T1
      M
    end
```

Baretools handles schema conversion + tool execution mechanics, while you keep full control over prompts, orchestration loops, retries policy, and state management.

---

## Technical Specification

### Core Components

#### 1. Tool Registration
```python
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # Your implementation
    pass
```

**Capabilities:**
- Automatic schema generation from type hints and docstrings
- Support for Pydantic models as parameters
- Optional manual schema override
- Validation of function signatures

#### 2. Schema Generation
```python
tools = ToolRegistry()
tools.register(search_web)

# Get LLM-compatible schema
schemas = tools.get_schemas("openai")  # or anthropic/gemini/json_schema
```

**Output formats:**
- OpenAI function calling format
- Anthropic tool use format
- Gemini function declaration format
- Generic JSON schema

#### 3. Tool Execution
```python
# Parse LLM response
tool_calls = parse_tool_calls(llm_response)

# Execute tools
results = tools.execute(tool_calls)

# Format for next LLM call
formatted_results = format_tool_results(results)
```

**Features:**
- Parallel execution support
- Error capture without crashing
- Execution metadata (timing, success/failure)
- Type validation on inputs

#### 4. Result Handling
```python
{
    "tool_call_id": "call_abc123",
    "output": "Search results...",
    "error": None,
    "execution_time_ms": 234
}
```

---

## Roadmap

### ✅ Phase 1: Core (v0.1.0)
- [x] Tool decorator and registration
- [x] Schema generation (OpenAI format)
- [x] Basic tool execution
- [x] Error handling

### 🎯 Phase 2: Multi-Provider (v0.2.0)
- [x] Anthropic format support
- [x] Google/Gemini format support
- [x] Provider-agnostic schema conversion

### 🔮 Phase 3: Developer Experience (v0.3.0)
- [x] Async tool execution
- [x] Built-in logging/tracing hooks
- [ ] Pydantic model support
- [ ] Streaming tool results

### 💡 Phase 4: Advanced (v0.4.0)
- [ ] Tool composition (tools calling tools)
- [ ] Execution sandboxing options
- [ ] Rate limiting helpers
- [ ] Cost tracking utilities

---


## CI

GitHub Actions now uses `uv` to run `ruff check .` and `pytest -q` on pushes and pull requests targeting `main`.

---

## Current Implementation Status

This repository now includes a minimal `v0.1.0` implementation in `src/baretools` with:
- `@tool` decorator metadata
- `ToolRegistry.register()` schema generation
- `ToolRegistry.execute()` with optional parallel execution
- `parse_tool_calls()` and `format_tool_results()` helpers

Run locally:
```bash
uv sync --group dev
uv run pytest -q
```

---

## Installation

```bash
pip install baretools-ai
```

---

## Quick Start

```python
from baretools import tool, ToolRegistry
from openai import OpenAI

# 1. Define your tools
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Your implementation
    return f"Sunny, 72°F in {location}"

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)  # In production, use safe eval!

# 2. Register tools
tools = ToolRegistry()
tools.register(get_weather)
tools.register(calculate)

# 3. Use with your LLM (you control everything)
client = OpenAI()
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools.get_schemas()  # ← Baretools helps here
    )

    message = response.choices[0].message

    # No tool calls? We're done
    if not message.tool_calls:
        print(message.content)
        break

    # Execute tools
    results = tools.execute(message.tool_calls)  # ← Baretools helps here

    # YOU decide how to handle results
    messages.append(message)
    messages.append({
        "role": "tool",
        "tool_call_id": results[0]["tool_call_id"],
        "content": results[0]["output"]
    })
```

---

## Usage Examples

### Custom Orchestration
```python
# You control the loop, retries, and logic
max_iterations = 5
for i in range(max_iterations):
    response = llm.chat(messages, tools=tools.get_schemas())

    if not response.tool_calls:
        break

    results = tools.execute(response.tool_calls)

    # Your custom logic
    if any(r["error"] for r in results):
        # Handle errors your way
        messages.append({"role": "user", "content": "That failed, try differently"})
    else:
        # Success - add results
        for result in results:
            messages.append(format_tool_result(result))
```

### Async Tools
```python
@tool
async def fetch_customer(customer_id: str) -> dict:
    return await api_client.get_customer(customer_id)

# Works in sync loops
sync_results = tools.execute(tool_calls)

# Works in async loops
async_results = await tools.execute_async(tool_calls, parallel=True, retries=1)
```

### Parallel Calls, Logging, and Retries
```python
import logging
from baretools import ToolRegistry

events = []
registry = ToolRegistry(logger=logging.getLogger("baretools"), on_event=events.append)

results = registry.execute(
    tool_calls,
    parallel=True,
    max_workers=8,
    retries=2,
    retry_delay_seconds=0.2,
)
```

`results` includes `attempts` metadata for each call, and `on_event` receives structured tool execution events (attempt/retry/failure).

### Parallel Execution
```python
# Baretools executes in parallel by default
tool_calls = [
    {"name": "search_web", "arguments": {"query": "AI news"}},
    {"name": "search_web", "arguments": {"query": "Python tips"}},
    {"name": "get_weather", "arguments": {"location": "NYC"}}
]

results = tools.execute(tool_calls, parallel=True)
```

### Error Handling
```python
results = tools.execute(tool_calls)

for result in results:
    if result["error"]:
        print(f"Tool {result['tool_name']} failed: {result['error']}")
        # You decide: retry? skip? fail? continue?
    else:
        print(f"Success: {result['output']}")
```

---

## Philosophy

**We believe:**
- Developers should control their prompts, not frameworks
- Context engineering is application-specific
- Orchestration logic belongs in your code
- Less abstraction = more clarity
- The best framework is one you barely notice

**We don't:**
- Manage your conversation state
- Decide when to call tools
- Handle your retries or error recovery
- Abstract away the LLM interaction
- Try to be "smart" on your behalf

---

## Contributing

Baretools AI is intentionally minimal. Before proposing features, ask:
- Does this belong in every tool-calling application?
- Can developers easily implement this themselves?
- Does this add opinions about orchestration or prompting?

If the answer to any is "no," it probably doesn't belong in Baretools.

---

## License

MIT

---

## Why "Baretools"?

**Bare** (adj): without addition or embellishment; simple and basic.

That's us. 🛠️
