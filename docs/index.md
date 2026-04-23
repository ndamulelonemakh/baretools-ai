---
title: Baretools AI — The un-framework for AI Agents
---

# Baretools AI

**The un-framework for AI Agents** — just the plumbing, no supply-chain baggage.

**[Get Started](getting-started)** &nbsp;·&nbsp; **[Why Baretools?](why-baretools)** &nbsp;·&nbsp; **[Concepts](concepts)** &nbsp;·&nbsp; **[Provider Integrations](providers)** &nbsp;·&nbsp; **[API Reference](api-reference)** &nbsp;·&nbsp; **[Changelog](changelog)**

---

`pip install baretools-ai` installs **one package and zero runtime dependencies**. No Pydantic required. No httpx. No hidden transitive packages that drift, get compromised, or bloat your Docker image.

## What Baretools Does

Baretools handles the mechanical plumbing between your Python functions and LLM tool calls:

- **Function → schema**: converts your typed Python function into the exact JSON shape each provider expects
- **Response → calls**: parses raw provider responses into normalised `ToolCall` dicts
- **Calls → results**: executes your functions, capturing errors without crashing
- **Results → messages**: formats outputs back into the provider-native message shape

Everything else — prompts, context, retries, state, orchestration — stays in your code.

## Quick Start

```python
from baretools import tool, ToolRegistry, parse_tool_calls, format_tool_results

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Sunny, 72°F in {location}"

tools = ToolRegistry()
tools.register(get_weather)

# Same pattern for OpenAI, Anthropic, or Gemini — only provider= changes
schemas    = tools.get_schemas("openai")
tool_calls = parse_tool_calls(llm_response, provider="openai")
results    = tools.execute(tool_calls)
messages   = format_tool_results(results, provider="openai")
```

## Install

```bash
pip install baretools-ai
```

Optional Pydantic support (only needed if your tools accept `BaseModel` parameters):

```bash
pip install "baretools-ai[pydantic]"
```

## Supported Providers

| Provider | Schema format | `parse_tool_calls` | `format_tool_results` |
|---|---|---|---|
| OpenAI | `{"type":"function","function":{...}}` | ✓ | ✓ |
| Anthropic | `{"name":...,"input_schema":{...}}` | ✓ | ✓ |
| Gemini | `{"functionDeclarations":[...]}` | ✓ | ✓ |

---

_[View source on GitHub](https://github.com/ndamulelonemakh/baretools-ai)_
