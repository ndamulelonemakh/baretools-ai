---
title: Getting Started — Baretools AI
---

# Getting Started

[← Home](.) &nbsp;·&nbsp; [Why Baretools?](why-baretools) &nbsp;·&nbsp; [Concepts](concepts) &nbsp;·&nbsp; [Providers](providers) &nbsp;·&nbsp; [API Reference](api-reference)

---

## Install

```bash
pip install baretools-ai
```

No other packages are installed. Verify:

```python
import baretools
print(baretools.__version__)  # e.g. "0.3.0"
```

## Step 1 — Define a Tool

Decorate any plain Python function with `@tool`:

```python
from baretools import tool

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate Body Mass Index."""
    return round(weight_kg / height_m ** 2, 1)
```

Type hints become the JSON schema. The docstring becomes the tool description sent to the LLM.

## Step 2 — Register It

```python
from baretools import ToolRegistry

tools = ToolRegistry()
tools.register(calculate_bmi)
```

## Step 3 — Get the Schema for Your Provider

```python
openai_schemas    = tools.get_schemas("openai")     # OpenAI / OpenAI-compatible
anthropic_schemas = tools.get_schemas("anthropic")  # Claude
gemini_schemas    = tools.get_schemas("gemini")     # Gemini
```

Pass the result directly to your provider SDK's `tools=` parameter.

## Step 4 — Parse, Execute, Format

After the model responds with tool calls:

```python
from baretools import parse_tool_calls, format_tool_results

tool_calls = parse_tool_calls(llm_response, provider="openai")  # normalise
results    = tools.execute(tool_calls)                          # run your functions
messages   = format_tool_results(results, provider="openai")    # ready to append
```

## Full Minimal Example (OpenAI)

```python
from openai import OpenAI
from baretools import tool, ToolRegistry, parse_tool_calls, format_tool_results

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate Body Mass Index."""
    return round(weight_kg / height_m ** 2, 1)

tools = ToolRegistry()
tools.register(calculate_bmi)

client   = OpenAI()
messages = [{"role": "user", "content": "My weight is 80 kg and height 1.75 m. What's my BMI?"}]

for _ in range(6):
    response   = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools.get_schemas("openai"),
    )
    message    = response.choices[0].message
    tool_calls = parse_tool_calls(message, "openai")

    if not tool_calls:
        print(message.content)
        break

    results = tools.execute(tool_calls, parallel=True)
    messages.append(message.model_dump(exclude_none=True))
    messages.extend(format_tool_results(results, "openai"))
```

## Next Steps

- [Core concepts](concepts) — `ToolCall`, `ToolResult`, execution options, streaming
- [Provider integrations](providers) — Anthropic and Gemini patterns with copy-paste loops
- [API reference](api-reference) — complete interface documentation
