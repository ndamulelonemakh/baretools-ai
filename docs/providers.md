---
title: Provider Integrations — Baretools AI
---

# Provider Integrations

[← Home](.) &nbsp;·&nbsp; [Get Started](getting-started) &nbsp;·&nbsp; [Concepts](concepts) &nbsp;·&nbsp; [API Reference](api-reference)

---

The same three-step pattern applies across all providers: **get schemas → parse response → format results**. Only the `provider=` argument changes.

---

## OpenAI

```python
from openai import OpenAI
from baretools import tool, ToolRegistry, parse_tool_calls, format_tool_results

tools = ToolRegistry()
tools.register(my_fn)

client   = OpenAI()
messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

for _ in range(6):
    response   = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools.get_schemas("openai", strict=True),
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

`strict=True` sets every property as required, unions optional params with `null`, and adds `"strict": true` per OpenAI's structured-outputs spec.

---

## Anthropic

```python
import anthropic
from baretools import tool, ToolRegistry, parse_tool_calls, format_tool_results

tools = ToolRegistry()
tools.register(my_fn)

client   = anthropic.Anthropic()
messages = [{"role": "user", "content": "..."}]

for _ in range(6):
    response   = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system="...",
        tools=tools.get_schemas("anthropic"),
        messages=messages,
    )
    tool_calls = parse_tool_calls(response, "anthropic")

    if not tool_calls:
        text = "\n".join(b.text for b in response.content if b.type == "text")
        print(text)
        break

    results = tools.execute(tool_calls, parallel=True)
    messages.append({"role": "assistant", "content": response.model_dump(exclude_none=True)["content"]})
    messages.append({"role": "user", "content": format_tool_results(results, "anthropic")})
```

`format_tool_results` returns a list of `tool_result` content blocks. Wrap it directly in a `user` message — Anthropic requires tool results in a `user` turn.

---

## Gemini

```python
from google import genai
from google.genai import types
from baretools import tool, ToolRegistry, parse_tool_calls, format_tool_results

tools        = ToolRegistry()
tools.register(my_fn)

client       = genai.Client()
declarations = tools.get_schemas("gemini")[0]["functionDeclarations"]
config       = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=declarations)],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
contents = [types.Content(role="user", parts=[types.Part(text="...")])]

for _ in range(6):
    response   = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=config,
    )
    tool_calls = parse_tool_calls(response, "gemini")

    if not tool_calls:
        print(response.text)
        break

    results = tools.execute(tool_calls, parallel=True)
    contents.append(response.candidates[0].content)
    contents.append(types.Content(
        role="user",
        parts=[
            types.Part.from_function_response(**item)
            for item in format_tool_results(results, "gemini")
        ],
    ))
```

Disable `automatic_function_calling` so baretools controls execution. Each item from `format_tool_results` has `name` and `response` keys, which map directly to `Part.from_function_response(**item)`.

---

## Runnable Examples

The `examples/` directory contains complete, live-tested BMI-agent loops for each provider:

```bash
pip install openai
OPENAI_API_KEY=sk-... python examples/openai_agent.py

pip install anthropic
ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_agent.py

pip install google-genai
GOOGLE_API_KEY=... python examples/gemini_agent.py
```

Optional W&B Weave tracing (all three examples):

```bash
pip install weave wandb
WANDB_API_KEY=... WEAVE_PROJECT=baretools python examples/openai_agent.py
```
