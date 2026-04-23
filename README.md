# Baretools AI

**The un-framework for AI Agents** — just the plumbing, no supply-chain baggage.

[![PyPI version](https://img.shields.io/pypi/v/baretools-ai.svg)](https://pypi.org/project/baretools-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/baretools-ai.svg)](https://pypi.org/project/baretools-ai/)
[![CI](https://github.com/ndamulelonemakh/baretools-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/ndamulelonemakh/baretools-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Status: Alpha.
[**Documentation**](https://ndamulelonemakh.github.io/baretools-ai/) &nbsp;·&nbsp; [API Reference](https://ndamulelonemakh.github.io/baretools-ai/api-reference) &nbsp;·&nbsp; [Why Baretools?](https://ndamulelonemakh.github.io/baretools-ai/why-baretools) &nbsp;·&nbsp; [Changelog](https://ndamulelonemakh.github.io/baretools-ai/changelog)

---

## Why

Modern agent frameworks own your prompts, your orchestration, and your state. That trade-off is fine for demos and POCs, but in production it costs you the control needed to build your own [Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness). You also end up knowing the framework better than the underlying engineering — which is exactly backwards.

Baretools handles only the mechanical glue between your Python functions and the LLM:

- Function → provider tool schema (OpenAI, Anthropic, Gemini, generic JSON Schema)
- Parsing tool calls out of provider responses
- Validating and executing those calls (sync, async, parallel, streaming)
- Formatting results back into provider-shaped messages

Everything else — prompts, loops, retries, memory, guardrails — stays in your code.

## Install

```bash
pip install baretools-ai
```

### Pre-requisite:

- Python >=3.10x
- Optional Pydantic support: `pip install "baretools-ai[pydantic]"`.

## Quickstart

```python
from baretools import tool, ToolRegistry, parse_tool_calls, format_tool_results
from openai import OpenAI

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Sunny, 72°F in {location}"

tools = ToolRegistry()
tools.register(get_weather)

client = OpenAI()
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

max_iterations = 5 # Can be very high in real world agents
iteration = 0
while iteration < max_iterations:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools.get_schemas("openai"),
    )
    iteration += 1
    message = response.choices[0].message
    messages.append(message)

    tool_calls = parse_tool_calls(message, "openai")
    if not tool_calls:
        print("Final Response:", message.content)
        break

    results = tools.execute(tool_calls)
    messages.extend(format_tool_results(results, "openai"))
```

You write the loop. Baretools handles the schema, parsing, execution, and formatting on each side.

## Runnable Examples

Working agents for each provider are in [examples/](examples/):

```bash
OPENAI_API_KEY=...    uv run python examples/openai_agent.py
ANTHROPIC_API_KEY=... uv run python examples/anthropic_agent.py
GOOGLE_API_KEY=...    uv run python examples/gemini_agent.py
```

## Features

- **Zero runtime dependencies** — stdlib only; no transitive supply chain to audit
- **Multi-provider schemas** — `tools.get_schemas("openai" | "anthropic" | "gemini" | "json_schema")`
- **Sync, async, streaming** — `execute`, `execute_async`, `execute_stream`, `execute_stream_async`
- **Parallel tool execution** — pass `parallel=True` with `max_workers` (sync) or `max_concurrency` (async) to fan out independent calls
- **Tool call hooks** — `before_tool` and `after_tool` callbacks let you plug in guardrails, redaction, auditing, or tracing around every call
- **Retries with structured events** — pass `on_event=...` to observe attempts/retries/failures
- **Type-driven validation** — `dataclasses` work out of the box; `pydantic` BaseModels supported when installed
- **Provider-native parsing/formatting** — `parse_tool_calls()` and `format_tool_results()` for all four providers

See the [docs site](https://ndamulelonemakh.github.io/baretools-ai/) for the full API reference, advanced patterns, and design notes.

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run pytest -q
```

CI runs `ruff`, `pytest`, and a package build on every push and PR to `main`.

## Contributing

Baretools is intentionally minimal. Before proposing a feature, ask whether it belongs in *every* tool-calling app and whether a developer could implement it in a few lines themselves. If either answer is "no," it probably doesn't belong here. See [CONTRIBUTING](https://github.com/ndamulelonemakh/baretools-ai/blob/main/CONTRIBUTING.md) if present, otherwise open an issue first.

## License

MIT — see [LICENSE](LICENSE).
