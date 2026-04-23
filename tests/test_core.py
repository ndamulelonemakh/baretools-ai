from __future__ import annotations

import asyncio
from time import perf_counter, sleep

import pytest

from baretools import ToolRegistry, format_tool_results, tool


def test_schema_generation_and_execution() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        """Add two numbers"""
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "add"

    results = registry.execute(
        [
            {"id": "call1", "name": "add", "arguments": {"a": 3, "b": 2}},
        ]
    )

    assert results[0]["error"] is None
    assert results[0]["output"] == 5
    assert results[0]["attempts"] == 1


def test_schema_generation_for_multiple_providers() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        """Add two numbers"""
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    openai_schemas = registry.get_schemas("openai")
    anthropic_schemas = registry.get_schemas("anthropic")
    gemini_schemas = registry.get_schemas("gemini")
    json_schemas = registry.get_schemas("json_schema")

    assert openai_schemas[0]["type"] == "function"
    assert openai_schemas[0]["function"]["name"] == "add"

    assert anthropic_schemas[0]["name"] == "add"
    assert anthropic_schemas[0]["input_schema"]["type"] == "object"

    assert len(gemini_schemas) == 1
    assert gemini_schemas[0]["functionDeclarations"][0]["name"] == "add"
    assert (
        gemini_schemas[0]["functionDeclarations"][0]["parameters"]["properties"]["a"]["type"]
        == "integer"
    )

    assert json_schemas[0]["name"] == "add"
    assert json_schemas[0]["schema"]["required"] == ["a"]


def test_get_schemas_rejects_unknown_provider() -> None:
    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry()
    registry.register(ping)

    with pytest.raises(ValueError, match="Unsupported provider"):
        registry.get_schemas("unsupported")  # type: ignore[arg-type]


def test_get_schemas_rejects_unknown_provider_when_registry_is_empty() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unsupported provider"):
        registry.get_schemas("unsupported")  # type: ignore[arg-type]


def test_get_schemas_returns_empty_list_for_empty_registry() -> None:
    registry = ToolRegistry()

    assert registry.get_schemas("openai") == []
    assert registry.get_schemas("anthropic") == []
    assert registry.get_schemas("gemini") == []
    assert registry.get_schemas("json_schema") == []


def test_openai_strict_mode_marks_all_required_and_unions_optionals_with_null() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    schema = registry.get_schemas("openai", strict=True)[0]
    assert schema["function"]["strict"] is True
    params = schema["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert sorted(params["required"]) == ["a", "b"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == ["integer", "null"]


def test_openai_default_is_not_strict() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    schema = registry.get_schemas("openai")[0]
    assert "strict" not in schema["function"]
    assert schema["function"]["parameters"]["required"] == ["a"]


def test_strict_rejected_for_non_openai_providers() -> None:
    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry()
    registry.register(ping)

    for provider in ("anthropic", "gemini", "json_schema"):
        with pytest.raises(ValueError, match="strict=True"):
            registry.get_schemas(provider, strict=True)  # type: ignore[arg-type]


def test_get_schemas_returns_defensive_copies() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    openai_schemas = registry.get_schemas("openai")
    openai_schemas[0]["function"]["parameters"]["properties"]["a"]["type"] = "string"

    fresh_openai_schemas = registry.get_schemas("openai")
    assert fresh_openai_schemas[0]["function"]["parameters"]["properties"]["a"]["type"] == "integer"

    anthropic_schemas = registry.get_schemas("anthropic")
    anthropic_schemas[0]["input_schema"]["properties"]["a"]["type"] = "string"

    fresh_anthropic_schemas = registry.get_schemas("anthropic")
    assert fresh_anthropic_schemas[0]["input_schema"]["properties"]["a"]["type"] == "integer"

    gemini_schemas = registry.get_schemas("gemini")
    gemini_schemas[0]["functionDeclarations"][0]["parameters"]["properties"]["a"]["type"] = "string"

    fresh_gemini_schemas = registry.get_schemas("gemini")
    assert (
        fresh_gemini_schemas[0]["functionDeclarations"][0]["parameters"]["properties"]["a"]["type"]
        == "integer"
    )


def test_gemini_schema_omits_additional_properties() -> None:
    @tool
    def add(a: int, b: int = 1) -> int:
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    declaration = registry.get_schemas("gemini")[0]["functionDeclarations"][0]
    assert "additionalProperties" not in declaration["parameters"]
    assert declaration["parameters"]["required"] == ["a"]


def test_gemini_schema_omits_parameters_for_zero_arg_tools() -> None:
    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry()
    registry.register(ping)

    declaration = registry.get_schemas("gemini")[0]["functionDeclarations"][0]
    assert declaration == {"name": "ping", "description": ""}


def test_gemini_schema_omits_empty_required() -> None:
    @tool
    def greet(name: str = "world") -> str:
        return f"hello {name}"

    registry = ToolRegistry()
    registry.register(greet)

    declaration = registry.get_schemas("gemini")[0]["functionDeclarations"][0]
    assert "required" not in declaration["parameters"]
    assert declaration["parameters"]["properties"] == {"name": {"type": "string"}}


def test_error_is_captured() -> None:
    @tool
    def crash() -> None:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(crash)

    result = registry.execute([{"id": "call2", "name": "crash", "arguments": {}}])[0]
    assert "boom" in result["error"]
    assert result["attempts"] == 1


def test_format_tool_results() -> None:
    formatted = format_tool_results(
        [
            {"tool_call_id": "c1", "output": "ok", "error": None},
            {"tool_call_id": "c2", "output": None, "error": "bad"},
        ]
    )

    assert formatted[0]["role"] == "tool"
    assert formatted[1]["content"] == "ERROR: bad"


def test_parallel_execution_is_faster_than_sequential() -> None:
    @tool
    def slow_double(value: int) -> int:
        sleep(0.2)
        return value * 2

    calls = [
        {"id": "t1", "name": "slow_double", "arguments": {"value": 1}},
        {"id": "t2", "name": "slow_double", "arguments": {"value": 2}},
        {"id": "t3", "name": "slow_double", "arguments": {"value": 3}},
    ]

    registry = ToolRegistry()
    registry.register(slow_double)

    seq_start = perf_counter()
    seq_results = registry.execute(calls, parallel=False)
    seq_duration = perf_counter() - seq_start

    par_start = perf_counter()
    par_results = registry.execute(calls, parallel=True, max_workers=3)
    par_duration = perf_counter() - par_start

    assert [r["output"] for r in seq_results] == [2, 4, 6]
    assert [r["output"] for r in par_results] == [2, 4, 6]
    assert par_duration < seq_duration * 0.7


def test_retry_succeeds_after_transient_failures_and_emits_events() -> None:
    attempts = {"count": 0}
    events: list[dict[str, object]] = []

    @tool
    def flaky(value: int) -> int:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")
        return value + 1

    registry = ToolRegistry(on_event=events.append)
    registry.register(flaky)

    result = registry.execute(
        [{"id": "r1", "name": "flaky", "arguments": {"value": 7}}],
        retries=2,
    )[0]

    assert result["error"] is None
    assert result["output"] == 8
    assert result["attempts"] == 3
    assert any(e["event"] == "tool_retry" for e in events)


def test_retry_exhaustion_returns_error() -> None:
    @tool
    def always_fail() -> None:
        raise RuntimeError("nope")

    registry = ToolRegistry()
    registry.register(always_fail)

    result = registry.execute(
        [{"id": "r2", "name": "always_fail", "arguments": {}}],
        retries=2,
        retry_delay_seconds=0.01,
    )[0]

    assert result["output"] is None
    assert "nope" in result["error"]
    assert result["attempts"] == 3


def test_sync_execute_supports_async_tools() -> None:
    @tool
    async def async_add(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a + b

    registry = ToolRegistry()
    registry.register(async_add)

    result = registry.execute(
        [
            {"id": "a1", "name": "async_add", "arguments": {"a": 3, "b": 4}},
        ]
    )[0]

    assert result["error"] is None
    assert result["output"] == 7


def test_execute_async_parallel_and_retry() -> None:
    calls_seen = {"count": 0}

    @tool
    async def flaky_async(value: int) -> int:
        calls_seen["count"] += 1
        await asyncio.sleep(0.05)
        if calls_seen["count"] == 1:
            raise RuntimeError("try again")
        return value * 10

    registry = ToolRegistry()
    registry.register(flaky_async)

    results = asyncio.run(
        registry.execute_async(
            [
                {"id": "aa1", "name": "flaky_async", "arguments": {"value": 2}},
                {"id": "aa2", "name": "flaky_async", "arguments": {"value": 3}},
            ],
            parallel=True,
            max_concurrency=2,
            retries=1,
        )
    )

    outputs = sorted(r["output"] for r in results if r["error"] is None)
    assert outputs == [20, 30]
    assert max(r["attempts"] for r in results) >= 1
