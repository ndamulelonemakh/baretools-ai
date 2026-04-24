from __future__ import annotations

import asyncio
from time import perf_counter, sleep

import pytest

from baretools import ToolRegistry, format_tool_results, parse_tool_calls, tool


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


def test_parse_tool_calls_openai_dict() -> None:
    message = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": '{"x": 1}'}},
        ]
    }
    calls = parse_tool_calls(message, "openai")
    assert calls == [{"id": "c1", "name": "f", "arguments": '{"x": 1}'}]


def test_parse_tool_calls_anthropic() -> None:
    class Block:
        def __init__(self, **kw: object) -> None:
            self.__dict__.update(kw)

    class Msg:
        content = [
            Block(type="text", text="thinking"),
            Block(type="tool_use", id="tu_1", name="f", input={"x": 1}),
        ]

    calls = parse_tool_calls(Msg(), "anthropic")
    assert calls == [{"id": "tu_1", "name": "f", "arguments": {"x": 1}}]


def test_parse_tool_calls_gemini_function_calls_attr() -> None:
    class Call:
        id = "g_1"
        name = "f"
        args = {"x": 1}

    class Resp:
        function_calls = [Call()]

    calls = parse_tool_calls(Resp(), "gemini")
    assert calls == [{"id": "g_1", "name": "f", "arguments": {"x": 1}}]


def test_parse_tool_calls_gemini_parts_fallback() -> None:
    class FC:
        id = None
        name = "f"
        args = {"x": 1}

    class Part:
        function_call = FC()

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()

    class Resp:
        function_calls = None
        candidates = [Candidate()]

    calls = parse_tool_calls(Resp(), "gemini")
    assert calls == [{"id": None, "name": "f", "arguments": {"x": 1}}]


def test_parse_tool_calls_unknown_provider() -> None:
    with pytest.raises(ValueError):
        parse_tool_calls({}, "cohere")


def test_format_tool_results_anthropic() -> None:
    formatted = format_tool_results(
        [
            {"tool_call_id": "tu_1", "tool_name": "f", "output": {"v": 1}, "error": None},
            {"tool_call_id": "tu_2", "tool_name": "g", "output": None, "error": "boom"},
        ],
        "anthropic",
    )
    assert formatted[0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "{'v': 1}"}
    assert formatted[1]["is_error"] is True
    assert formatted[1]["content"] == "ERROR: boom"


def test_format_tool_results_gemini() -> None:
    formatted = format_tool_results(
        [
            {"tool_call_id": "g1", "tool_name": "f", "output": {"v": 1}, "error": None},
            {"tool_call_id": "g2", "tool_name": "g", "output": None, "error": "boom"},
        ],
        "gemini",
    )
    assert formatted == [
        {"name": "f", "response": {"result": {"v": 1}}},
        {"name": "g", "response": {"error": "boom"}},
    ]


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


def test_pydantic_model_parameter_schema_and_coercion() -> None:
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Address(BaseModel):
        street: str
        city: str
        zip: str

    @tool
    def create_user(name: str, address: Address) -> dict:
        return {"name": name, "city": address.city, "zip": address.zip}

    registry = ToolRegistry()
    registry.register(create_user)

    schema = registry.get_schemas("openai")[0]["function"]["parameters"]
    assert schema["properties"]["name"] == {"type": "string"}
    address_schema = schema["properties"]["address"]
    assert address_schema["type"] == "object"
    assert set(address_schema["properties"].keys()) == {"street", "city", "zip"}
    assert sorted(address_schema["required"]) == ["city", "street", "zip"]

    result = registry.execute(
        [
            {
                "id": "p1",
                "name": "create_user",
                "arguments": {
                    "name": "Ada",
                    "address": {"street": "1 Infinite Loop", "city": "Cupertino", "zip": "95014"},
                },
            }
        ]
    )[0]

    assert result["error"] is None
    assert result["output"] == {"name": "Ada", "city": "Cupertino", "zip": "95014"}


def test_pydantic_validation_error_surfaces_as_tool_error() -> None:
    pytest.importorskip("pydantic")
    from pydantic import BaseModel

    class Item(BaseModel):
        qty: int

    @tool
    def buy(item: Item) -> int:
        return item.qty

    registry = ToolRegistry()
    registry.register(buy)

    result = registry.execute(
        [{"id": "p2", "name": "buy", "arguments": {"item": {"qty": "not-an-int"}}}]
    )[0]

    assert result["output"] is None
    assert result["error"] is not None
    assert "qty" in result["error"]


def test_execute_stream_yields_results_as_completed() -> None:
    @tool
    def slow(n: int) -> int:
        sleep(0.05 if n == 1 else 0.0)
        return n

    registry = ToolRegistry()
    registry.register(slow)

    calls = [
        {"id": "s1", "name": "slow", "arguments": {"n": 1}},
        {"id": "s2", "name": "slow", "arguments": {"n": 2}},
    ]
    yielded = list(registry.execute_stream(calls, parallel=True, max_workers=2))

    assert sorted(r["output"] for r in yielded) == [1, 2]
    assert yielded[0]["output"] == 2  # fast one finishes first


def test_execute_stream_serial_preserves_order() -> None:
    @tool
    def echo(n: int) -> int:
        return n

    registry = ToolRegistry()
    registry.register(echo)

    calls = [{"id": str(i), "name": "echo", "arguments": {"n": i}} for i in range(3)]
    outputs = [r["output"] for r in registry.execute_stream(calls)]
    assert outputs == [0, 1, 2]


def test_execute_stream_async_yields_as_completed() -> None:
    @tool
    async def slow_async(n: int) -> int:
        await asyncio.sleep(0.05 if n == 1 else 0.0)
        return n

    registry = ToolRegistry()
    registry.register(slow_async)

    async def collect() -> list[int]:
        out: list[int] = []
        async for result in registry.execute_stream_async(
            [
                {"id": "a1", "name": "slow_async", "arguments": {"n": 1}},
                {"id": "a2", "name": "slow_async", "arguments": {"n": 2}},
            ],
            parallel=True,
            max_concurrency=2,
        ):
            out.append(result["output"])
        return out

    outputs = asyncio.run(collect())
    assert sorted(outputs) == [1, 2]
    assert outputs[0] == 2


def test_dataclass_model_parameter_schema_and_coercion() -> None:
    from dataclasses import dataclass

    @dataclass
    class Address:
        street: str
        city: str
        zip: str

    @tool
    def create_user(name: str, address: Address) -> dict:
        return {"name": name, "city": address.city, "zip": address.zip}

    registry = ToolRegistry()
    registry.register(create_user)

    schema = registry.get_schemas("openai")[0]["function"]["parameters"]
    assert schema["properties"]["name"] == {"type": "string"}
    address_schema = schema["properties"]["address"]
    assert address_schema["type"] == "object"
    assert set(address_schema["properties"].keys()) == {"street", "city", "zip"}
    assert sorted(address_schema["required"]) == ["city", "street", "zip"]

    # Test coercion mapping dictionary back to the dataclass model
    args = {"name": "Bob", "address": {"street": "123 Elm", "city": "NYC", "zip": "10001"}}
    result = registry.execute([{"id": "test", "name": "create_user", "arguments": args}])[0]
    assert result["output"] == {"name": "Bob", "city": "NYC", "zip": "10001"}


def test_register_rejects_duplicate_tool_name() -> None:
    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry()
    registry.register(ping)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ping)


def test_execute_returns_error_for_malformed_json_arguments() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    registry = ToolRegistry()
    registry.register(add)

    result = registry.execute([{"id": "bad-json", "name": "add", "arguments": '{"a": 1,'}])[0]

    assert result["output"] is None
    assert result["attempts"] == 1
    assert "Expecting" in (result["error"] or "")


def test_execute_reports_unknown_tool_calls_cleanly() -> None:
    registry = ToolRegistry()

    result = registry.execute([{"id": "missing", "name": "does_not_exist", "arguments": {}}])[0]

    assert result["output"] is None
    assert result["attempts"] == 1
    assert "Unknown tool" in (result["error"] or "")


def test_execute_async_rejects_negative_retries() -> None:
    @tool
    async def ping() -> str:
        return "pong"

    registry = ToolRegistry()
    registry.register(ping)

    with pytest.raises(ValueError, match="retries must be >= 0"):
        asyncio.run(
            registry.execute_async(
                [{"id": "x", "name": "ping", "arguments": {}}],
                retries=-1,
            )
        )


def test_format_tool_results_rejects_json_schema_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        format_tool_results([], "json_schema")


def test_execute_returns_error_when_tool_receives_unexpected_arguments() -> None:
    @tool
    def multiply(a: int, b: int) -> int:
        return a * b

    registry = ToolRegistry()
    registry.register(multiply)

    result = registry.execute(
        [
            {
                "id": "too-many-args",
                "name": "multiply",
                "arguments": {"a": 2, "b": 3, "c": 4},
            }
        ]
    )[0]

    assert result["output"] is None
    assert result["attempts"] == 1
    assert "unexpected keyword argument" in (result["error"] or "")


def test_execute_parallel_handles_large_batches_of_tool_calls() -> None:
    @tool
    def identity(n: int) -> int:
        return n

    registry = ToolRegistry()
    registry.register(identity)

    calls = [{"id": f"i{idx}", "name": "identity", "arguments": {"n": idx}} for idx in range(200)]

    results = registry.execute(calls, parallel=True, max_workers=8)

    assert len(results) == len(calls)
    assert all(result["error"] is None for result in results)
    assert sorted(result["output"] for result in results) == list(range(200))


def test_execute_stream_parallel_handles_large_batches_of_tool_calls() -> None:
    @tool
    def identity(n: int) -> int:
        return n

    registry = ToolRegistry()
    registry.register(identity)

    calls = [{"id": f"s{idx}", "name": "identity", "arguments": {"n": idx}} for idx in range(150)]

    streamed = list(registry.execute_stream(calls, parallel=True, max_workers=6))

    assert len(streamed) == len(calls)
    assert all(result["error"] is None for result in streamed)
    assert sorted(result["output"] for result in streamed) == list(range(150))


def test_registry_rejects_invalid_overload_limits() -> None:
    with pytest.raises(ValueError, match="max_tool_calls_per_batch"):
        ToolRegistry(max_tool_calls_per_batch=0)

    with pytest.raises(ValueError, match="max_argument_payload_chars"):
        ToolRegistry(max_argument_payload_chars=0)


def test_execute_rejects_batches_larger_than_configured_limit() -> None:
    @tool
    def identity(n: int) -> int:
        return n

    registry = ToolRegistry(max_tool_calls_per_batch=2)
    registry.register(identity)

    calls = [
        {"id": "c1", "name": "identity", "arguments": {"n": 1}},
        {"id": "c2", "name": "identity", "arguments": {"n": 2}},
        {"id": "c3", "name": "identity", "arguments": {"n": 3}},
    ]

    with pytest.raises(ValueError, match="Too many tool calls"):
        registry.execute(calls)


def test_execute_rejects_argument_payloads_over_configured_limit() -> None:
    @tool
    def echo(text: str) -> str:
        return text

    registry = ToolRegistry(max_argument_payload_chars=20)
    registry.register(echo)

    with pytest.raises(ValueError, match="payload exceeds limit"):
        registry.execute(
            [
                {
                    "id": "big-args",
                    "name": "echo",
                    "arguments": {"text": "x" * 100},
                }
            ]
        )


def test_execute_stream_rejects_batches_larger_than_configured_limit() -> None:
    @tool
    def identity(n: int) -> int:
        return n

    registry = ToolRegistry(max_tool_calls_per_batch=1)
    registry.register(identity)

    calls = [
        {"id": "s1", "name": "identity", "arguments": {"n": 1}},
        {"id": "s2", "name": "identity", "arguments": {"n": 2}},
    ]

    with pytest.raises(ValueError, match="Too many tool calls"):
        list(registry.execute_stream(calls))


def test_execute_sync_timeout_returns_error_result() -> None:
    @tool
    def slow() -> str:
        sleep(0.5)
        return "done"

    registry = ToolRegistry()
    registry.register(slow)

    [result] = registry.execute(
        [{"id": "t1", "name": "slow", "arguments": {}}],
        timeout=0.05,
    )
    assert result["error"] is not None
    assert "timeout" in result["error"].lower()
    assert result["output"] is None


def test_execute_sync_timeout_allows_retry_to_succeed() -> None:
    attempts: list[float] = []

    @tool
    def flaky() -> str:
        attempts.append(perf_counter())
        if len(attempts) == 1:
            sleep(0.5)
        return "ok"

    registry = ToolRegistry()
    registry.register(flaky)

    [result] = registry.execute(
        [{"id": "t1", "name": "flaky", "arguments": {}}],
        timeout=0.05,
        retries=1,
    )
    assert result["error"] is None
    assert result["output"] == "ok"
    assert result["attempts"] == 2


def test_execute_async_timeout_returns_error_result() -> None:
    @tool
    async def slow_async() -> str:
        await asyncio.sleep(0.5)
        return "done"

    registry = ToolRegistry()
    registry.register(slow_async)

    async def run() -> list:
        return await registry.execute_async(
            [{"id": "t1", "name": "slow_async", "arguments": {}}],
            timeout=0.05,
        )

    [result] = asyncio.run(run())
    assert result["error"] is not None
    assert "timeout" in result["error"].lower()


def test_execute_rejects_invalid_timeout() -> None:
    @tool
    def noop() -> int:
        return 1

    registry = ToolRegistry()
    registry.register(noop)
    call = [{"id": "t1", "name": "noop", "arguments": {}}]

    with pytest.raises(ValueError, match="timeout must be > 0"):
        registry.execute(call, timeout=0)
    with pytest.raises(TypeError, match="timeout must be a number"):
        registry.execute(call, timeout="1")  # type: ignore[arg-type]


def test_parse_tool_calls_strict_raises_on_malformed_openai_arguments() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "do_thing", "arguments": "{not-json"},
            }
        ]
    }

    with pytest.raises(ValueError, match="Malformed JSON arguments"):
        parse_tool_calls(message, "openai", strict=True)


def test_parse_tool_calls_strict_passes_valid_openai_arguments() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "do_thing", "arguments": '{"x": 1}'},
            }
        ]
    }

    [call] = parse_tool_calls(message, "openai", strict=True)
    assert call["name"] == "do_thing"
    assert call["arguments"] == '{"x": 1}'


def test_parse_tool_calls_strict_is_noop_for_anthropic_and_gemini() -> None:
    anthropic_msg = {
        "content": [{"type": "tool_use", "id": "tu_1", "name": "do_thing", "input": {"x": 1}}]
    }
    [a_call] = parse_tool_calls(anthropic_msg, "anthropic", strict=True)
    assert a_call["arguments"] == {"x": 1}

    gemini_msg = {"function_calls": [{"id": "g1", "name": "do_thing", "args": {"x": 1}}]}
    [g_call] = parse_tool_calls(gemini_msg, "gemini", strict=True)
    assert g_call["arguments"] == {"x": 1}


def test_execute_stream_timeout_emits_error_result_for_slow_call() -> None:
    @tool
    def quick() -> int:
        return 1

    @tool
    def slow() -> int:
        sleep(0.5)
        return 2

    registry = ToolRegistry()
    registry.register(quick)
    registry.register(slow)

    results = list(
        registry.execute_stream(
            [
                {"id": "q", "name": "quick", "arguments": {}},
                {"id": "s", "name": "slow", "arguments": {}},
            ],
            timeout=0.05,
        )
    )

    by_id = {r["tool_call_id"]: r for r in results}
    assert by_id["q"]["error"] is None
    assert by_id["q"]["output"] == 1
    assert by_id["s"]["error"] is not None
    assert "timeout" in by_id["s"]["error"].lower()


def test_execute_parallel_timeout_isolates_slow_call() -> None:
    @tool
    def quick(n: int) -> int:
        return n

    @tool
    def slow() -> str:
        sleep(0.5)
        return "done"

    registry = ToolRegistry()
    registry.register(quick)
    registry.register(slow)

    results = registry.execute(
        [
            {"id": "q1", "name": "quick", "arguments": {"n": 1}},
            {"id": "s1", "name": "slow", "arguments": {}},
            {"id": "q2", "name": "quick", "arguments": {"n": 2}},
        ],
        parallel=True,
        timeout=0.05,
    )

    by_id = {r["tool_call_id"]: r for r in results}
    assert by_id["q1"]["output"] == 1
    assert by_id["q2"]["output"] == 2
    assert by_id["s1"]["error"] is not None
    assert "timeout" in by_id["s1"]["error"].lower()


def test_execute_async_stream_timeout_emits_error_result_for_slow_call() -> None:
    @tool
    async def quick() -> int:
        return 1

    @tool
    async def slow() -> int:
        await asyncio.sleep(0.5)
        return 2

    registry = ToolRegistry()
    registry.register(quick)
    registry.register(slow)

    async def run() -> list:
        out = []
        async for item in registry.execute_stream_async(
            [
                {"id": "q", "name": "quick", "arguments": {}},
                {"id": "s", "name": "slow", "arguments": {}},
            ],
            timeout=0.05,
        ):
            out.append(item)
        return out

    results = asyncio.run(run())
    by_id = {r["tool_call_id"]: r for r in results}
    assert by_id["q"]["error"] is None
    assert by_id["s"]["error"] is not None
    assert "timeout" in by_id["s"]["error"].lower()


def test_execute_rejects_negative_and_bool_timeouts() -> None:
    @tool
    def noop() -> int:
        return 1

    registry = ToolRegistry()
    registry.register(noop)
    call = [{"id": "t1", "name": "noop", "arguments": {}}]

    with pytest.raises(ValueError, match="timeout must be > 0"):
        registry.execute(call, timeout=-0.5)
    with pytest.raises(TypeError, match="timeout must be a number"):
        registry.execute(call, timeout=True)  # type: ignore[arg-type]


def test_execute_async_rejects_invalid_timeout() -> None:
    @tool
    async def noop() -> int:
        return 1

    registry = ToolRegistry()
    registry.register(noop)
    call = [{"id": "t1", "name": "noop", "arguments": {}}]

    async def run_zero() -> None:
        await registry.execute_async(call, timeout=0)

    async def run_str() -> None:
        await registry.execute_async(call, timeout="1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="timeout must be > 0"):
        asyncio.run(run_zero())
    with pytest.raises(TypeError, match="timeout must be a number"):
        asyncio.run(run_str())


class _FakeOpenAIFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeOpenAIToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _FakeOpenAIFunction(name, arguments)


class _FakeOpenAIMessage:
    def __init__(self, tool_calls: list[_FakeOpenAIToolCall]) -> None:
        self.tool_calls = tool_calls


def test_parse_tool_calls_strict_raises_for_openai_sdk_object_form() -> None:
    message = _FakeOpenAIMessage([_FakeOpenAIToolCall("call_1", "do_thing", "{not-json")])

    with pytest.raises(ValueError, match="Malformed JSON arguments"):
        parse_tool_calls(message, "openai", strict=True)


def test_parse_tool_calls_strict_validates_every_openai_call() -> None:
    message = {
        "tool_calls": [
            {"id": "ok", "function": {"name": "a", "arguments": '{"x": 1}'}},
            {"id": "bad", "function": {"name": "b", "arguments": "{nope"}},
        ]
    }

    with pytest.raises(ValueError, match="Malformed JSON arguments for tool 'b'"):
        parse_tool_calls(message, "openai", strict=True)


def test_parse_tool_calls_default_defers_malformed_openai_arguments_to_execute() -> None:
    message = {
        "tool_calls": [{"id": "call_1", "function": {"name": "do_thing", "arguments": "{not-json"}}]
    }

    [call] = parse_tool_calls(message, "openai")
    assert call["arguments"] == "{not-json"

    @tool
    def do_thing(x: int) -> int:
        return x

    registry = ToolRegistry()
    registry.register(do_thing)
    [result] = registry.execute([call])
    assert result["error"] is not None
    assert "json" in result["error"].lower() or "expecting" in result["error"].lower()


def test_parse_tool_calls_strict_accepts_empty_openai_arguments() -> None:
    message = {"tool_calls": [{"id": "c1", "function": {"name": "noop", "arguments": ""}}]}

    [call] = parse_tool_calls(message, "openai", strict=True)
    assert call["arguments"] == ""
