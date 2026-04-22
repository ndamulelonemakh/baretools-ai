from __future__ import annotations

from time import perf_counter, sleep

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

    results = registry.execute([
        {"id": "call1", "name": "add", "arguments": {"a": 3, "b": 2}},
    ])

    assert results[0]["error"] is None
    assert results[0]["output"] == 5
    assert results[0]["attempts"] == 1


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
