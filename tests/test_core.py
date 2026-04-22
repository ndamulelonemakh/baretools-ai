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


def test_error_is_captured() -> None:
    @tool
    def crash() -> None:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(crash)

    result = registry.execute([{"id": "call2", "name": "crash", "arguments": {}}])[0]
    assert "boom" in result["error"]


def test_format_tool_results() -> None:
    formatted = format_tool_results(
        [
            {"tool_call_id": "c1", "output": "ok", "error": None},
            {"tool_call_id": "c2", "output": None, "error": "bad"},
        ]
    )

    assert formatted[0]["role"] == "tool"
    assert formatted[1]["content"] == "ERROR: bad"
