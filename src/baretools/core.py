from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import Signature, _empty, signature
from time import perf_counter
from typing import Any, Callable

JSON_SCHEMA_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    function: Callable[..., Any]
    schema: dict[str, Any]


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
):
    """Mark a function as a baretools tool with optional metadata overrides."""

    def _decorate(inner: Callable[..., Any]) -> Callable[..., Any]:
        inner.__baretools_tool__ = True
        inner.__baretools_name__ = name or inner.__name__
        inner.__baretools_description__ = description or (inner.__doc__ or "").strip()
        return inner

    if func is None:
        return _decorate
    return _decorate(func)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            raise TypeError("Tool must be callable")

        tool_name = getattr(fn, "__baretools_name__", fn.__name__)
        description = getattr(fn, "__baretools_description__", (fn.__doc__ or "").strip())

        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' already registered")

        sig = signature(fn)
        schema = _function_to_openai_schema(tool_name, description, sig)
        self._tools[tool_name] = RegisteredTool(
            name=tool_name,
            description=description,
            function=fn,
            schema=schema,
        )

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def execute(
        self,
        tool_calls: list[dict[str, Any]],
        parallel: bool = False,
    ) -> list[dict[str, Any]]:
        if parallel and len(tool_calls) > 1:
            with ThreadPoolExecutor() as pool:
                return list(pool.map(self._execute_one, tool_calls))
        return [self._execute_one(call) for call in tool_calls]

    def _execute_one(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")

        started = perf_counter()
        try:
            if name not in self._tools:
                raise KeyError(f"Unknown tool '{name}'")

            output = self._tools[name].function(**arguments)
            return {
                "tool_call_id": call_id,
                "tool_name": name,
                "output": output,
                "error": None,
                "execution_time_ms": _elapsed_ms(started),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "tool_call_id": call_id,
                "tool_name": name,
                "output": None,
                "error": str(exc),
                "execution_time_ms": _elapsed_ms(started),
            }


def parse_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Parse OpenAI-style message.tool_calls into a normalized internal shape."""
    raw_calls = getattr(message, "tool_calls", None) or []
    normalized = []
    for call in raw_calls:
        normalized.append(
            {
                "id": getattr(call, "id", None),
                "name": getattr(call.function, "name", None),
                "arguments": getattr(call.function, "arguments", "{}"),
            }
        )
    return normalized


def format_tool_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format internal tool results for an OpenAI tool-role message append."""
    formatted = []
    for result in results:
        content = result["output"] if result["error"] is None else f"ERROR: {result['error']}"
        formatted.append(
            {
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": str(content),
            }
        )
    return formatted


def _function_to_openai_schema(name: str, description: str, sig: Signature) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise TypeError(f"Unsupported parameter kind for '{param_name}': {param.kind}")

        json_type = _annotation_to_json_type(param.annotation)
        properties[param_name] = {"type": json_type}

        if param.default is _empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _annotation_to_json_type(annotation: Any) -> str:
    if annotation is _empty:
        return "string"

    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    return JSON_SCHEMA_TYPE_MAP.get(annotation, "string")


def _elapsed_ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)
