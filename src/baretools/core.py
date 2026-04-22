from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import Signature, _empty, signature
from time import perf_counter, sleep
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
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._logger = logger or logging.getLogger(__name__)
        self._on_event = on_event

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
        *,
        parallel: bool = False,
        max_workers: int | None = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Execute one or more tool calls.

        Args:
            tool_calls: Normalized tool calls from parse_tool_calls() or equivalent.
            parallel: Run calls concurrently with ThreadPoolExecutor.
            max_workers: Optional ThreadPoolExecutor max_workers override.
            retries: Number of retries after the initial failed attempt.
            retry_delay_seconds: Delay between retries.
        """
        if parallel and len(tool_calls) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(
                    pool.map(
                        lambda call: self._execute_with_retry(
                            call,
                            retries=retries,
                            retry_delay_seconds=retry_delay_seconds,
                        ),
                        tool_calls,
                    )
                )

        return [
            self._execute_with_retry(
                call,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            for call in tool_calls
        ]

    def _execute_with_retry(
        self,
        tool_call: dict[str, Any],
        *,
        retries: int,
        retry_delay_seconds: float,
    ) -> dict[str, Any]:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be >= 0")

        call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")

        started = perf_counter()
        attempts = 0
        last_error: Exception | None = None

        for attempt_idx in range(retries + 1):
            attempts = attempt_idx + 1
            try:
                if name not in self._tools:
                    raise KeyError(f"Unknown tool '{name}'")

                self._emit_event(
                    {
                        "event": "tool_attempt",
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "attempt": attempts,
                    }
                )

                output = self._tools[name].function(**arguments)
                self._logger.debug(
                    "tool call succeeded",
                    extra={"tool_name": name, "tool_call_id": call_id, "attempt": attempts},
                )
                return {
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "output": output,
                    "error": None,
                    "attempts": attempts,
                    "execution_time_ms": _elapsed_ms(started),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.warning(
                    "tool call failed",
                    extra={"tool_name": name, "tool_call_id": call_id, "attempt": attempts},
                )
                self._emit_event(
                    {
                        "event": "tool_retry" if attempt_idx < retries else "tool_failed",
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "attempt": attempts,
                        "error": str(exc),
                    }
                )

                if attempt_idx < retries and retry_delay_seconds > 0:
                    sleep(retry_delay_seconds)

        return {
            "tool_call_id": call_id,
            "tool_name": name,
            "output": None,
            "error": str(last_error) if last_error else "unknown tool execution error",
            "attempts": attempts,
            "execution_time_ms": _elapsed_ms(started),
        }

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event)


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
