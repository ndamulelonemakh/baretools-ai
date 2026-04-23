from __future__ import annotations

import asyncio
import inspect
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import Signature, _empty, signature
from time import perf_counter, sleep
from typing import Any, Callable, Literal, Mapping, TypedDict, TypeVar, cast


class ToolCall(TypedDict, total=False):
    id: str | None
    tool_call_id: str | None
    name: str | None
    arguments: dict[str, Any] | str


class ToolResult(TypedDict):
    tool_call_id: str | None
    tool_name: str | None
    output: Any
    error: str | None
    attempts: int
    execution_time_ms: int


class ToolEvent(TypedDict, total=False):
    event: Literal["tool_attempt", "tool_retry", "tool_failed"]
    tool_call_id: str | None
    tool_name: str | None
    attempt: int
    error: str


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str | None
    content: str


JSON_SCHEMA_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any]


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> F | Callable[[F], F]:
    """Mark a function as a baretools tool with optional metadata overrides."""

    def _decorate(inner: F) -> F:
        inner.__baretools_tool__ = True  # type: ignore[attr-defined]
        inner.__baretools_name__ = name or inner.__name__  # type: ignore[attr-defined]
        inner.__baretools_description__ = description or (inner.__doc__ or "").strip()  # type: ignore[attr-defined]
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
        parameters = _signature_to_json_schema(sig)
        self._tools[tool_name] = RegisteredTool(
            name=tool_name,
            description=description,
            function=fn,
            parameters=parameters,
        )

    def get_schemas(
        self,
        provider: Literal["openai", "anthropic", "gemini", "json_schema"] = "openai",
    ) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            if provider == "openai":
                schemas.append(_tool_to_openai_schema(tool))
            elif provider == "anthropic":
                schemas.append(_tool_to_anthropic_schema(tool))
            elif provider == "gemini":
                schemas.append(_tool_to_gemini_schema(tool))
            elif provider == "json_schema":
                schemas.append(_tool_to_json_schema(tool))
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        return schemas

    def execute(
        self,
        tool_calls: list[ToolCall] | list[Mapping[str, Any]],
        *,
        parallel: bool = False,
        max_workers: int | None = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
    ) -> list[ToolResult]:
        """Sync execution API; supports both sync and async tool functions."""
        if parallel and len(tool_calls) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(
                    pool.map(
                        lambda call: self._execute_with_retry_sync(
                            call,
                            retries=retries,
                            retry_delay_seconds=retry_delay_seconds,
                        ),
                        tool_calls,
                    )
                )

        return [
            self._execute_with_retry_sync(
                call,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            for call in tool_calls
        ]

    async def execute_async(
        self,
        tool_calls: list[ToolCall] | list[Mapping[str, Any]],
        *,
        parallel: bool = False,
        max_concurrency: int | None = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
    ) -> list[ToolResult]:
        """Async execution API; use this from existing async agent loops."""
        if parallel and len(tool_calls) > 1:
            semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

            async def _run(call: Mapping[str, Any]) -> ToolResult:
                if semaphore is None:
                    return await self._execute_with_retry_async(
                        call,
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                async with semaphore:
                    return await self._execute_with_retry_async(
                        call,
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                    )

            return list(await asyncio.gather(*(_run(call) for call in tool_calls)))

        return [
            await self._execute_with_retry_async(
                call,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            for call in tool_calls
        ]

    def _execute_with_retry_sync(
        self,
        tool_call: Mapping[str, Any],
        *,
        retries: int,
        retry_delay_seconds: float,
    ) -> ToolResult:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be >= 0")

        call_id, name, arguments = _normalize_call(tool_call)

        started = perf_counter()
        attempts = 0
        last_error: Exception | None = None

        for attempt_idx in range(retries + 1):
            attempts = attempt_idx + 1
            try:
                registered_tool = self._require_tool(name)
                self._emit_event(
                    {
                        "event": "tool_attempt",
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "attempt": attempts,
                    }
                )

                output = registered_tool.function(**arguments)
                if inspect.isawaitable(output):
                    output = _run_awaitable_in_sync(output)

                self._logger.debug(
                    "tool call succeeded",
                    extra={"tool_name": name, "tool_call_id": call_id, "attempt": attempts},
                )
                return _result(call_id, name, output, None, attempts, started)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._emit_failure(call_id, name, attempts, attempt_idx, retries, exc)
                if attempt_idx < retries and retry_delay_seconds > 0:
                    sleep(retry_delay_seconds)

        return _result(call_id, name, None, str(last_error), attempts, started)

    async def _execute_with_retry_async(
        self,
        tool_call: Mapping[str, Any],
        *,
        retries: int,
        retry_delay_seconds: float,
    ) -> ToolResult:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be >= 0")

        call_id, name, arguments = _normalize_call(tool_call)

        started = perf_counter()
        attempts = 0
        last_error: Exception | None = None

        for attempt_idx in range(retries + 1):
            attempts = attempt_idx + 1
            try:
                registered_tool = self._require_tool(name)
                self._emit_event(
                    {
                        "event": "tool_attempt",
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "attempt": attempts,
                    }
                )

                output = registered_tool.function(**arguments)
                if inspect.isawaitable(output):
                    output = await output

                self._logger.debug(
                    "tool call succeeded",
                    extra={"tool_name": name, "tool_call_id": call_id, "attempt": attempts},
                )
                return _result(call_id, name, output, None, attempts, started)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._emit_failure(call_id, name, attempts, attempt_idx, retries, exc)
                if attempt_idx < retries and retry_delay_seconds > 0:
                    await asyncio.sleep(retry_delay_seconds)

        return _result(call_id, name, None, str(last_error), attempts, started)

    def _require_tool(self, name: str | None) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'")
        return self._tools[name]

    def _emit_failure(
        self,
        call_id: str | None,
        name: str | None,
        attempts: int,
        attempt_idx: int,
        retries: int,
        exc: Exception,
    ) -> None:
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

    def _emit_event(self, event: ToolEvent) -> None:
        if self._on_event is not None:
            self._on_event(event)


def parse_tool_calls(message: Any) -> list[ToolCall]:
    """Parse OpenAI-style message.tool_calls into a normalized internal shape."""

    # Support both object attributes (OpenAI v1 objects) and dict keys (JSON payloads)
    if isinstance(message, dict):
        raw_calls = message.get("tool_calls", [])
    else:
        raw_calls = getattr(message, "tool_calls", None) or []

    normalized = []
    for call in raw_calls:
        if isinstance(call, dict):
            func_data = call.get("function", {})
            if isinstance(func_data, dict):
                fn_name = func_data.get("name")
                fn_args = func_data.get("arguments", "{}")
            else:
                fn_name = getattr(func_data, "name", None)
                fn_args = getattr(func_data, "arguments", "{}")
            normalized.append(
                {
                    "id": call.get("id"),
                    "name": fn_name,
                    "arguments": fn_args,
                }
            )
        else:
            func_data = getattr(call, "function", None)
            normalized.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(func_data, "name", None) if func_data else None,
                    "arguments": getattr(func_data, "arguments", "{}") if func_data else "{}",
                }
            )
    return normalized


def format_tool_results(results: list[ToolResult]) -> list[ToolMessage]:
    """Format internal tool results for an OpenAI tool-role message append."""
    formatted: list[ToolMessage] = []
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


def _normalize_call(tool_call: Mapping[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
    call_id = tool_call.get("id") or tool_call.get("tool_call_id")
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})

    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")

    return call_id, name, arguments


def _run_awaitable_in_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "Cannot execute async tool in sync execute() while an event loop is already running. "
        "Use await execute_async(...) instead."
    )


def _result(
    call_id: str | None,
    name: str | None,
    output: Any,
    error: str | None,
    attempts: int,
    started: float,
) -> ToolResult:
    return {
        "tool_call_id": call_id,
        "tool_name": name,
        "output": output,
        "error": error,
        "attempts": attempts,
        "execution_time_ms": _elapsed_ms(started),
    }


def _signature_to_json_schema(sig: Signature) -> dict[str, Any]:
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
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tool_to_openai_schema(tool: RegisteredTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _tool_to_anthropic_schema(tool: RegisteredTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _tool_to_gemini_schema(tool: RegisteredTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _tool_to_json_schema(tool: RegisteredTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "schema": cast(dict[str, Any], tool.parameters),
    }


def _annotation_to_json_type(annotation: Any) -> str:
    if annotation is _empty:
        return "string"

    if isinstance(annotation, str):
        return {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }.get(annotation, "string")

    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    return JSON_SCHEMA_TYPE_MAP.get(annotation, "string")


def _elapsed_ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)
