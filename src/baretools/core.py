from __future__ import annotations

import asyncio
import inspect
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import Signature, _empty, signature
from time import perf_counter, sleep
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Literal,
    Mapping,
    TypedDict,
    TypeVar,
    get_type_hints,
)


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


class _OpenAIToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str | None
    content: str


class _AnthropicToolResult(TypedDict, total=False):
    type: Literal["tool_result"]
    tool_use_id: str | None
    content: str
    is_error: bool


class _GeminiToolResult(TypedDict):
    name: str | None
    response: dict[str, Any]


ProviderToolResult = _OpenAIToolMessage | _AnthropicToolResult | _GeminiToolResult


JSON_SCHEMA_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini", "json_schema")

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any]
    coercions: dict[str, Callable[[Any], Any]] = field(default_factory=dict)


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
        caller_locals: dict[str, Any] = {}
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_locals = dict(frame.f_back.f_locals)
        hints = _resolve_type_hints(fn, caller_locals)
        parameters = _signature_to_json_schema(sig, hints)
        coercions = _extract_coercions(sig, hints)
        self._tools[tool_name] = RegisteredTool(
            name=tool_name,
            description=description,
            function=fn,
            parameters=parameters,
            coercions=coercions,
        )

    def get_schemas(
        self,
        provider: Literal["openai", "anthropic", "gemini", "json_schema"] = "openai",
        *,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        if strict and provider != "openai":
            raise ValueError("strict=True is only supported for the 'openai' provider")

        if not self._tools:
            return []

        if provider == "gemini":
            declarations = [
                _tool_to_gemini_function_declaration(registered)
                for registered in self._tools.values()
            ]
            return [{"functionDeclarations": declarations}]

        if provider == "openai":
            return [
                _tool_to_openai_schema(registered, strict=strict)
                for registered in self._tools.values()
            ]

        renderers = {
            "anthropic": _tool_to_anthropic_schema,
            "json_schema": _tool_to_json_schema,
        }
        render = renderers[provider]
        return [render(registered) for registered in self._tools.values()]

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

    def execute_stream(
        self,
        tool_calls: list[ToolCall] | list[Mapping[str, Any]],
        *,
        parallel: bool = False,
        max_workers: int | None = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
    ) -> Iterator[ToolResult]:
        """Yield results as each call finishes; order is completion order when parallel."""
        if parallel and len(tool_calls) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        self._execute_with_retry_sync,
                        call,
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                    for call in tool_calls
                ]
                for future in as_completed(futures):
                    yield future.result()
            return

        for call in tool_calls:
            yield self._execute_with_retry_sync(
                call,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )

    async def execute_stream_async(
        self,
        tool_calls: list[ToolCall] | list[Mapping[str, Any]],
        *,
        parallel: bool = False,
        max_concurrency: int | None = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
    ) -> AsyncIterator[ToolResult]:
        """Async generator yielding ToolResult values as each call finishes."""
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

            tasks = [asyncio.create_task(_run(call)) for call in tool_calls]
            for completed in asyncio.as_completed(tasks):
                yield await completed
            return

        for call in tool_calls:
            yield await self._execute_with_retry_async(
                call,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )

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

                coerced = _apply_coercions(arguments, registered_tool.coercions)
                output = registered_tool.function(**coerced)
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

                coerced = _apply_coercions(arguments, registered_tool.coercions)
                output = registered_tool.function(**coerced)
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


def parse_tool_calls(message: Any, provider: str = "openai") -> list[ToolCall]:
    """Normalize provider-native tool calls to baretools ToolCall dicts.

    Accepts either an SDK response object or its dict form. For ``gemini``,
    pass the ``GenerateContentResponse``; for ``anthropic``, pass the
    ``Message``; for ``openai``, pass ``response.choices[0].message``.
    """

    if provider not in _SUPPORTED_PROVIDERS or provider == "json_schema":
        raise ValueError(
            f"Unsupported provider: {provider}. Choose from openai, anthropic, gemini."
        )

    if provider == "openai":
        return _parse_openai_tool_calls(message)
    if provider == "anthropic":
        return _parse_anthropic_tool_calls(message)
    return _parse_gemini_tool_calls(message)


def _parse_openai_tool_calls(message: Any) -> list[ToolCall]:
    if isinstance(message, dict):
        raw_calls = message.get("tool_calls", [])
    else:
        raw_calls = getattr(message, "tool_calls", None) or []

    normalized: list[ToolCall] = []
    for call in raw_calls:
        if isinstance(call, dict):
            func_data = call.get("function", {})
            if isinstance(func_data, dict):
                fn_name = func_data.get("name")
                fn_args = func_data.get("arguments", "{}")
            else:
                fn_name = getattr(func_data, "name", None)
                fn_args = getattr(func_data, "arguments", "{}")
            normalized.append({"id": call.get("id"), "name": fn_name, "arguments": fn_args})
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


def _parse_anthropic_tool_calls(message: Any) -> list[ToolCall]:
    if isinstance(message, dict):
        blocks = message.get("content", [])
    else:
        blocks = getattr(message, "content", None) or []

    normalized: list[ToolCall] = []
    for block in blocks:
        if isinstance(block, dict):
            block_type = block.get("type")
            block_id = block.get("id")
            block_name = block.get("name")
            block_input = block.get("input", {})
        else:
            block_type = getattr(block, "type", None)
            block_id = getattr(block, "id", None)
            block_name = getattr(block, "name", None)
            block_input = getattr(block, "input", {})

        if block_type != "tool_use":
            continue
        normalized.append({"id": block_id, "name": block_name, "arguments": block_input or {}})
    return normalized


def _parse_gemini_tool_calls(message: Any) -> list[ToolCall]:
    raw_calls: list[Any] = []
    if isinstance(message, dict):
        raw_calls = list(message.get("function_calls") or [])
        if not raw_calls:
            candidates = message.get("candidates") or []
            if candidates:
                content = candidates[0].get("content") or {}
                for part in content.get("parts", []):
                    fc = part.get("function_call") if isinstance(part, dict) else None
                    if fc is not None:
                        raw_calls.append(fc)
    else:
        function_calls = getattr(message, "function_calls", None)
        if function_calls:
            raw_calls = list(function_calls)
        else:
            candidates = getattr(message, "candidates", None) or []
            if candidates:
                parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
                for part in parts:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        raw_calls.append(fc)

    normalized: list[ToolCall] = []
    for call in raw_calls:
        if isinstance(call, dict):
            args = call.get("args") or {}
            normalized.append(
                {"id": call.get("id"), "name": call.get("name"), "arguments": dict(args)}
            )
        else:
            args = getattr(call, "args", None) or {}
            normalized.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(call, "name", None),
                    "arguments": dict(args),
                }
            )
    return normalized


def format_tool_results(
    results: list[ToolResult],
    provider: str = "openai",
) -> list[ProviderToolResult]:
    """Format tool results as provider-ready content/messages.

    Returns ``list[ProviderToolResult]``; the concrete shape varies by provider:

    - ``openai``: ``{role, tool_call_id, content}`` messages to extend the
      conversation directly.
    - ``anthropic``: ``tool_result`` content blocks to wrap in a single
      ``{"role": "user", "content": [...]}`` message.
    - ``gemini``: ``{name, response}`` dicts to wrap in
      ``types.Part.from_function_response(**item)`` and a ``user`` ``Content``.
    """

    if provider not in _SUPPORTED_PROVIDERS or provider == "json_schema":
        raise ValueError(
            f"Unsupported provider: {provider}. Choose from openai, anthropic, gemini."
        )

    if provider == "openai":
        formatted_openai: list[_OpenAIToolMessage] = []
        for result in results:
            content = result["output"] if result["error"] is None else f"ERROR: {result['error']}"
            formatted_openai.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": str(content),
                }
            )
        return list(formatted_openai)

    if provider == "anthropic":
        formatted_anthropic: list[_AnthropicToolResult] = []
        for result in results:
            block: _AnthropicToolResult = {
                "type": "tool_result",
                "tool_use_id": result["tool_call_id"],
            }
            if result["error"] is None:
                block["content"] = str(result["output"])
            else:
                block["content"] = f"ERROR: {result['error']}"
                block["is_error"] = True
            formatted_anthropic.append(block)
        return list(formatted_anthropic)

    formatted_gemini: list[_GeminiToolResult] = []
    for result in results:
        if result["error"] is None:
            payload: dict[str, Any] = {"result": result["output"]}
        else:
            payload = {"error": result["error"]}
        formatted_gemini.append({"name": result["tool_name"], "response": payload})
    return list(formatted_gemini)


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


def _signature_to_json_schema(
    sig: Signature,
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = hints or {}

    for param_name, param in sig.parameters.items():
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise TypeError(f"Unsupported parameter kind for '{param_name}': {param.kind}")

        annotation = hints.get(param_name, param.annotation)
        pydantic_schema = _pydantic_schema_for(annotation)
        dataclass_schema = _dataclass_schema_for(annotation)
        if pydantic_schema is not None:
            properties[param_name] = pydantic_schema
        elif dataclass_schema is not None:
            properties[param_name] = dataclass_schema
        else:
            properties[param_name] = {"type": _annotation_to_json_type(annotation)}

        if param.default is _empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _pydantic_base_model() -> type | None:
    try:
        from pydantic import BaseModel
    except ImportError:
        return None
    return BaseModel


def _resolve_type_hints(
    fn: Callable[..., Any],
    extra_locals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    localns: dict[str, Any] = {}
    if extra_locals:
        localns.update(extra_locals)
    try:
        closure = inspect.getclosurevars(fn)
        localns.update(closure.nonlocals)
        localns.update(closure.globals)
    except (TypeError, ValueError):
        pass
    try:
        return get_type_hints(fn, localns=localns)
    except Exception:
        return {}


def _is_pydantic_model(annotation: Any) -> bool:
    base = _pydantic_base_model()
    if base is None:
        return False
    return isinstance(annotation, type) and issubclass(annotation, base)


def _pydantic_schema_for(annotation: Any) -> dict[str, Any] | None:
    if not _is_pydantic_model(annotation):
        return None
    schema = annotation.model_json_schema()
    schema.pop("title", None)
    return schema


def _dataclass_schema_for(annotation: Any) -> dict[str, Any] | None:
    import dataclasses

    if not dataclasses.is_dataclass(annotation):
        return None

    properties: dict[str, Any] = {}
    required: list[str] = []

    for f in dataclasses.fields(annotation):
        properties[f.name] = {"type": _annotation_to_json_type(f.type)}
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _extract_coercions(
    sig: Signature,
    hints: dict[str, Any] | None = None,
) -> dict[str, Callable[[Any], Any]]:
    coercions: dict[str, Callable[[Any], Any]] = {}
    hints = hints or {}
    for param_name, param in sig.parameters.items():
        annotation = hints.get(param_name, param.annotation)
        if _is_pydantic_model(annotation):
            model = annotation
            coercions[param_name] = lambda value, m=model: (
                value if isinstance(value, m) else m.model_validate(value)
            )
            continue

        import dataclasses

        if dataclasses.is_dataclass(annotation):
            model = annotation
            coercions[param_name] = lambda value, m=model: (
                value if isinstance(value, m) else m(**value)
            )

    return coercions


def _apply_coercions(
    arguments: dict[str, Any],
    coercions: dict[str, Callable[[Any], Any]],
) -> dict[str, Any]:
    if not coercions:
        return arguments
    coerced = dict(arguments)
    for name, coerce in coercions.items():
        if name in coerced:
            coerced[name] = coerce(coerced[name])
    return coerced


def _tool_to_openai_schema(tool: RegisteredTool, *, strict: bool = False) -> dict[str, Any]:
    parameters = deepcopy(tool.parameters)
    if strict:
        properties = parameters.get("properties") or {}
        existing_required = set(parameters.get("required") or [])
        for prop_name, prop_schema in properties.items():
            if prop_name not in existing_required and isinstance(prop_schema, dict):
                prop_type = prop_schema.get("type")
                if isinstance(prop_type, str) and prop_type != "null":
                    prop_schema["type"] = [prop_type, "null"]
                elif isinstance(prop_type, list) and "null" not in prop_type:
                    prop_schema["type"] = [*prop_type, "null"]
        parameters["required"] = list(properties.keys())
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": True,
            },
        }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _tool_to_anthropic_schema(tool: RegisteredTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": deepcopy(tool.parameters),
    }


def _tool_to_gemini_function_declaration(tool: RegisteredTool) -> dict[str, Any]:
    declaration: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
    }
    properties = tool.parameters.get("properties") or {}
    if not properties:
        return declaration

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": deepcopy(properties),
    }
    required = tool.parameters.get("required") or []
    if required:
        parameters["required"] = list(required)
    declaration["parameters"] = parameters
    return declaration


def _tool_to_json_schema(tool: RegisteredTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "schema": deepcopy(tool.parameters),
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
