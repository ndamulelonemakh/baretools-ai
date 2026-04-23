"""OpenAI tool-calling BMI agent.

Install: pip install openai
Run: OPENAI_API_KEY=... python examples/openai_agent.py
        Or uv run --env-file .env -- python examples/openai_agent.py
Optional: OPENAI_MODEL=gpt-4.1
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Any, Callable, Literal

from baretools import ToolRegistry, parse_tool_calls, tool

TraceDecorator = Callable[[Callable[..., Any]], Callable[..., Any]]


def _noop_trace_op(*_args: Any, **_kwargs: Any) -> TraceDecorator:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


def _noop_trace_attributes(_attrs: dict[str, Any]) -> Any:
    return nullcontext()


def _setup_weave() -> tuple[
    Callable[..., TraceDecorator],
    Callable[[dict[str, Any]], Any],
]:
    project = os.environ.get("WEAVE_PROJECT", "baretools-ai-examples")
    if not project:
        return _noop_trace_op, _noop_trace_attributes

    try:
        import weave
    except ImportError as exc:
        raise RuntimeError(
            "WEAVE_PROJECT is set, but `weave` is not installed. Run `pip install weave wandb`."
        ) from exc

    weave.init(project)
    return weave.op, weave.attributes


TRACE_OP, TRACE_ATTRIBUTES = _setup_weave()

SYSTEM_PROMPT = (
    "You are a careful health assistant. Use tools instead of mental math. "
    "For imperial units, convert them first, then compute BMI, then call "
    "bmi_category, and only then answer the user."
)
USER_PROMPT = "I weigh 180 lb and I'm 5 ft 11 in. Am I healthy?"


@tool
@TRACE_OP()
def to_metric(value: float, unit: Literal["kg", "lb", "m", "cm", "ft", "in"]) -> dict:
    factors = {
        "kg": (1.0, "kg"),
        "lb": (0.45359237, "kg"),
        "m": (1.0, "m"),
        "cm": (0.01, "m"),
        "ft": (0.3048, "m"),
        "in": (0.0254, "m"),
    }
    factor, target_unit = factors[unit]
    return {"value": round(value * factor, 4), "unit": target_unit}


@tool
@TRACE_OP()
def compute_bmi(weight_kg: float, height_m: float) -> dict:
    bmi = weight_kg / (height_m * height_m)
    return {"bmi": round(bmi, 1)}


@tool
@TRACE_OP()
def bmi_category(bmi: float) -> dict:
    if bmi < 18.5:
        return {"category": "underweight"}
    if bmi < 25:
        return {"category": "normal weight"}
    if bmi < 30:
        return {"category": "overweight"}
    return {"category": "obesity"}


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(to_metric)
    registry.register(compute_bmi)
    registry.register(bmi_category)
    return registry


def _content_for_model(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


@TRACE_OP()
def run_agent() -> str:
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    registry = build_registry()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    with TRACE_ATTRIBUTES({"provider": "openai", "example": "bmi"}):
        for _ in range(6):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=registry.get_schemas("openai", strict=True),
            )
            message = response.choices[0].message
            tool_calls = parse_tool_calls(message)
            if not tool_calls:
                return message.content or ""

            print("assistant requested", [call["name"] for call in tool_calls])
            results = registry.execute(tool_calls, parallel=True)
            messages.append(message.model_dump(exclude_none=True))
            for result in results:
                print("tool result", result["tool_name"], result["output"])
                content = result["output"]
                if result["error"] is not None:
                    content = {"error": result["error"]}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": _content_for_model(content),
                    }
                )

    raise RuntimeError("Agent loop exceeded max iterations")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this example")
    print(run_agent())
