"""Gemini function-calling BMI agent.

Install: pip install google-genai
Run: GOOGLE_API_KEY=... python examples/gemini_agent.py
    Or uv run --env-file .env -- python examples/gemini_agent.py
Optional: GEMINI_MODEL=gemini-3-flash-preview
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable, Literal

from baretools import ToolRegistry, format_tool_results, parse_tool_calls, tool

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

USER_PROMPT = (
    "You are a careful health assistant. Use tools instead of mental math. "
    "For imperial units, convert them first, then compute BMI, then call "
    "bmi_category, and only then answer this question: I weigh 180 lb and I'm "
    "5 ft 11 in. Am I healthy?"
)


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


@TRACE_OP()
def run_agent() -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    registry = build_registry()
    declarations = registry.get_schemas("gemini")[0]["functionDeclarations"]
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=USER_PROMPT)])]

    with TRACE_ATTRIBUTES({"provider": "gemini", "example": "bmi"}):
        for _ in range(6):
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            tool_calls = parse_tool_calls(response, "gemini")
            if not tool_calls:
                return response.text or ""

            print("assistant requested", [call["name"] for call in tool_calls])
            results = registry.execute(tool_calls, parallel=True)
            contents.append(response.candidates[0].content)

            for result in results:
                print("tool result", result["tool_name"], result["output"])
            parts = [
                types.Part.from_function_response(**item)
                for item in format_tool_results(results, "gemini")
            ]
            contents.append(types.Content(role="user", parts=parts))

    raise RuntimeError("Agent loop exceeded max iterations")


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("Set GOOGLE_API_KEY before running this example")
    print(run_agent())
