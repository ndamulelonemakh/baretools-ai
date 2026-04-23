"""Gemini function-calling BMI agent.

Install: pip install google-genai
Run: GOOGLE_API_KEY=... python examples/gemini_agent.py
Optional: GEMINI_MODEL=gemini-2.5-flash
"""

from __future__ import annotations

import os
from typing import Any, Literal

from baretools import ToolRegistry, tool


USER_PROMPT = (
    "You are a careful health assistant. Use tools instead of mental math. "
    "For imperial units, convert them first, then compute BMI, then call bmi_category, "
    "and only then answer this question: I weigh 180 lb and I'm 5 ft 11 in. Am I healthy?"
)


@tool
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
def compute_bmi(weight_kg: float, height_m: float) -> dict:
    bmi = weight_kg / (height_m * height_m)
    return {"bmi": round(bmi, 1)}


@tool
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


def _function_calls(response: Any) -> list[Any]:
    if getattr(response, "function_calls", None):
        return list(response.function_calls)
    calls = []
    for part in response.candidates[0].content.parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            calls.append(function_call)
    return calls


def run_agent() -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    registry = build_registry()
    declarations = registry.get_schemas("gemini")[0]["functionDeclarations"]
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents: list[Any] = [
        types.Content(role="user", parts=[types.Part(text=USER_PROMPT)])
    ]

    for _ in range(6):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        function_calls = _function_calls(response)
        if not function_calls:
            return response.text or ""

        tool_calls = [
            {"id": call.id, "name": call.name, "arguments": dict(call.args)}
            for call in function_calls
        ]
        print("assistant requested", [call["name"] for call in tool_calls])
        results = registry.execute(tool_calls, parallel=True)
        contents.append(response.candidates[0].content)
        parts = []
        for result in results:
            print("tool result", result["tool_name"], result["output"])
            payload = {"result": result["output"]}
            if result["error"] is not None:
                payload = {"error": result["error"]}
            parts.append(
                types.Part.from_function_response(
                    id=result["tool_call_id"],
                    name=result["tool_name"],
                    response=payload,
                )
            )
        contents.append(types.Content(role="user", parts=parts))

    raise RuntimeError("Agent loop exceeded max iterations")


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("Set GOOGLE_API_KEY before running this example")
    print(run_agent())"""Gemini function-calling BMI agent.

Install: pip install google-genai baretools-ai
Run: GOOGLE_API_KEY=... uv run python examples/gemini_agent.py
"""

from __future__ import annotations

import os
from typing import Any, Literal

from baretools import ToolRegistry, tool


SYSTEM_PROMPT = (
    "You are a precise BMI assistant. Use tools instead of mental math. "
    "If the user gives imperial units, convert them first. "
    "Call bmi_category before writing the final answer."
)

USER_PROMPT = "I weigh 180 lb and I'm 5 ft 11 in. Am I healthy?"


@tool
def to_metric(value: float, unit: Literal["kg", "lb", "m", "cm", "ft", "in"]) -> dict:
    """Convert one imperial or metric value into a normalized metric value."""
    if unit == "kg":
        return {"value": value, "unit": "kg"}
    if unit == "lb":
        return {"value": round(value * 0.45359237, 4), "unit": "kg"}
    if unit == "m":
        return {"value": value, "unit": "m"}
    if unit == "cm":
        return {"value": round(value / 100, 4), "unit": "m"}
    if unit == "ft":
        return {"value": round(value * 0.3048, 4), "unit": "m"}
    return {"value": round(value * 0.0254, 4), "unit": "m"}


@tool
def compute_bmi(weight_kg: float, height_m: float) -> float:
    """Compute BMI from metric weight and height."""
    return round(weight_kg / (height_m**2), 2)


@tool
def bmi_category(bmi: float) -> str:
    """Classify a BMI value using standard adult thresholds."""
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(to_metric)
    registry.register(compute_bmi)
    registry.register(bmi_category)
    return registry


def _tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
    calls = list(getattr(response, "function_calls", []) or [])
    return [
        {
            "id": getattr(call, "id", None),
            "name": call.name,
            "arguments": dict(call.args or {}),
        }
        for call in calls
    ]


def run_agent(
    user_prompt: str,
    model: str = "gemini-2.5-flash",
    max_iters: int = 6,
) -> str:
    from google import genai
    from google.genai import types

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY before running this example.")

    client = genai.Client()
    registry = build_registry()
    declarations = registry.get_schemas("gemini")[0]["functionDeclarations"]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=declarations)],
    )
    contents: list[Any] = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]

    for _ in range(max_iters):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        tool_calls = _tool_calls_from_response(response)
        if not tool_calls:
            return response.text or ""

        contents.append(response.candidates[0].content)
        results = registry.execute(tool_calls, parallel=True)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        id=result["tool_call_id"],
                        name=result["tool_name"],
                        response=(
                            {"result": result["output"]}
                            if result["error"] is None
                            else {"error": result["error"]}
                        ),
                    )
                    for result in results
                ],
            )
        )

    raise RuntimeError("Agent exceeded max_iters without producing a final answer.")


if __name__ == "__main__":
    print(run_agent(USER_PROMPT))