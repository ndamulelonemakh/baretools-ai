"""OpenAI tool-calling BMI agent.

Install: pip install openai
Run: OPENAI_API_KEY=... python examples/openai_agent.py
Optional: OPENAI_MODEL=gpt-4.1
"""

from __future__ import annotations

import os
from typing import Literal

from baretools import ToolRegistry, parse_tool_calls, tool


SYSTEM_PROMPT = (
    "You are a careful health assistant. Use tools instead of mental math. "
    "For imperial units, convert them first, then compute BMI, then call bmi_category, "
    "and only then answer the user."
)
USER_PROMPT = "I weigh 180 lb and I'm 5 ft 11 in. Am I healthy?"


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


def run_agent() -> str:
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    registry = build_registry()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": str(result["output"]),
                }
            )

    raise RuntimeError("Agent loop exceeded max iterations")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this example")
    print(run_agent())"""OpenAI tool-calling BMI agent.

Install: pip install openai baretools-ai
Run: OPENAI_API_KEY=... uv run python examples/openai_agent.py
"""

from __future__ import annotations

import os
from typing import Literal

from baretools import ToolRegistry, format_tool_results, parse_tool_calls, tool


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


def run_agent(user_prompt: str, model: str = "gpt-4.1-mini", max_iters: int = 6) -> str:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    client = OpenAI()
    registry = build_registry()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=registry.get_schemas("openai", strict=True),
        )
        message = response.choices[0].message
        tool_calls = parse_tool_calls(message)
        if not tool_calls:
            return message.content or ""

        messages.append(message.model_dump(exclude_none=True))
        messages.extend(format_tool_results(registry.execute(tool_calls, parallel=True)))

    raise RuntimeError("Agent exceeded max_iters without producing a final answer.")


if __name__ == "__main__":
    print(run_agent(USER_PROMPT))