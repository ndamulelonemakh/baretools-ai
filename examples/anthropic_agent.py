"""Anthropic tool-use BMI agent.

Install: pip install anthropic
Run: ANTHROPIC_API_KEY=... python examples/anthropic_agent.py
Optional: ANTHROPIC_MODEL=claude-sonnet-4-5
"""

from __future__ import annotations

import os
from typing import Any, Literal

from baretools import ToolRegistry, tool


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


def _assistant_content_blocks(message: Any) -> list[dict[str, Any]]:
    return message.model_dump(exclude_none=True)["content"]


def _tool_calls_from_response(message: Any) -> list[dict[str, Any]]:
    calls = []
    for block in message.content:
        if block.type == "tool_use":
            calls.append({"id": block.id, "name": block.name, "arguments": block.input})
    return calls


def _tool_results_message(results: list[dict[str, Any]]) -> dict[str, Any]:
    content = []
    for result in results:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": result["tool_call_id"],
            "content": str(result["output"] if result["error"] is None else result["error"]),
        }
        if result["error"] is not None:
            block["is_error"] = True
        content.append(block)
    return {"role": "user", "content": content}


def _final_text(message: Any) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text").strip()


def run_agent() -> str:
    import anthropic

    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    registry = build_registry()
    messages: list[dict[str, Any]] = [{"role": "user", "content": USER_PROMPT}]

    for _ in range(6):
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=registry.get_schemas("anthropic"),
        )
        tool_calls = _tool_calls_from_response(response)
        if not tool_calls:
            return _final_text(response)

        print("assistant requested", [call["name"] for call in tool_calls])
        results = registry.execute(tool_calls, parallel=True)
        messages.append({"role": "assistant", "content": _assistant_content_blocks(response)})
        for result in results:
            print("tool result", result["tool_name"], result["output"])
        messages.append(_tool_results_message(results))

    raise RuntimeError("Agent loop exceeded max iterations")


if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Set ANTHROPIC_API_KEY before running this example")
    print(run_agent())"""Anthropic tool-use BMI agent.

Install: pip install anthropic baretools-ai
Run: ANTHROPIC_API_KEY=... uv run python examples/anthropic_agent.py
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


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "arguments": dict(block.input),
                }
            )
    return calls


def _tool_results_message(results: list[dict[str, Any]]) -> dict[str, Any]:
    content = []
    for result in results:
        block = {
            "type": "tool_result",
            "tool_use_id": result["tool_call_id"],
            "content": str(result["output"]),
        }
        if result["error"] is not None:
            block["content"] = f"ERROR: {result['error']}"
            block["is_error"] = True
        content.append(block)
    return {"role": "user", "content": content}


def _text_from_response(response: Any) -> str:
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(parts).strip()


def run_agent(
    user_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_iters: int = 6,
) -> str:
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY before running this example.")

    client = anthropic.Anthropic()
    registry = build_registry()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

    for _ in range(max_iters):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=registry.get_schemas("anthropic"),
        )
        tool_calls = _tool_calls_from_response(response)
        if not tool_calls:
            return _text_from_response(response)

        messages.append({"role": "assistant", "content": [_dump(block) for block in response.content]})
        messages.append(_tool_results_message(registry.execute(tool_calls, parallel=True)))

    raise RuntimeError("Agent exceeded max_iters without producing a final answer.")


if __name__ == "__main__":
    print(run_agent(USER_PROMPT))