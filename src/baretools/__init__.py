from importlib.metadata import PackageNotFoundError, version

from .core import (
    ProviderToolResult,
    ToolCall,
    ToolEvent,
    ToolRegistry,
    ToolResult,
    format_tool_results,
    parse_tool_calls,
    tool,
)

try:
    __version__ = version("baretools-ai")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "tool",
    "ToolRegistry",
    "ToolCall",
    "ToolResult",
    "ToolEvent",
    "ProviderToolResult",
    "parse_tool_calls",
    "format_tool_results",
    "__version__",
]
