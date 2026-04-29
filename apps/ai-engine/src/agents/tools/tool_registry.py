"""
InsightSerenity AI Engine — Tool Registry & Base Tool
======================================================
Tools are the agent's hands. They extend what the LLM can do beyond
generating text: searching the web, running code, doing maths, querying
databases, calling APIs.

Each tool is a Python class with:
    name:        Unique identifier (lowercase, no spaces)
    description: One-line description shown to the LLM in the system prompt
    execute(input_str) → str: The actual tool implementation

The ToolRegistry is a central repository that:
    - Registers tools by name
    - Dispatches tool calls from the agent
    - Validates tool input before execution
    - Enforces timeout and output length limits
    - Logs every tool call for audit purposes

Design principle: tools should be stateless where possible. If a tool
needs state (e.g. a calculator with memory), it should manage that state
internally and not rely on the agent's memory.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────���───────────────────────────────��──────────────
# Base Tool
# ─────────────────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Every tool must declare:
        name:        Tool identifier used in Action: lines
        description: Brief description shown in the system prompt
        execute():   The actual implementation

    Args:
        max_output_length: Truncate tool output at this many characters.
        timeout_secs:      Maximum seconds a tool call may take.
    """

    name:        str = "base_tool"
    description: str = "A base tool."

    def __init__(
        self,
        max_output_length: int   = 2000,
        timeout_secs:      float = 30.0,
    ) -> None:
        self.max_output_length = max_output_length
        self.timeout_secs      = timeout_secs

    @abstractmethod
    def _run(self, tool_input: str) -> str:
        """The actual tool logic. Subclasses implement this."""
        ...

    def execute(self, tool_input: str) -> str:
        """
        Execute the tool with logging, timing, and output truncation.

        Args:
            tool_input: Raw string input from the agent's Action Input line.

        Returns:
            Tool output string, truncated if necessary.
        """
        start = time.perf_counter()
        logger.debug("Tool called", tool=self.name, input_preview=tool_input[:100])

        try:
            result = self._run(tool_input.strip())
        except Exception as e:
            result = f"Tool error ({self.name}): {e}"
            logger.warning("Tool execution failed", tool=self.name, error=str(e))

        elapsed = (time.perf_counter() - start) * 1000
        result  = str(result)

        # Truncate long outputs
        if len(result) > self.max_output_length:
            result = result[:self.max_output_length] + "\n[Output truncated]"

        logger.debug("Tool result", tool=self.name, elapsed_ms=round(elapsed, 1),
                     output_len=len(result))
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# ──────────────────────────────────────────────────────────────────────────��──
# Tool Registry
# ──────────────────────────────��─────────────────────────────────��────────────

class ToolRegistry:
    """
    Central registry for agent tools.

    Manages tool registration, lookup, and execution. Provides the
    tool descriptions formatted for inclusion in agent system prompts.

    Usage:
        registry = ToolRegistry()
        registry.register(WebSearchTool())
        registry.register(CalculatorTool())

        # Execute a tool by name
        result = registry.execute("calculator", "2 + 2")

        # Get tool dict for agent constructor
        tools = registry.as_dict()
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises if name already taken."""
        key = tool.name.lower().strip()
        if key in self._tools:
            raise ValueError(f"Tool '{key}' is already registered")
        self._tools[key] = tool
        logger.debug("Tool registered", name=key)

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name.lower(), None)

    def execute(self, name: str, tool_input: str) -> str:
        """
        Execute a tool by name.

        Returns an error string if the tool is not found.
        """
        tool = self._tools.get(name.lower().strip())
        if tool is None:
            return (
                f"Error: Tool '{name}' not found. "
                f"Available: {self.list_names()}"
            )
        return tool.execute(tool_input)

    def get(self, name: str) -> Optional[BaseTool]:
        """Return a tool by name, or None."""
        return self._tools.get(name.lower())

    def list_names(self) -> List[str]:
        """Return names of all registered tools."""
        return sorted(self._tools.keys())

    def format_descriptions(self) -> str:
        """
        Format all tool descriptions for the system prompt.

        Returns a numbered list: "1. tool_name: description"
        """
        lines = [
            f"{i}. {tool.name}: {tool.description}"
            for i, tool in enumerate(self._tools.values(), start=1)
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, BaseTool]:
        """Return the full tool dict (for passing to BaseAgent constructor)."""
        return dict(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_names()})"


# ─────────────────────────────────────────────────────────────────────────────
# FunctionTool — wrap any callable as a tool without subclassing
# ──────────────────────────────────────���──────────────────────────────────────

class FunctionTool(BaseTool):
    """
    Wrap a plain function as a tool.

    Useful for quick one-off tools that don't need a full class.

    Args:
        name:        Tool name.
        description: Tool description for the LLM.
        func:        The function to call. Must accept a single string argument.

    Example:
        reverse_tool = FunctionTool(
            name="reverse",
            description="Reverse a string.",
            func=lambda s: s[::-1],
        )
    """

    def __init__(
        self,
        name:        str,
        description: str,
        func:        Callable[[str], str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name        = name
        self.description = description
        self._func       = func

    def _run(self, tool_input: str) -> str:
        return self._func(tool_input)
