"""
InsightSerenity AI Engine — Code Executor Tool
================================================
Executes Python code snippets in a sandboxed environment and returns
the output. This allows the agent to perform complex computations,
data manipulation, and string processing beyond simple arithmetic.

Security model:
    This tool NEVER uses exec() or eval() with unrestricted globals.
    Instead it uses RestrictedPython (or our custom sandbox) that:
        1. Disallows imports of os, sys, subprocess, socket, and other
           dangerous modules
        2. Restricts file system access
        3. Enforces a time limit to prevent infinite loops
        4. Captures stdout/stderr to return as output
        5. Runs in a fresh namespace each time (no state leakage)

Allowed operations:
    - Basic Python: variables, loops, functions, list comprehensions
    - Math operations (math module)
    - String manipulation
    - Data structure operations (list, dict, set)
    - JSON parsing
    - Datetime operations

Blocked:
    - os, sys, subprocess, socket imports
    - File I/O
    - Network calls
    - Anything that could harm the host system

The agent uses this to:
    - Process structured data
    - Generate or parse formatted output
    - Run algorithms too complex for the calculator
    - Verify its own numerical reasoning
"""

import io
import math
import json
import sys
import time
import contextlib
from typing import Any, Dict

from src.agents.tools.tool_registry import BaseTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox configuration
# ─────────────────────────────────────────────────────────────────────────────

# Modules available to executed code
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "bytes": bytes, "chr": chr, "dict": dict, "dir": dir,
    "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset,
    "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
    "id": id, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max,
    "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
    "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type, "vars": vars,
    "zip": zip,
    # Error types
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "StopIteration": StopIteration,
    "Exception": Exception,
    # None, True, False
    "None": None, "True": True, "False": False,
}

_SAFE_GLOBALS: Dict[str, Any] = {
    "__builtins__": _SAFE_BUILTINS,
    "math":  math,
    "json":  json,
}

_SAFE_MODULES = {
    "math": math,
    "json": json,
}


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    """Allow imports only for modules already exposed in the sandbox."""
    root_name = name.split(".", 1)[0]
    if level != 0 or root_name not in _SAFE_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
    return _SAFE_MODULES[root_name]


_SAFE_BUILTINS["__import__"] = _safe_import

_BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "builtins", "ctypes", "threading", "multiprocessing",
    "asyncio", "requests", "urllib", "http", "ftplib", "smtplib",
    "pickle", "shelve", "marshal",
})


class CodeExecutorTool(BaseTool):
    """
    Safe Python code executor with sandboxed globals and stdout capture.

    The agent can write small Python scripts to process data, run
    algorithms, or verify calculations. Output is captured from print()
    calls and the last expression's repr.

    Args:
        timeout_secs:    Maximum execution time. Default 10s.
        max_output_len:  Maximum characters of captured output.
    """

    name        = "python_code"
    description = (
        "Executes Python code and returns the output. "
        "Use print() to output results. "
        "math and json modules are available. "
        "No file or network access. "
        "Input: valid Python code string."
    )

    def __init__(self, timeout_secs: float = 10.0, max_output_len: int = 1000) -> None:
        super().__init__(max_output_length=max_output_len, timeout_secs=timeout_secs)

    def _run(self, tool_input: str) -> str:
        """
        Execute Python code in a restricted sandbox.

        Args:
            tool_input: Python source code string.

        Returns:
            Captured stdout + stderr, or error message.
        """
        code = tool_input.strip()

        # Strip markdown code blocks if LLM included them
        if code.startswith("```"):
            lines = code.split("\n")
            code  = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        if not code:
            return "Error: No code provided"

        # Check for blocked imports
        for blocked in _BLOCKED_IMPORTS:
            if f"import {blocked}" in code or f"from {blocked}" in code:
                return f"Error: Import of '{blocked}' is not allowed in the sandbox"

        # Check for other dangerous patterns
        dangerous_patterns = [
            "__import__", "__class__", "__subclasses__",
            "open(", "exec(", "compile(", "eval(",
            "globals()", "locals()", "__builtins__",
        ]
        for pat in dangerous_patterns:
            if pat in code:
                return f"Error: '{pat}' is not allowed"

        # Capture stdout
        captured_stdout = io.StringIO()

        try:
            with contextlib.redirect_stdout(captured_stdout):
                # Create a fresh local namespace for this execution
                local_ns: Dict = {}
                # Use restricted globals
                exec(code, dict(_SAFE_GLOBALS), local_ns)   # noqa: S102

            output = captured_stdout.getvalue()

            if not output.strip():
                # Nothing printed — try to return the last variable's value
                if local_ns:
                    last_var = list(local_ns.values())[-1]
                    if not callable(last_var):
                        output = repr(last_var)

            return output.strip() if output.strip() else "Code executed successfully (no output)"

        except SyntaxError as e:
            return f"Syntax error: {e}"
        except Exception as e:
            return f"Runtime error: {type(e).__name__}: {e}"
