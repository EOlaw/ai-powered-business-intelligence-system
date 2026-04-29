"""
InsightSerenity AI Engine — Calculator Tool
============================================
Evaluates mathematical expressions safely without using Python's eval().

Why not just use eval()?
    eval("__import__('os').system('rm -rf /')") — never use eval() on
    untrusted input. The agent's tool input comes from LLM output which
    is untrusted (could be injected via adversarial prompts).

Instead, we use a custom expression parser that only supports:
    - Numbers (integers and floats)
    - Arithmetic operators: + - * / // % **
    - Parentheses for grouping
    - Built-in math functions: sin, cos, tan, sqrt, log, abs, ceil, floor
    - Constants: pi, e, inf

The parser uses Python's ast module to parse the expression into an
Abstract Syntax Tree and then evaluates only safe node types.
Anything not in the allowed set raises a ValueError.

This gives us the full power of mathematical expressions while being
completely safe against code injection.
"""

import ast
import math
from typing import Any, Dict

from src.agents.tools.tool_registry import BaseTool


# Allowed node types in the AST
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Call, ast.Name, ast.Load,
)

# Allowed function names
_ALLOWED_FUNCTIONS: Dict[str, Any] = {
    "sin":    math.sin,
    "cos":    math.cos,
    "tan":    math.tan,
    "asin":   math.asin,
    "acos":   math.acos,
    "atan":   math.atan,
    "atan2":  math.atan2,
    "sqrt":   math.sqrt,
    "log":    math.log,
    "log2":   math.log2,
    "log10":  math.log10,
    "exp":    math.exp,
    "abs":    abs,
    "ceil":   math.ceil,
    "floor":  math.floor,
    "round":  round,
    "pow":    pow,
    "max":    max,
    "min":    min,
}

# Allowed constants
_ALLOWED_NAMES: Dict[str, Any] = {
    "pi":  math.pi,
    "e":   math.e,
    "inf": math.inf,
    "tau": math.tau,
}


def _safe_eval(node: ast.AST) -> Any:
    """
    Recursively evaluate an AST node, allowing only safe constructs.

    Raises ValueError for any unsupported node type.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    elif isinstance(node, ast.Constant):   # Python 3.8+: numbers are Constant
        return node.value

    elif isinstance(node, ast.BinOp):
        left  = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op    = node.op

        if isinstance(op, ast.Add):       return left + right
        if isinstance(op, ast.Sub):       return left - right
        if isinstance(op, ast.Mult):      return left * right
        if isinstance(op, ast.Pow):
            # Prevent very large exponentiations (DoS protection)
            if abs(right) > 1000:
                raise ValueError("Exponent too large (max 1000)")
            return left ** right
        if isinstance(op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        if isinstance(op, ast.FloorDiv):
            if right == 0:
                raise ValueError("Division by zero")
            return left // right
        if isinstance(op, ast.Mod):       return left % right
        raise ValueError(f"Unsupported operator: {type(op).__name__}")

    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub): return -operand
        if isinstance(node.op, ast.UAdd): return +operand
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")

    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not allowed")
        args = [_safe_eval(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)

    elif isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Name '{node.id}' is not allowed")

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class CalculatorTool(BaseTool):
    """
    Safe mathematical expression evaluator.

    Evaluates arithmetic expressions and common math functions without
    using Python's eval(). Input is parsed as a restricted AST.

    Supports:
        - Arithmetic: +, -, *, /, //, %, **
        - Functions: sin, cos, sqrt, log, abs, ceil, floor, round, ...
        - Constants: pi, e, inf, tau
        - Parentheses for grouping

    Examples:
        "2 + 2"                 → "4"
        "sqrt(144)"             → "12.0"
        "sin(pi / 2)"           → "1.0"
        "2 ** 10"               → "1024"
        "(3 + 4) * (2 - 1)"     → "7"
    """

    name        = "calculator"
    description = "Evaluates a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, log, abs, ceil, floor, pi, e. Input: expression string."

    def _run(self, tool_input: str) -> str:
        """
        Evaluate the expression and return the result as a string.

        Args:
            tool_input: Mathematical expression string.

        Returns:
            Numeric result as a string, or an error message.
        """
        # Clean up common LLM formatting artifacts
        expr = tool_input.strip()
        expr = expr.replace("^", "**")          # ^ → ** (LLMs often write ^)
        expr = expr.replace("×", "*")           # × → *
        expr = expr.replace("÷", "/")           # ÷ → /
        expr = expr.strip("`'\"")               # Strip quote artifacts

        if not expr:
            return "Error: Empty expression"

        try:
            tree   = ast.parse(expr, mode="eval")
            result = _safe_eval(tree)
            # Format the result cleanly
            if isinstance(result, float):
                if result == int(result) and abs(result) < 1e15:
                    return str(int(result))
                return f"{result:.10g}"
            return str(result)
        except ValueError as e:
            return f"Calculation error: {e}"
        except SyntaxError:
            return f"Syntax error in expression: '{expr}'"
        except Exception as e:
            return f"Error: {e}"
