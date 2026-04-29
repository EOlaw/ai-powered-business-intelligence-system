"""
Unit tests for Phase 9 — Agentic AI

Coverage:
    TestToolRegistry         — register, dispatch, duplicate raises, unknown returns error
    TestCalculatorTool       — correct arithmetic, safe functions, injection blocked
    TestCodeExecutorTool     — print capture, dangerous import blocked, syntax error handled
    TestShortTermMemory      — add/get, truncation by TAIL, token counting
    TestLongTermMemory       — store/retrieve keyword fallback, size property
    TestRetrievalTool        — keyword search, empty KB message
    TestAgentStep            — dataclass construction
    TestChainOfThought       — zero_shot_cot prompt, few_shot_cot format
    TestSelfCritic           — positive critique short-circuits, result fields set
    TestAgentRegistry        — register, list, format_for_prompt
    TestAgentExecutor        — _parse_step extracts action/input/final answer
    TestMockAgentRun         — full run with mock generator returns AgentRun
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_register_and_execute(self):
        from src.agents import ToolRegistry, FunctionTool
        reg  = ToolRegistry()
        tool = FunctionTool("echo", "Echoes input", lambda s: f"Echo: {s}")
        reg.register(tool)
        result = reg.execute("echo", "hello")
        assert result == "Echo: hello"

    def test_duplicate_registration_raises(self):
        from src.agents import ToolRegistry, FunctionTool
        reg  = ToolRegistry()
        tool = FunctionTool("dup", "desc", lambda s: s)
        reg.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(tool)

    def test_unknown_tool_returns_error(self):
        from src.agents import ToolRegistry
        reg    = ToolRegistry()
        result = reg.execute("nonexistent", "input")
        assert "not found" in result.lower()

    def test_list_names(self):
        from src.agents import ToolRegistry, FunctionTool
        reg = ToolRegistry()
        for name in ["alpha", "beta", "gamma"]:
            reg.register(FunctionTool(name, "", lambda s: s))
        assert sorted(reg.list_names()) == ["alpha", "beta", "gamma"]

    def test_format_descriptions(self):
        from src.agents import ToolRegistry, FunctionTool
        reg = ToolRegistry()
        reg.register(FunctionTool("calc", "Does math", lambda s: s))
        desc = reg.format_descriptions()
        assert "calc" in desc
        assert "Does math" in desc

    def test_as_dict(self):
        from src.agents import ToolRegistry, FunctionTool
        reg  = ToolRegistry()
        tool = FunctionTool("t", "d", lambda s: s)
        reg.register(tool)
        d = reg.as_dict()
        assert "t" in d
        assert d["t"] is tool


# ─────────────────────────────────────────────────────────────────────────────
# Calculator Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculatorTool:

    def test_basic_arithmetic(self):
        from src.agents import CalculatorTool
        calc = CalculatorTool()
        assert calc.execute("2 + 2")           == "4"
        assert calc.execute("10 - 3")          == "7"
        assert calc.execute("3 * 4")           == "12"
        assert calc.execute("15 / 3")          == "5"
        assert calc.execute("2 ** 10")         == "1024"

    def test_math_functions(self):
        from src.agents import CalculatorTool
        import math
        calc = CalculatorTool()
        assert calc.execute("sqrt(144)")       == "12"
        assert calc.execute("abs(-7)")         == "7"
        assert float(calc.execute("sin(0)")) == pytest.approx(0.0, abs=1e-5)

    def test_constants(self):
        from src.agents import CalculatorTool
        import math
        calc   = CalculatorTool()
        result = float(calc.execute("pi"))
        assert abs(result - math.pi) < 1e-8

    def test_hat_operator_replaced(self):
        from src.agents import CalculatorTool
        calc = CalculatorTool()
        # LLMs sometimes write ^ instead of **
        assert calc.execute("2^8") == "256"

    def test_division_by_zero(self):
        from src.agents import CalculatorTool
        calc = CalculatorTool()
        result = calc.execute("1 / 0")
        assert "zero" in result.lower() or "error" in result.lower()

    def test_injection_blocked(self):
        from src.agents import CalculatorTool
        calc   = CalculatorTool()
        result = calc.execute("__import__('os').system('echo pwned')")
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_empty_expression(self):
        from src.agents import CalculatorTool
        calc = CalculatorTool()
        assert "error" in calc.execute("").lower()

    def test_complex_expression(self):
        from src.agents import CalculatorTool
        calc = CalculatorTool()
        result = float(calc.execute("(3 + 4) * (10 - 3) / 7"))
        assert result == pytest.approx(7.0, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Code Executor Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeExecutorTool:

    def test_print_captured(self):
        from src.agents import CodeExecutorTool
        tool   = CodeExecutorTool()
        result = tool.execute('print("hello world")')
        assert "hello world" in result

    def test_arithmetic_result(self):
        from src.agents import CodeExecutorTool
        tool   = CodeExecutorTool()
        result = tool.execute("result = 7 * 6\nprint(result)")
        assert "42" in result

    def test_os_import_blocked(self):
        from src.agents import CodeExecutorTool
        tool   = CodeExecutorTool()
        result = tool.execute("import os\nos.system('echo pwned')")
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_subprocess_blocked(self):
        from src.agents import CodeExecutorTool
        tool   = CodeExecutorTool()
        result = tool.execute("import subprocess")
        assert "not allowed" in result.lower()

    def test_syntax_error_handled(self):
        from src.agents import CodeExecutorTool
        tool   = CodeExecutorTool()
        result = tool.execute("def bad_function(:\n    pass")
        assert "syntax" in result.lower() or "error" in result.lower()

    def test_math_module_available(self):
        from src.agents import CodeExecutorTool
        tool   = CodeExecutorTool()
        result = tool.execute("import math\nprint(math.sqrt(256))")
        assert "16" in result

    def test_empty_code(self):
        from src.agents import CodeExecutorTool
        tool   = CodeExecutorTool()
        result = tool.execute("")
        assert "no code" in result.lower() or "error" in result.lower()

    def test_multiline_code(self):
        from src.agents import CodeExecutorTool
        tool = CodeExecutorTool()
        code = "\n".join([
            "total = 0",
            "for i in range(10):",
            "    total += i",
            "print(total)",
        ])
        result = tool.execute(code)
        assert "45" in result


# ─────────────────────────────────────────────────────────────────────────────
# Short-Term Memory
# ─────────────────────────────────────────────────────────────────────────────

class TestShortTermMemory:

    def test_add_and_get(self):
        from src.agents import ShortTermMemory
        mem = ShortTermMemory(max_tokens=512)
        mem.add("user",      "Hello!")
        mem.add("assistant", "Hi there!")
        ctx = mem.get_context()
        assert len(ctx) == 2

    def test_system_message_always_present(self):
        from src.agents import ShortTermMemory
        mem = ShortTermMemory(max_tokens=512)
        mem.set_system("You are helpful.")
        mem.add("user", "Hello")
        ctx = mem.get_context()
        assert ctx[0].role == "system"
        assert ctx[0].content == "You are helpful."

    def test_tail_truncation_removes_oldest(self):
        from src.agents import ShortTermMemory, TruncationStrategy
        # Use very small budget to force truncation
        mem = ShortTermMemory(max_tokens=30, strategy=TruncationStrategy.TAIL,
                              reserve_for_output=0)
        for i in range(10):
            mem.add("user", f"Message {i} with some text padding to count tokens")

        # After truncation, the oldest messages should be gone
        ctx   = mem.get_context()
        texts = [m.content for m in ctx]
        assert not any("Message 0" in t for t in texts), "Oldest message should be truncated"

    def test_clear_removes_messages(self):
        from src.agents import ShortTermMemory
        mem = ShortTermMemory()
        mem.set_system("System.")
        mem.add("user", "Hello")
        mem.add("assistant", "Hi")
        mem.clear()
        assert len(mem) == 0

    def test_context_string_includes_roles(self):
        from src.agents import ShortTermMemory
        mem = ShortTermMemory()
        mem.add("user",      "What is 2+2?")
        mem.add("assistant", "4")
        ctx_str = mem.get_context_string()
        assert "USER" in ctx_str or "user" in ctx_str
        assert "ASSISTANT" in ctx_str or "assistant" in ctx_str

    def test_token_usage_dict(self):
        from src.agents import ShortTermMemory
        mem = ShortTermMemory(max_tokens=1000)
        mem.add("user", "Hello world")
        usage = mem.token_usage()
        assert "total" in usage
        assert "budget" in usage
        assert usage["total"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Long-Term Memory (no FAISS, keyword fallback)
# ─────────────────────────────────────────────────────────────────────────────

class TestLongTermMemory:

    def test_store_increases_size(self):
        from src.agents import LongTermMemory
        mem = LongTermMemory()
        assert mem.size == 0
        mem.store("The capital of France is Paris.")
        assert mem.size == 1

    def test_retrieve_keyword_finds_relevant(self):
        from src.agents import LongTermMemory
        mem = LongTermMemory()
        mem.store("The capital of France is Paris.")
        mem.store("Python is a programming language.")
        mem.store("The Eiffel Tower is in Paris, France.")
        result = mem.retrieve("France Paris capital")
        assert "Paris" in result or "France" in result

    def test_retrieve_empty_returns_empty_string(self):
        from src.agents import LongTermMemory
        mem = LongTermMemory()
        assert mem.retrieve("anything") == ""

    def test_store_fact_and_episode(self):
        from src.agents import LongTermMemory
        mem = LongTermMemory()
        mem.store_fact("Water freezes at 0°C")
        mem.store_episode("Find boiling point of water", "100°C")
        assert mem.size == 2

    def test_clear_resets_size(self):
        from src.agents import LongTermMemory
        mem = LongTermMemory()
        mem.store("Some fact")
        mem.store("Another fact")
        mem.clear()
        assert mem.size == 0


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalTool:

    def test_empty_kb_returns_empty_message(self):
        from src.agents import RetrievalTool
        tool   = RetrievalTool()
        result = tool.execute("any query")
        assert "empty" in result.lower() or "no documents" in result.lower()

    def test_keyword_retrieval_finds_match(self):
        from src.agents import RetrievalTool
        tool = RetrievalTool()
        tool.add_documents([
            "Machine learning uses data to train models.",
            "Deep learning is a subset of machine learning.",
            "Cooking requires fresh ingredients.",
        ])
        result = tool.execute("machine learning training")
        assert "machine learning" in result.lower() or "learning" in result.lower()

    def test_output_truncated_to_max(self):
        from src.agents import RetrievalTool
        tool = RetrievalTool()
        long_doc = "word " * 1000
        tool.add_documents([long_doc])
        result = tool.execute("word")
        assert len(result) <= tool.max_output_length + 100


# ─────────────────────────────────────────────────────────────────────────────
# Agent Step and Chain of Thought
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentStep:

    def test_default_fields(self):
        from src.agents import AgentStep
        step = AgentStep(thought="I need to search for X")
        assert step.action      is None
        assert step.observation is None
        assert step.is_final    is False

    def test_final_step(self):
        from src.agents import AgentStep
        step = AgentStep(thought="Done", is_final=True, final_answer="42")
        assert step.is_final
        assert step.final_answer == "42"


class TestChainOfThought:

    def test_zero_shot_cot_contains_trigger(self):
        from src.agents import zero_shot_cot
        prompt = zero_shot_cot("What is 17 * 24?")
        assert "step by step" in prompt.lower() or "think" in prompt.lower()

    def test_zero_shot_cot_contains_question(self):
        from src.agents import zero_shot_cot
        question = "What is the capital of Japan?"
        prompt   = zero_shot_cot(question)
        assert question in prompt

    def test_few_shot_cot_contains_examples(self):
        from src.agents import few_shot_cot
        examples = [
            {"question": "2+2?", "reasoning": "2+2=4", "answer": "4"},
        ]
        prompt = few_shot_cot("3+3?", examples)
        assert "2+2" in prompt
        assert "3+3" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Self-Critic
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfCritic:

    def _make_mock_generator(self, response: str) -> Any:
        gen = MagicMock()
        gen.generate = MagicMock(return_value=response)
        return gen

    def test_positive_critique_returns_original(self):
        from src.agents import SelfCritic
        gen    = self._make_mock_generator("the answer is correct and complete")
        critic = SelfCritic(generator=gen)
        result = critic.critique("What is 2+2?", "4")
        assert result.improved is False
        assert result.improved_answer == "4"

    def test_negative_critique_generates_improvement(self):
        from src.agents import SelfCritic
        gen = MagicMock()
        # First call: critique; second call: improvement
        gen.generate = MagicMock(side_effect=[
            "The answer is incomplete — missing the explanation.",
            "2+2 equals 4 because when you add 2 units to 2 units you get 4 units.",
        ])
        critic = SelfCritic(generator=gen)
        result = critic.critique("What is 2+2?", "4")
        assert result.improved is True
        assert "4" in result.improved_answer

    def test_critique_result_fields(self):
        from src.agents import SelfCritic
        gen    = self._make_mock_generator("the answer looks good already")
        critic = SelfCritic(generator=gen)
        result = critic.critique("Q", "A")
        assert hasattr(result, "original_answer")
        assert hasattr(result, "critique")
        assert hasattr(result, "improved_answer")
        assert hasattr(result, "improved")


# ─────────────────────────────────────────────────────────────────────────────
# Agent Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentRegistry:

    def test_register_and_get(self):
        from src.agents.multi_agent.orchestrator import AgentRegistry
        registry = AgentRegistry()
        mock_agent = MagicMock()
        registry.register("research", mock_agent, "Searches the web")
        assert registry.get("research") is mock_agent

    def test_case_insensitive_lookup(self):
        from src.agents.multi_agent.orchestrator import AgentRegistry
        registry = AgentRegistry()
        registry.register("CODE", MagicMock(), "Runs code")
        assert registry.get("code") is not None
        assert registry.get("CODE") is not None

    def test_list_agents_returns_descriptions(self):
        from src.agents.multi_agent.orchestrator import AgentRegistry
        registry = AgentRegistry()
        registry.register("math", MagicMock(), "Solves maths")
        agents = registry.list_agents()
        assert "math" in agents
        assert agents["math"] == "Solves maths"

    def test_format_for_prompt(self):
        from src.agents.multi_agent.orchestrator import AgentRegistry
        registry = AgentRegistry()
        registry.register("writer", MagicMock(), "Writes text")
        prompt = registry.format_for_prompt()
        assert "writer" in prompt
        assert "Writes text" in prompt

    def test_len(self):
        from src.agents.multi_agent.orchestrator import AgentRegistry
        registry = AgentRegistry()
        assert len(registry) == 0
        registry.register("a", MagicMock(), "desc a")
        registry.register("b", MagicMock(), "desc b")
        assert len(registry) == 2


# ─────────────────────────────────────────────────────────────────────────────
# AgentExecutor (parse_step and run with mocked generator)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentExecutor:

    def test_parse_step_extracts_action(self):
        from src.agents import AgentExecutor, AgentConfig
        gen  = MagicMock()
        exec_agent = AgentExecutor(generator=gen, tools={}, config=AgentConfig())
        step = exec_agent._parse_step(
            " I need to calculate something.\nAction: calculator\nAction Input: 3 * 7"
        )
        assert step.action      == "calculator"
        assert step.action_input == "3 * 7"
        assert step.is_final    is False
        assert "calculate" in step.thought.lower()

    def test_parse_step_extracts_final_answer(self):
        from src.agents import AgentExecutor, AgentConfig
        gen  = MagicMock()
        exec_agent = AgentExecutor(generator=gen, tools={}, config=AgentConfig())
        step = exec_agent._parse_step(
            " I know the answer now.\nFinal Answer: The result is 21."
        )
        assert step.is_final    is True
        assert "21" in (step.final_answer or "")

    def test_run_with_mock_generator_returns_agentrun(self):
        from src.agents import AgentExecutor, AgentConfig, AgentRun, CalculatorTool
        gen = MagicMock()
        # Simulate LLM producing a final answer on the first call
        gen.generate = MagicMock(return_value=(
            " The answer is 42.\nFinal Answer: 42"
        ))
        tools = {"calculator": CalculatorTool()}
        agent = AgentExecutor(
            generator=gen,
            tools=tools,
            config=AgentConfig(max_steps=3, verbose=False),
        )
        run = agent.run("What is 6 times 7?")
        assert isinstance(run, AgentRun)
        assert run.final_answer is not None

    def test_run_uses_calculator_tool(self):
        from src.agents import AgentExecutor, AgentConfig, CalculatorTool
        gen = MagicMock()
        call_count = [0]

        def mock_generate(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: use calculator
                return " I need to calculate.\nAction: calculator\nAction Input: 6 * 7"
            else:
                # Second call: final answer
                return " The calculation returned 42.\nFinal Answer: 42"

        gen.generate = MagicMock(side_effect=mock_generate)
        tools = {"calculator": CalculatorTool()}
        agent = AgentExecutor(
            generator=gen,
            tools=tools,
            config=AgentConfig(max_steps=5, verbose=False),
        )
        run = agent.run("What is 6 times 7?")
        assert run.success
        assert "42" in (run.final_answer or "")

    def test_run_handles_unknown_tool_gracefully(self):
        from src.agents import AgentExecutor, AgentConfig, AgentRun
        gen = MagicMock()
        call_count = [0]

        def mock_generate(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return " Let me use the magic tool.\nAction: magic_tool\nAction Input: test"
            return " I see magic_tool is not available. Final Answer: Unavailable"

        gen.generate = MagicMock(side_effect=mock_generate)
        agent = AgentExecutor(
            generator=gen,
            tools={},
            config=AgentConfig(max_steps=3, verbose=False),
        )
        run = agent.run("Do something magic")
        # Should complete without exception
        assert isinstance(run, AgentRun)
