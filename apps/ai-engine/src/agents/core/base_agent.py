"""
InsightSerenity AI Engine — Base Agent
========================================
The agent is an autonomous entity that perceives its environment (through
observations), reasons about what to do next (using our LLM), and acts
(by calling tools or producing output).

Agent = LLM + Tools + Memory + Planning Strategy

The base class implements the fundamental run loop:
    1. Receive a task from the user
    2. Build a prompt (system + memory + task + conversation history)
    3. Call the LLM to get the next thought/action
    4. Parse the LLM output to extract tool calls
    5. Execute tool calls and collect observations
    6. Add thought + action + observation to memory
    7. Repeat until a final answer is produced or max_steps reached

This is the ReAct pattern (Yao et al., 2022): Reason + Act interleaved.

The loop terminates when:
    - The LLM produces a "Final Answer:" response
    - max_steps is reached (safety limit)
    - A tool raises an exception marked as terminal

Agents use only our own LLM (TextGenerator) — no external API calls.
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """One step in an agent's reasoning process."""
    thought:     str
    action:      Optional[str]    = None   # Tool name called (None if final answer)
    action_input: Optional[str]   = None   # Input passed to the tool
    observation: Optional[str]    = None   # Tool output
    is_final:    bool             = False
    final_answer: Optional[str]   = None
    elapsed_ms:  float            = 0.0


@dataclass
class AgentRun:
    """
    Complete record of one agent execution.
    Stores every step, the final answer, and performance metadata.
    """
    task:         str
    steps:        List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str]   = None
    success:      bool            = False
    total_steps:  int             = 0
    elapsed_secs: float           = 0.0
    error:        Optional[str]   = None


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    max_steps:       int   = 10       # Maximum reasoning steps before giving up
    max_tokens:      int   = 512      # Max tokens per LLM call
    temperature:     float = 0.0      # 0 = deterministic (greedy) for tool use
    stop_sequences:  List[str] = field(default_factory=lambda: [
        "Observation:", "Human:", "<|end_turn|>"
    ])
    verbose:         bool  = True     # Log each step
    return_steps:    bool  = True     # Include all steps in the result


# ─────────────────────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base class for all InsightSerenity agents.

    Concrete agents must implement:
        _build_prompt(task, history) → str
        _parse_step(llm_output) → AgentStep
        _should_stop(step) → bool

    The base class handles:
        - The run loop
        - Tool dispatch
        - Memory updates
        - Logging
        - Error handling

    Args:
        generator:   TextGenerator instance (our LLM).
        tools:       Dict of tool_name → tool callable.
        memory:      Optional memory object with read/write interface.
        config:      AgentConfig.
    """

    def __init__(
        self,
        generator:  Any,   # TextGenerator
        tools:      Optional[Dict[str, Any]] = None,
        memory:     Optional[Any]            = None,
        config:     Optional[AgentConfig]    = None,
    ) -> None:
        self.generator = generator
        self.tools     = tools or {}
        self.memory    = memory
        self.config    = config or AgentConfig()

    # ── Abstract methods ───────────────────────────────────────────────────────

    @abstractmethod
    def _build_prompt(self, task: str, history: List[AgentStep]) -> str:
        """
        Construct the full prompt to send to the LLM.

        Args:
            task:    The user's task/question.
            history: Steps taken so far in this run.

        Returns:
            Complete prompt string.
        """
        ...

    @abstractmethod
    def _parse_step(self, llm_output: str) -> AgentStep:
        """
        Parse the LLM's text output into a structured AgentStep.

        Args:
            llm_output: Raw text from the LLM.

        Returns:
            AgentStep with thought, action, action_input, is_final populated.
        """
        ...

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, task: str) -> AgentRun:
        """
        Execute the agent on a task and return the full run record.

        Args:
            task: Natural language task description.

        Returns:
            AgentRun with the final answer and all intermediate steps.
        """
        run      = AgentRun(task=task)
        start    = time.perf_counter()
        history: List[AgentStep] = []

        if self.config.verbose:
            logger.info("Agent run started", task=task[:100])

        # Load relevant long-term memory context
        memory_context = self._retrieve_memory(task)

        for step_num in range(self.config.max_steps):
            step_start = time.perf_counter()

            try:
                # Build prompt including memory context and history
                prompt = self._build_prompt(
                    task=task,
                    history=history,
                    memory_context=memory_context,
                )

                # Query the LLM
                llm_output = self.generator.generate(
                    prompt=prompt,
                    max_new_tokens=self.config.max_tokens,
                    strategy="greedy",
                    temperature=self.config.temperature,
                    stop_strings=self.config.stop_sequences,
                )

                # Parse LLM output into a step
                step             = self._parse_step(llm_output)
                step.elapsed_ms  = (time.perf_counter() - step_start) * 1000

                if self.config.verbose:
                    self._log_step(step_num + 1, step)

                # Execute tool if action was requested
                if step.action and not step.is_final:
                    step.observation = self._execute_tool(step.action, step.action_input)

                history.append(step)
                run.steps.append(step)

                # Save this step to long-term memory
                self._store_memory(task, step)

                # Check for termination
                if step.is_final:
                    run.final_answer = step.final_answer
                    run.success      = True
                    break

            except Exception as e:
                error_msg = f"Step {step_num + 1} failed: {e}"
                logger.error("Agent step error", error=str(e), step=step_num + 1)
                run.error = error_msg
                break

        # Fallback: use last observation as answer if no Final Answer was produced
        if not run.success and history:
            last_with_obs = next(
                (s for s in reversed(history) if s.observation), None
            )
            if last_with_obs:
                run.final_answer = last_with_obs.observation
                run.success      = True

        run.total_steps  = len(history)
        run.elapsed_secs = time.perf_counter() - start

        if self.config.verbose:
            logger.info(
                "Agent run complete",
                steps=run.total_steps,
                success=run.success,
                elapsed=round(run.elapsed_secs, 2),
            )

        return run

    # ── Tool dispatch ──────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_input: Optional[str]) -> str:
        """
        Look up and execute a named tool.

        Args:
            tool_name:  Name of the tool to call.
            tool_input: Input string for the tool.

        Returns:
            Tool output as a string.
        """
        tool = self.tools.get(tool_name.strip().lower())
        if tool is None:
            available = list(self.tools.keys())
            return (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {available}"
            )

        try:
            result = tool.execute(tool_input or "")
            return str(result)[:2000]   # Truncate very long tool outputs
        except Exception as e:
            return f"Tool execution error: {e}"

    # ── Memory ─────────────────────────────────────────────────────────────────

    def _retrieve_memory(self, query: str) -> str:
        """Retrieve relevant long-term memory for the current task."""
        if self.memory is None:
            return ""
        try:
            return self.memory.retrieve(query, top_k=3)
        except Exception:
            return ""

    def _store_memory(self, task: str, step: AgentStep) -> None:
        """Store a completed step in long-term memory."""
        if self.memory is None or not step.observation:
            return
        try:
            text = f"Task: {task}\nThought: {step.thought}\nObservation: {step.observation}"
            self.memory.store(text)
        except Exception:
            pass

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_step(self, step_num: int, step: AgentStep) -> None:
        if step.is_final:
            logger.info("Agent final answer", step=step_num, answer=step.final_answer[:100] if step.final_answer else "")
        else:
            logger.info(
                "Agent step",
                step=step_num,
                action=step.action,
                input_preview=(step.action_input or "")[:80],
            )

    # ── Convenience ────────────────────────────────────────────────────────────

    def add_tool(self, tool: Any) -> None:
        """Register a tool with this agent."""
        self.tools[tool.name.lower()] = tool

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self.tools.keys())

    def _build_prompt(self, task: str, history: List[AgentStep], memory_context: str = "") -> str:
        """Default implementation — subclasses should override for their planning strategy."""
        raise NotImplementedError
