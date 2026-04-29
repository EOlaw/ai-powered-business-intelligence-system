"""
InsightSerenity AI Engine — Agent Executor
==========================================
The AgentExecutor wires together all components of the ReAct agent:
    LLM + Tools + Memory + Planning + Reflection

It is the production-ready, fully configured agent you instantiate and call.
Think of it as the "assembled vehicle" while base_agent.py is the chassis.

ReAct format (Yao et al., 2022):
    The agent produces text in a structured format that alternates between
    Thought (internal reasoning) and Action (tool calls):

    Thought: I need to find the population of France.
    Action: web_search
    Action Input: "France population 2024"
    Observation: France has approximately 68 million people.
    Thought: I now have the answer.
    Final Answer: France's population is approximately 68 million (2024).

This format is parsed by _parse_step() using regex to extract the
structured components from the LLM's free-form text output.

The executor also handles:
    - Tool descriptions injected into the system prompt
    - Memory retrieval formatted into context
    - Automatic reflection pass if the initial answer looks wrong
    - Step-level timing and token counting
"""

import re
from typing import Any, Dict, List, Optional

from src.agents.core.base_agent import BaseAgent, AgentConfig, AgentStep
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt template for the ReAct agent
# ─────────────────────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are InsightSerenity Agent, an autonomous AI assistant that solves tasks step by step.

You have access to the following tools:
{tool_descriptions}

To use a tool, respond in this EXACT format:
Thought: <your reasoning about what to do next>
Action: <exact tool name from the list above>
Action Input: <the input to the tool>

After seeing an Observation, continue thinking until you can provide a final answer:
Thought: <reasoning based on the observation>
... (repeat Thought/Action/Observation as needed)

When you have enough information to answer, respond EXACTLY like this:
Thought: I now know the final answer.
Final Answer: <your complete answer to the original question>

Rules:
- Always think before acting.
- Never make up information — only use what you learn from tools.
- If a tool fails, try a different approach.
- Be concise in your final answer.
{memory_section}"""

MEMORY_SECTION = """
Relevant memory from previous interactions:
{memory_context}"""


class AgentExecutor(BaseAgent):
    """
    Production ReAct agent executor.

    Fully configured agent that implements the ReAct reasoning loop,
    integrates with all tool types, uses long-term FAISS memory,
    and optionally performs a self-reflection pass on its final answer.

    Args:
        generator:        TextGenerator (our LLM).
        tools:            Dict of {tool_name: tool_instance}.
        memory:           Optional LongTermMemory instance.
        config:           AgentConfig.
        enable_reflection: Run a self-critique pass after initial answer.
    """

    def __init__(
        self,
        generator:         Any,
        tools:             Optional[Dict[str, Any]] = None,
        memory:            Optional[Any]            = None,
        config:            Optional[AgentConfig]    = None,
        enable_reflection: bool                     = False,
    ) -> None:
        super().__init__(generator=generator, tools=tools, memory=memory, config=config)
        self.enable_reflection = enable_reflection

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_prompt(
        self,
        task:           str,
        history:        List[AgentStep],
        memory_context: str = "",
    ) -> str:
        """
        Build the full ReAct prompt for the LLM.

        Includes:
        1. System prompt with tool descriptions
        2. Memory context (if available)
        3. User task
        4. Conversation history (interleaved Thought/Action/Observation)
        5. Prompt for the next step

        Args:
            task:           The original user task.
            history:        Steps completed so far.
            memory_context: Relevant text retrieved from long-term memory.

        Returns:
            Complete prompt string ready to send to the LLM.
        """
        # Build tool descriptions
        tool_descriptions = self._format_tool_descriptions()

        # Optional memory section
        memory_section = ""
        if memory_context.strip():
            memory_section = MEMORY_SECTION.format(memory_context=memory_context)

        system = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            memory_section=memory_section,
        )

        # User task
        prompt = f"{system}\n\nQuestion: {task}\n\n"

        # Append conversation history
        for step in history:
            prompt += f"Thought: {step.thought}\n"
            if step.action and not step.is_final:
                prompt += f"Action: {step.action}\n"
                prompt += f"Action Input: {step.action_input or ''}\n"
                if step.observation:
                    prompt += f"Observation: {step.observation}\n"
            elif step.is_final and step.final_answer:
                prompt += f"Final Answer: {step.final_answer}\n"

        # The LLM continues from here
        prompt += "Thought:"
        return prompt

    # ── Step parsing ────────────────────────────────────────────────────────────

    def _parse_step(self, llm_output: str) -> AgentStep:
        """
        Parse raw LLM text into a structured AgentStep.

        The LLM is prompted to produce "Thought:" and "Action:" etc.
        We use regex to extract each component. If "Final Answer:" is present,
        the step is marked as terminal.

        Handles:
            - Normal steps: Thought + Action + Action Input
            - Final answer steps: Thought + Final Answer
            - Malformed output: gracefully extract what we can

        Args:
            llm_output: Raw text produced by the LLM (after "Thought:").

        Returns:
            Structured AgentStep.
        """
        # The LLM output continues from where we left off (after "Thought:")
        # Prepend it back so regex works cleanly
        full_text = "Thought:" + llm_output

        thought   = ""
        action    = None
        action_in = None
        is_final  = False
        final_ans = None

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", full_text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for Final Answer
        final_match = re.search(r"Final Answer:\s*(.+?)$", full_text, re.DOTALL)
        if final_match:
            final_ans = final_match.group(1).strip()
            is_final  = True
        else:
            # Check for Action
            action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", full_text, re.DOTALL)
            input_match  = re.search(r"Action Input:\s*(.+?)(?=Observation:|$)", full_text, re.DOTALL)

            if action_match:
                action    = action_match.group(1).strip()
            if input_match:
                action_in = input_match.group(1).strip()

        # If neither action nor final answer, treat the whole thing as a final answer
        # (the model said something but didn't follow the format)
        if not action and not is_final and thought:
            final_ans = thought
            is_final  = True

        return AgentStep(
            thought=thought,
            action=action,
            action_input=action_in,
            is_final=is_final,
            final_answer=final_ans,
        )

    # ── Tool descriptions ──────────────────────────────────────────────────────

    def _format_tool_descriptions(self) -> str:
        """
        Format all registered tools as a numbered list for the system prompt.

        Each tool exposes a .name and .description attribute.
        """
        if not self.tools:
            return "No tools available."

        lines = []
        for i, (name, tool) in enumerate(self.tools.items(), start=1):
            desc = getattr(tool, "description", "No description.")
            lines.append(f"{i}. {name}: {desc}")

        return "\n".join(lines)

    # ── Reflection integration ─────────────────────────────────────────────────

    def run(self, task: str):
        """
        Execute the agent, optionally followed by a reflection pass.

        Overrides BaseAgent.run() to add post-run self-reflection.
        """
        result = super().run(task)

        if self.enable_reflection and result.success and result.final_answer:
            from src.agents.reflection.self_critic import SelfCritic
            try:
                critic   = SelfCritic(generator=self.generator)
                improved = critic.critique_and_improve(
                    task=task,
                    answer=result.final_answer,
                )
                if improved != result.final_answer:
                    result.final_answer = improved
                    logger.info("Answer improved by reflection")
            except Exception as e:
                logger.debug("Reflection step failed", error=str(e))

        return result
