"""
InsightSerenity AI Engine — Multi-Agent Orchestrator
=====================================================
Coordinates multiple specialised agents to solve tasks that are too complex
or too broad for a single agent. Each specialist agent has deep capability
in one domain; the orchestrator routes, decomposes, and integrates.

Architecture: Supervisor + Worker pattern
    Orchestrator (Supervisor) receives the user task, determines which
    specialist(s) should handle which sub-tasks, dispatches work, and
    assembles the final response.

    Worker agents (Specialists) are standard ReActAgents with domain-
    specific tools and system prompts:
        ResearchAgent   — web search, document retrieval
        CodeAgent       — code writing, execution, debugging
        MathAgent       — calculation, symbolic reasoning
        WriterAgent     — synthesis, summarisation, drafting

Message passing:
    Agents communicate via AgentMessage objects. The orchestrator
    sends tasks to workers and collects their results. Workers do not
    communicate directly with each other — all routing goes through
    the orchestrator (prevents circular dependencies and makes
    execution traces auditable).

Decomposition strategies:
    SEQUENTIAL:  Sub-tasks run one after another. Output of task N
                 is available as context to task N+1.
    PARALLEL:    Sub-tasks run independently (conceptually — single
                 process here, but the pattern supports async future).
    ADAPTIVE:    Orchestrator decides the decomposition dynamically
                 using the LLM after each step.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from src.agents.core.base_agent import AgentRun
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Message types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    """
    A message passed between agents or from orchestrator to worker.

    Attributes:
        sender:    Name of the sending agent (or "orchestrator").
        receiver:  Name of the target agent (or "orchestrator").
        task:      The sub-task or question being delegated.
        context:   Additional context from previous steps.
        result:    The agent's response (populated after execution).
    """
    sender:   str
    receiver: str
    task:     str
    context:  str             = ""
    result:   Optional[str]   = None


@dataclass
class OrchestratorPlan:
    """
    The orchestrator's decomposition of a task into sub-tasks.
    Each sub-task is assigned to a specific worker agent.
    """
    original_task: str
    sub_tasks:     List[Dict[str, str]] = field(default_factory=list)
    # Each dict: {"agent": agent_name, "task": sub_task_str, "context": ""}
    strategy:      str = "sequential"


# ─────────────────────────────────────────────────────────────────────────────
# Worker registry
# ─────────────────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Holds named worker agents and routes tasks to them.

    Each agent has a name, description (used by the orchestrator to decide
    which agent to call), and a run() method.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, agent: Any, description: str) -> None:
        """
        Register a worker agent.

        Args:
            name:        Agent identifier (e.g. "research", "code").
            agent:       Agent instance with a .run(task) → AgentRun method.
            description: One-line description of what this agent does best.
        """
        self._agents[name.lower()]       = agent
        self._descriptions[name.lower()] = description
        logger.debug("Agent registered", name=name)

    def get(self, name: str) -> Optional[Any]:
        return self._agents.get(name.lower())

    def list_agents(self) -> Dict[str, str]:
        """Return {name: description} for all registered agents."""
        return dict(self._descriptions)

    def format_for_prompt(self) -> str:
        """Format agent list for inclusion in the orchestrator's prompt."""
        lines = []
        for name, desc in self._descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._agents)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Orchestrator that decomposes complex tasks and routes sub-tasks
    to specialised worker agents.

    The orchestrator uses our LLM to:
        1. Analyse the task and determine if decomposition is needed
        2. Generate a decomposition plan with sub-tasks and agent assignments
        3. Execute sub-tasks in order (sequential) or dispatch all (parallel)
        4. Synthesise all results into a coherent final answer

    Args:
        generator:   TextGenerator (our LLM) — used for decomposition/synthesis.
        registry:    AgentRegistry containing all worker agents.
        max_workers: Maximum sub-tasks to dispatch in one run.
    """

    DECOMPOSE_PROMPT = """You are a task coordinator. Given a complex task and available specialist agents, decompose the task into sub-tasks and assign each to the best agent.

Available agents:
{agents}

Task: {task}

Respond with a numbered list of sub-tasks. For each sub-task:
Sub-task N:
Agent: <agent name>
Task: <specific sub-task description>

Only create sub-tasks if the task genuinely requires multiple specialists.
If one agent can handle the whole task, assign it to that agent as a single sub-task.

Sub-task 1:"""

    SYNTHESISE_PROMPT = """You are synthesising results from multiple specialist agents into a coherent final answer.

Original question: {task}

Results from specialists:
{results}

Write a complete, well-organised final answer that integrates all the information above:"""

    def __init__(
        self,
        generator: Any,
        registry:  AgentRegistry,
        max_workers: int = 5,
    ) -> None:
        self.generator   = generator
        self.registry    = registry
        self.max_workers = max_workers

    def run(self, task: str) -> str:
        """
        Execute a task using multi-agent orchestration.

        Args:
            task: The user's complete task or question.

        Returns:
            Final synthesised answer string.
        """
        logger.info("Orchestrator run started", task=task[:100], agents=len(self.registry))

        if len(self.registry) == 0:
            return "No agents registered. Cannot execute task."

        # Step 1: Decompose the task
        plan = self._decompose(task)
        logger.info("Task decomposed", n_subtasks=len(plan.sub_tasks))

        if not plan.sub_tasks:
            # Fallback: route the whole task to the first available agent
            first_agent_name = list(self.registry.list_agents().keys())[0]
            plan.sub_tasks = [{"agent": first_agent_name, "task": task, "context": ""}]

        # Step 2: Execute sub-tasks
        results: List[Dict[str, str]] = []
        accumulated_context = ""

        for i, sub_task in enumerate(plan.sub_tasks[:self.max_workers]):
            agent_name = sub_task.get("agent", "").lower()
            agent_task = sub_task.get("task", task)
            agent      = self.registry.get(agent_name)

            if agent is None:
                # Unknown agent — skip or log
                logger.warning("Unknown agent in plan", agent=agent_name)
                results.append({
                    "agent":  agent_name,
                    "task":   agent_task,
                    "result": f"[Agent '{agent_name}' not found]",
                })
                continue

            # Add accumulated context from previous steps
            full_task = agent_task
            if accumulated_context:
                full_task = f"Context from previous steps:\n{accumulated_context}\n\n{agent_task}"

            message = AgentMessage(
                sender="orchestrator",
                receiver=agent_name,
                task=full_task,
            )

            logger.info("Dispatching sub-task", agent=agent_name, task=agent_task[:80])
            message = self._dispatch(message, agent)

            results.append({
                "agent":  agent_name,
                "task":   agent_task,
                "result": message.result or "",
            })

            # Accumulate context for sequential execution
            if message.result:
                accumulated_context += f"\n[{agent_name}]: {message.result}"

        # Step 3: Synthesise results
        if len(results) == 1:
            # Single agent — no synthesis needed
            return results[0]["result"]

        return self._synthesise(task, results)

    def _decompose(self, task: str) -> OrchestratorPlan:
        """Use the LLM to decompose the task into agent-assigned sub-tasks."""
        agent_descriptions = self.registry.format_for_prompt()
        prompt = self.DECOMPOSE_PROMPT.format(
            agents=agent_descriptions,
            task=task,
        )

        raw = self.generator.generate(
            prompt=prompt,
            max_new_tokens=400,
            strategy="greedy",
        )

        # Parse the numbered sub-tasks
        plan = OrchestratorPlan(original_task=task)
        full_text = "Sub-task 1:" + raw

        import re
        blocks = re.split(r"Sub-task \d+:", full_text)
        for block in blocks[1:]:  # Skip the empty first element
            agent_match = re.search(r"Agent:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            task_match  = re.search(r"Task:\s*(.+?)(?:\n|$)", block, re.IGNORECASE | re.DOTALL)

            agent_name = agent_match.group(1).strip().lower() if agent_match else ""
            sub_task   = task_match.group(1).strip() if task_match else ""

            if agent_name and sub_task:
                plan.sub_tasks.append({"agent": agent_name, "task": sub_task, "context": ""})

        return plan

    def _dispatch(self, message: AgentMessage, agent: Any) -> AgentMessage:
        """Send a message to a worker agent and collect the result."""
        try:
            run: AgentRun = agent.run(message.task)
            message.result = run.final_answer or "No answer produced"
        except Exception as e:
            message.result = f"Agent error: {e}"
            logger.error("Worker agent failed", agent=message.receiver, error=str(e))
        return message

    def _synthesise(self, task: str, results: List[Dict[str, str]]) -> str:
        """Use the LLM to combine multiple agent results into one answer."""
        results_text = "\n\n".join(
            f"[{r['agent'].upper()}] — {r['task']}\n{r['result']}"
            for r in results
        )

        prompt = self.SYNTHESISE_PROMPT.format(
            task=task,
            results=results_text,
        )

        return self.generator.generate(
            prompt=prompt,
            max_new_tokens=600,
            strategy="greedy",
        ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build a standard multi-agent system
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_agent_system(generator: Any, search_url: Optional[str] = None) -> MultiAgentOrchestrator:
    """
    Build a ready-to-use multi-agent system with standard specialists.

    Creates four specialists:
        research: web search + retrieval
        code:     Python execution + calculation
        math:     Calculator + code executor
        writer:   Synthesis and writing (no tools, pure LLM)

    Args:
        generator:   TextGenerator (our LLM).
        search_url:  Optional search endpoint for the research agent.

    Returns:
        Configured MultiAgentOrchestrator.
    """
    from src.agents.planning.react import ReActAgent
    from src.agents.tools.tool_registry import ToolRegistry
    from src.agents.tools.web_search_tool import WebSearchTool
    from src.agents.tools.calculator_tool import CalculatorTool
    from src.agents.tools.code_executor_tool import CodeExecutorTool
    from src.agents.core.base_agent import AgentConfig

    registry = AgentRegistry()

    # Research agent — web search + retrieval
    research_registry = ToolRegistry()
    research_registry.register(WebSearchTool(search_endpoint=search_url))
    research_agent = ReActAgent(
        generator=generator,
        tools=research_registry.as_dict(),
        config=AgentConfig(max_steps=6),
    )
    registry.register("research", research_agent, "Searches the web and retrieves information")

    # Code agent — Python execution
    code_registry = ToolRegistry()
    code_registry.register(CodeExecutorTool())
    code_registry.register(CalculatorTool())
    code_agent = ReActAgent(
        generator=generator,
        tools=code_registry.as_dict(),
        config=AgentConfig(max_steps=6),
    )
    registry.register("code", code_agent, "Writes and executes Python code, solves programming tasks")

    # Math agent — calculation focused
    math_registry = ToolRegistry()
    math_registry.register(CalculatorTool())
    math_registry.register(CodeExecutorTool())
    math_agent = ReActAgent(
        generator=generator,
        tools=math_registry.as_dict(),
        config=AgentConfig(max_steps=5),
    )
    registry.register("math", math_agent, "Solves mathematical and quantitative problems")

    # Writer agent — synthesis (no tools, pure LLM generation)
    writer_agent = ReActAgent(
        generator=generator,
        tools={},
        config=AgentConfig(max_steps=3),
    )
    registry.register("writer", writer_agent, "Writes, edits, and synthesises information")

    return MultiAgentOrchestrator(generator=generator, registry=registry)
