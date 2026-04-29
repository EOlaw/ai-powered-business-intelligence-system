"""
InsightSerenity AI Engine — ReAct Planning Strategy
====================================================
ReAct (Reasoning + Acting) is the core planning strategy used by the
AgentExecutor. It interleaves reasoning (Thought) with action (tool calls)
in a structured loop.

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
           Yao et al., 2022. https://arxiv.org/abs/2210.03629

The ReAct loop:
    [Task]
    Thought: I need to find X.
    Action: web_search
    Action Input: "X query"
    Observation: X is 42.
    Thought: I know X is 42. Now I can compute Y.
    Action: calculator
    Action Input: 42 * 3
    Observation: 126
    Thought: I now know the answer.
    Final Answer: The answer is 126.

Key properties:
    1. Reasoning before acting: the Thought step prevents blind tool use
    2. Grounded reasoning: Observations from tools anchor the reasoning
    3. Trace interpretability: every step is logged and explainable
    4. Self-correction: if an observation shows an error, the next Thought
       can correct course

The ReAct strategy is implemented as a complete AgentExecutor subclass
that the user can instantiate with just a generator and tools.
"""

from typing import Any, Dict, List, Optional

from src.agents.core.agent_executor import AgentExecutor
from src.agents.core.base_agent import AgentConfig, AgentRun
from src.agents.tools.calculator_tool import CalculatorTool
from src.agents.tools.web_search_tool import WebSearchTool
from src.agents.tools.code_executor_tool import CodeExecutorTool
from src.agents.tools.retrieval_tool import RetrievalTool
from src.agents.tools.tool_registry import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReActAgent(AgentExecutor):
    """
    ReAct agent: the primary autonomous reasoning agent.

    Pre-configured with the standard tool suite and the ReAct prompt format.
    This is what you instantiate in production.

    Args:
        generator:         TextGenerator (our LLM).
        tools:             Optional dict of additional tools.
        memory:            Optional LongTermMemory.
        config:            AgentConfig.
        enable_reflection: Run self-critique after each answer.
        search_endpoint:   URL for web search service. If None, URL reading only.
    """

    def __init__(
        self,
        generator:         Any,
        tools:             Optional[Dict[str, Any]] = None,
        memory:            Optional[Any]            = None,
        config:            Optional[AgentConfig]    = None,
        enable_reflection: bool                     = False,
        search_endpoint:   Optional[str]            = None,
    ) -> None:
        # Build the standard tool registry
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(CodeExecutorTool())
        registry.register(WebSearchTool(search_endpoint=search_endpoint))
        registry.register(RetrievalTool(embed_fn=None))

        # Merge with any extra tools passed in
        all_tools = registry.as_dict()
        if tools:
            all_tools.update(tools)

        super().__init__(
            generator=generator,
            tools=all_tools,
            memory=memory,
            config=config or AgentConfig(max_steps=10, verbose=True),
            enable_reflection=enable_reflection,
        )

    @classmethod
    def create(
        cls,
        generator:    Any,
        search_url:   Optional[str] = None,
        persist_dir:  Optional[str] = None,
        embed_fn:     Optional[Any] = None,
        max_steps:    int           = 10,
        reflect:      bool          = False,
    ) -> "ReActAgent":
        """
        Factory method for creating a fully configured ReAct agent.

        Args:
            generator:    TextGenerator (our LLM).
            search_url:   Self-hosted search endpoint URL.
            persist_dir:  Directory for long-term memory persistence.
            embed_fn:     Embedding function for long-term memory retrieval.
            max_steps:    Maximum reasoning steps.
            reflect:      Enable self-reflection after answers.

        Returns:
            Configured ReActAgent.
        """
        memory = None
        if persist_dir or embed_fn:
            from src.agents.memory.long_term_memory import LongTermMemory
            memory = LongTermMemory(
                embed_fn=embed_fn,
                persist_dir=persist_dir,
            )

        return cls(
            generator=generator,
            memory=memory,
            config=AgentConfig(max_steps=max_steps, verbose=True),
            enable_reflection=reflect,
            search_endpoint=search_url,
        )
