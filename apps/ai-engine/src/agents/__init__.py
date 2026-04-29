"""
InsightSerenity AI Engine — Agents Package
==========================================
Complete agentic AI system: tools, memory, planning, reflection, multi-agent.

Core:
    from src.agents import AgentExecutor, ReActAgent, AgentConfig, AgentRun

Tools:
    from src.agents import ToolRegistry, BaseTool, FunctionTool
    from src.agents import CalculatorTool, CodeExecutorTool
    from src.agents import WebSearchTool, URLReaderTool, RetrievalTool

Memory:
    from src.agents import ShortTermMemory, LongTermMemory

Planning:
    from src.agents import ChainOfThoughtPlanner, zero_shot_cot
    from src.agents import TreeOfThoughtPlanner
    from src.agents import ReActAgent

Reflection:
    from src.agents import SelfCritic, CritiqueChain

Multi-agent:
    from src.agents import MultiAgentOrchestrator, AgentRegistry
    from src.agents import build_multi_agent_system
"""

from src.agents.core.base_agent import BaseAgent, AgentConfig, AgentRun, AgentStep
from src.agents.core.agent_executor import AgentExecutor
from src.agents.tools.tool_registry import ToolRegistry, BaseTool, FunctionTool
from src.agents.tools.calculator_tool import CalculatorTool
from src.agents.tools.code_executor_tool import CodeExecutorTool
from src.agents.tools.web_search_tool import WebSearchTool, URLReaderTool
from src.agents.tools.retrieval_tool import RetrievalTool
from src.agents.memory.short_term_memory import ShortTermMemory, TruncationStrategy
from src.agents.memory.long_term_memory import LongTermMemory
from src.agents.planning.chain_of_thought import (
    ChainOfThoughtPlanner, CoTResult, zero_shot_cot, few_shot_cot,
)
from src.agents.planning.react import ReActAgent
from src.agents.planning.tree_of_thought import TreeOfThoughtPlanner, ThoughtNode
from src.agents.reflection.self_critic import SelfCritic, CritiqueChain, CritiqueResult
from src.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator, AgentRegistry, AgentMessage,
    OrchestratorPlan, build_multi_agent_system,
)

__all__ = [
    # Core
    "BaseAgent", "AgentConfig", "AgentRun", "AgentStep",
    "AgentExecutor",
    # Tools
    "ToolRegistry", "BaseTool", "FunctionTool",
    "CalculatorTool", "CodeExecutorTool",
    "WebSearchTool", "URLReaderTool", "RetrievalTool",
    # Memory
    "ShortTermMemory", "TruncationStrategy",
    "LongTermMemory",
    # Planning
    "ChainOfThoughtPlanner", "CoTResult", "zero_shot_cot", "few_shot_cot",
    "ReActAgent",
    "TreeOfThoughtPlanner", "ThoughtNode",
    # Reflection
    "SelfCritic", "CritiqueChain", "CritiqueResult",
    # Multi-agent
    "MultiAgentOrchestrator", "AgentRegistry", "AgentMessage",
    "OrchestratorPlan", "build_multi_agent_system",
]
