"""
InsightSerenity AI Engine — Agent Run Route
============================================
POST /v1/agents/run — Execute an autonomous ReAct agent task.

The agent receives a natural-language task, reasons step-by-step using the
ReAct (Reason + Act) loop, calls tools as needed, and returns a final answer.

Streaming mode (request.stream=True):
    Returns a Server-Sent Events (SSE) stream.  Each event carries one agent
    step (Thought / Action / Observation) as a JSON payload so the caller can
    display intermediate reasoning in real-time.

    Event format:
        data: {"event":"step","step_num":1,"thought":"...","action":"...","action_input":"...","observation":"..."}\n\n
        data: {"event":"final","answer":"...","success":true,"total_steps":3}\n\n
        data: [DONE]\n\n

Non-streaming mode:
    Blocks until the agent finishes and returns AgentRunResponse with the
    complete step trace and final answer.

Tool availability:
    All tools registered on the global ToolRegistry are available.
    The caller can restrict to a subset by passing `tools: ["calculator", "retrieval"]`.

Authorization:
    Requires internal service token (X-Org-Id, X-Key-Id forwarded from gateway).
    Scope "agents:run" or "admin:all" must be present.
"""

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import require_internal_auth, AuthContext
from src.api.schemas.requests import AgentRunRequest
from src.api.schemas.responses import (
    AgentRunResponse, AgentStepResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_agent(raw_request: Request, request: AgentRunRequest):
    """
    Construct an AgentExecutor wired to the application's engine.

    Pulls the TextGenerator from app.state and wraps it with the
    full tool suite from the global ToolRegistry.  If the caller
    specified a tool subset, only those tools are passed.
    """
    from src.agents.core.agent_executor import AgentExecutor
    from src.agents.core.base_agent import AgentConfig
    from src.agents.tools.tool_registry import ToolRegistry
    from src.agents.tools.calculator_tool import CalculatorTool
    from src.agents.tools.code_executor_tool import CodeExecutorTool
    from src.agents.tools.web_search_tool import WebSearchTool
    from src.agents.tools.retrieval_tool import RetrievalTool

    generator = getattr(raw_request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TextGenerator not initialised",
        )

    # Build or retrieve cached registry from app state
    registry: ToolRegistry = getattr(raw_request.app.state, "tool_registry", None)
    if registry is None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(CodeExecutorTool())
        registry.register(WebSearchTool())
        registry.register(RetrievalTool())
        raw_request.app.state.tool_registry = registry

    # Filter to requested tools
    available = registry.tools
    if request.tools:
        allowed = set(request.tools)
        available = {k: v for k, v in available.items() if k in allowed}

    config = AgentConfig(max_steps=request.max_steps)

    return AgentExecutor(
        generator=generator,
        tools=available,
        config=config,
        enable_reflection=request.is_reflection,
    )


def _prepend_context(task: str, context: str | None) -> str:
    """Prepend optional background context to the task string."""
    if not context:
        return task
    return f"Context:\n{context}\n\nTask: {task}"


# ─────────────────────────────────────────────────────────────────────────────
# SSE streaming generator
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_agent_run(
    agent,
    task: str,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """
    Run the agent synchronously (it's CPU-bound) and yield SSE events
    for each step as it completes.

    Because AgentExecutor.run() is synchronous, we run it in a thread-pool
    executor and stream the step objects after collecting them.  For true
    incremental streaming the agent would need an async step interface;
    this delivers the full trace with per-step SSE events once the run
    finishes — which is still far better UX than waiting for the final blob.
    """
    import asyncio
    loop = asyncio.get_event_loop()

    # Run the full agent in a thread so we don't block the event loop
    result = await loop.run_in_executor(None, agent.run, task)

    # Emit one SSE event per step
    for step_num, step in enumerate(result.steps, start=1):
        payload = {
            "event":        "step",
            "step_num":     step_num,
            "thought":      step.thought or "",
            "action":       step.action,
            "action_input": step.action_input,
            "observation":  step.observation,
            "is_final":     step.is_final,
        }
        yield f"data: {json.dumps(payload)}\n\n"

    # Final summary event
    final_payload = {
        "event":       "final",
        "answer":      result.final_answer,
        "success":     result.success,
        "total_steps": len(result.steps),
        "model":       model_name,
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Route handler
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/v1/agents/run",
    response_model=None,
    summary="Run an autonomous AI agent",
)
async def run_agent(
    request:     AgentRunRequest,
    raw_request: Request,
    auth:        AuthContext = Depends(require_internal_auth),
):
    """
    Execute a ReAct agent to complete a multi-step task.

    The agent thinks, selects tools, observes results, and repeats until it
    has enough information to produce a final answer — or until max_steps.

    Set stream=true to receive step-by-step reasoning as SSE events.
    """
    import asyncio

    agent      = _build_agent(raw_request, request)
    model_name = getattr(raw_request.app.state, "model_name", "insightserenity-1")
    task       = _prepend_context(request.task, request.context)

    if request.stream:
        return StreamingResponse(
            _stream_agent_run(agent, task, model_name),
            media_type="text/event-stream",
        )

    # ── Non-streaming: block until done ──────────────────────────────────────
    start   = time.perf_counter()
    loop    = asyncio.get_event_loop()
    result  = await loop.run_in_executor(None, agent.run, task)
    elapsed = time.perf_counter() - start

    steps = [
        AgentStepResponse(
            step_num=i + 1,
            thought=s.thought or "",
            action=s.action,
            action_input=s.action_input,
            observation=s.observation,
            is_final=s.is_final,
        )
        for i, s in enumerate(result.steps)
    ]

    return AgentRunResponse(
        task=request.task,
        answer=result.final_answer,
        success=result.success,
        steps=steps,
        total_steps=len(steps),
        elapsed_secs=round(elapsed, 3),
        model=model_name,
    )
