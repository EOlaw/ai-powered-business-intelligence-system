"""
InsightSerenity AI Engine — API Response Schemas
================================================
Pydantic models for all API responses.
Compatible with the OpenAI API response format so existing clients
that parse OpenAI responses work without modification.
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


# ─────────────────────────────────────────────────────────────────────────────
# Usage tracking
# ─────────────────────────────────────────────────────────────────────────────

class UsageInfo(BaseModel):
    """Token usage counters for a completion."""
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Text completion response
# ─────────────────────────────────────────────────────────────────────────────

class CompletionChoice(BaseModel):
    text:          str
    index:         int
    finish_reason: Optional[str] = None   # "stop" | "length"
    logprobs:      Optional[Any] = None


class CompletionResponse(BaseModel):
    """Response from POST /v1/completions."""
    id:      str            = Field(default_factory=lambda: _new_id("cmpl"))
    object:  str            = "text_completion"
    created: int            = Field(default_factory=lambda: int(time.time()))
    model:   str            = "insightserenity-1"
    choices: List[CompletionChoice]
    usage:   UsageInfo      = Field(default_factory=UsageInfo)


# ─────────────────────────────────────────────────────────────────────────────
# Chat completion response
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatChoice(BaseModel):
    index:         int
    message:       ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response from POST /v1/chat/completions (non-streaming)."""
    id:      str            = Field(default_factory=lambda: _new_id("chatcmpl"))
    object:  str            = "chat.completion"
    created: int            = Field(default_factory=lambda: int(time.time()))
    model:   str            = "insightserenity-1"
    choices: List[ChatChoice]
    usage:   UsageInfo      = Field(default_factory=UsageInfo)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming delta (SSE chunk)
# ─────────────────────────────────────────────────────────────────────────────

class DeltaMessage(BaseModel):
    """Delta content in a streaming chunk."""
    role:    Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index:         int
    delta:         DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """One SSE chunk in a streaming chat completion."""
    id:      str = Field(default_factory=lambda: _new_id("chatcmpl"))
    object:  str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model:   str = "insightserenity-1"
    choices: List[StreamChoice]


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings response
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingObject(BaseModel):
    object:    str         = "embedding"
    embedding: List[float]
    index:     int


class EmbeddingResponse(BaseModel):
    """Response from POST /v1/embeddings."""
    object: str  = "list"
    data:   List[EmbeddingObject]
    model:  str  = "insightserenity-1"
    usage:  UsageInfo = Field(default_factory=UsageInfo)


# ─────────────────────────────────────────────────────────────────────────────
# Agent run response
# ─────────────────────────────────────────────────────────────────────────────

class AgentStepResponse(BaseModel):
    """One reasoning step in an agent run."""
    step_num:     int
    thought:      str
    action:       Optional[str] = None
    action_input: Optional[str] = None
    observation:  Optional[str] = None
    is_final:     bool          = False


class AgentRunResponse(BaseModel):
    """Response from POST /v1/agents/run."""
    id:           str  = Field(default_factory=lambda: _new_id("agentrun"))
    task:         str
    answer:       Optional[str] = None
    success:      bool          = False
    steps:        List[AgentStepResponse] = Field(default_factory=list)
    total_steps:  int           = 0
    elapsed_secs: float         = 0.0
    model:        str           = "insightserenity-1"


# ─────────────────────────────────────────────────────────────────────────────
# Error response
# ─────────────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """OpenAI-compatible error detail."""
    message: str
    type:    str  = "api_error"
    code:    Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard API error response."""
    error: ErrorDetail


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str  = "ok"
    model:   Optional[str] = None
    version: str  = "1.0.0"
    uptime:  float = 0.0


class ModelsListResponse(BaseModel):
    """Response from GET /v1/models (list available models)."""
    object: str = "list"
    data:   List[Dict[str, Any]]
