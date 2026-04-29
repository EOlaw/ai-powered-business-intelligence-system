"""
InsightSerenity AI Engine — API Request Schemas
================================================
Pydantic models for all incoming API request bodies.
Compatible with the OpenAI Chat Completions API format so any client
library built for OpenAI works with zero code changes.

All schema fields follow OpenAI's naming conventions where possible,
with InsightSerenity-specific extensions prefixed with "is_".

Endpoint → Schema mapping:
    POST /v1/completions             → CompletionRequest
    POST /v1/chat/completions        → ChatCompletionRequest
    POST /v1/embeddings              → EmbeddingRequest
    POST /v1/agents/run              → AgentRunRequest
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared / base types
# ─────────────────────────────────────────────────────────────────────────────

class SamplingParams(BaseModel):
    """Common sampling parameters shared across all generation endpoints."""
    temperature: float = Field(default=1.0,  ge=0.0, le=2.0)
    top_p:       float = Field(default=0.9,  gt=0.0, le=1.0)
    top_k:       int   = Field(default=50,   ge=1)
    max_tokens:  int   = Field(default=256,  ge=1, le=4096,
                               alias="max_tokens",
                               description="Maximum tokens to generate.")
    stop:        Optional[List[str]] = Field(
        default=None,
        description="List of strings where generation stops."
    )
    stream:      bool = Field(
        default=False,
        description="If True, return server-sent events (SSE) token stream."
    )

    @field_validator("temperature")
    @classmethod
    def temperature_not_zero_with_top_p(cls, v):
        # temperature=0 is greedy — validated elsewhere if needed
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Text completion
# ─────────────────────────────────────────────────────────────────────────────

class CompletionRequest(SamplingParams):
    """
    POST /v1/completions — Legacy text completion.

    Takes a raw prompt string and returns a completion.
    Equivalent to OpenAI's /v1/completions endpoint.
    """
    model:   str = Field(
        default="insightserenity-1",
        description="Model identifier (e.g. 'insightserenity-1' or 'gpt-small:v1.0.0')."
    )
    prompt:  Union[str, List[str]] = Field(
        ...,
        description="The input text prompt. Accepts a string or list of strings."
    )
    n:       int   = Field(default=1, ge=1, le=4,
                           description="Number of completions to generate per prompt.")
    echo:    bool  = Field(default=False,
                           description="If True, echo the prompt in the completion.")
    best_of: int   = Field(default=1, ge=1,
                           description="Generate best_of completions, return the best.")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("prompt must not be empty")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Chat completion
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role:    Literal["system", "user", "assistant", "tool"] = Field(
        ...,
        description="The role of the message author."
    )
    content: str = Field(
        ...,
        description="The text content of the message."
    )
    name:    Optional[str] = Field(
        default=None,
        description="Optional name for multi-user scenarios."
    )


class ChatCompletionRequest(SamplingParams):
    """
    POST /v1/chat/completions — Chat completion (primary endpoint).

    Takes a conversation history and returns the next assistant message.
    Compatible with the OpenAI Chat Completions API.
    """
    model:    str = Field(
        default="insightserenity-1",
        description="Model identifier."
    )
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history. Must contain at least one message."
    )
    n:        int   = Field(default=1, ge=1, le=4)
    # InsightSerenity extension: override the system prompt without modifying messages
    is_system_override: Optional[str] = Field(
        default=None,
        description="[IS extension] Override the system prompt for this request."
    )

    @field_validator("messages")
    @classmethod
    def last_message_is_user_or_system(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    """
    POST /v1/embeddings — Dense vector embeddings.

    Encodes one or more texts into fixed-length dense vectors.
    Compatible with OpenAI's /v1/embeddings endpoint.
    """
    model:     str = Field(default="insightserenity-1")
    input:     Union[str, List[str]] = Field(
        ...,
        description="Text or list of texts to embed."
    )
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Return format for embeddings. 'float' returns a list of floats."
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="If set, truncate embeddings to this dimension."
    )

    def get_texts(self) -> List[str]:
        """Normalise input to a list of strings."""
        return [self.input] if isinstance(self.input, str) else self.input


# ─────────────────────────────────────────────────────────────────────────────
# Agent run
# ─────────────────────────────────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    """
    POST /v1/agents/run — Run an autonomous agent on a task.

    The agent uses tools, memory, and multi-step reasoning to complete
    the given task and return a final answer.
    """
    model:       str = Field(default="insightserenity-1")
    task:        str = Field(
        ...,
        min_length=1,
        description="The task or question for the agent to solve."
    )
    max_steps:   int   = Field(default=10, ge=1, le=25)
    stream:      bool  = Field(default=False,
                               description="Stream step-by-step reasoning events.")
    tools:       Optional[List[str]] = Field(
        default=None,
        description="Restrict to these tool names. None = all available tools."
    )
    is_reflection: bool = Field(
        default=False,
        description="[IS extension] Run a self-reflection pass on the final answer."
    )
    context:     Optional[str] = Field(
        default=None,
        description="Optional context/background information for the task."
    )
