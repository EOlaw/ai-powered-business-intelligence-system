"""
InsightSerenity AI Engine — Short-Term (Context Window) Memory
===============================================================
Short-term memory manages the agent's immediate conversation context —
everything the agent has seen and done in the current session that fits
within the LLM's context window.

The context window is finite (e.g. 2048 tokens for our small LLM).
Short-term memory decides WHAT to include in each prompt:
    - Always include: the original task
    - Include if space: recent conversation turns (newest first)
    - Summarise if too long: compress old turns to free up token budget

Three truncation strategies:
    TAIL:    Keep only the most recent N turns (sliding window)
    HEAD:    Keep only the first N turns (preserve initial context)
    SUMMARY: Use the LLM to compress older turns into a summary

For most agent use cases, TAIL is correct: the most recent
observations are most relevant to deciding the next action.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TruncationStrategy(Enum):
    TAIL    = "tail"     # Keep most recent turns
    HEAD    = "head"     # Keep oldest turns
    SUMMARY = "summary"  # Summarise old turns with LLM


@dataclass
class ContextMessage:
    """A single message in the short-term context."""
    role:    str   # "system" | "user" | "assistant" | "tool"
    content: str
    tokens:  int  = 0   # Estimated token count


class ShortTermMemory:
    """
    Manages the agent's context window for a single conversation session.

    Stores messages and handles truncation when the total estimated
    token count approaches the context window limit.

    Token estimation: We use a simple heuristic of 4 characters per token
    (reasonable for English text). This avoids the overhead of running a
    real tokenizer on every memory operation.

    Args:
        max_tokens:   Maximum context window size in tokens.
        strategy:     How to truncate when over the limit.
        reserve_for_output: Tokens reserved for the LLM's response.
        generator:    Optional LLM for summarisation strategy.
    """

    CHARS_PER_TOKEN = 4    # Heuristic: 4 chars ≈ 1 token

    def __init__(
        self,
        max_tokens:          int                = 2048,
        strategy:            TruncationStrategy = TruncationStrategy.TAIL,
        reserve_for_output:  int                = 512,
        generator:           Optional[Any]      = None,
    ) -> None:
        self.max_tokens         = max_tokens
        self.strategy           = strategy
        self.reserve_for_output = reserve_for_output
        self.generator          = generator

        self._messages: List[ContextMessage] = []
        self._system_message: Optional[ContextMessage] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_system(self, content: str) -> None:
        """Set (or replace) the system message. Always kept, never truncated."""
        self._system_message = ContextMessage(
            role="system",
            content=content,
            tokens=self._estimate_tokens(content),
        )

    def add(self, role: str, content: str) -> None:
        """
        Add a message to the context.

        Automatically truncates older messages if the total token count
        would exceed the budget.
        """
        msg = ContextMessage(
            role=role,
            content=content,
            tokens=self._estimate_tokens(content),
        )
        self._messages.append(msg)

        # Truncate if over budget
        if self._total_tokens() > self._token_budget():
            self._truncate()

    def get_context(self) -> List[ContextMessage]:
        """
        Return all messages to include in the next prompt.

        The system message (if set) is always prepended.
        """
        messages = []
        if self._system_message:
            messages.append(self._system_message)
        messages.extend(self._messages)
        return messages

    def get_context_string(self) -> str:
        """
        Format the context as a single string for concatenation with the prompt.
        Role labels are included for clarity.
        """
        parts = []
        for msg in self.get_context():
            parts.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all messages (but keep the system message)."""
        self._messages.clear()

    def token_usage(self) -> dict:
        """Return current token usage statistics."""
        total = self._total_tokens()
        return {
            "total":   total,
            "budget":  self._token_budget(),
            "used_pct": round(total / max(self._token_budget(), 1) * 100, 1),
            "messages": len(self._messages),
        }

    # ── Truncation ─────────────────────────────────────────────────────────────

    def _truncate(self) -> None:
        """Remove messages until total token count is under the budget."""
        budget = self._token_budget()

        if self.strategy == TruncationStrategy.TAIL:
            # Remove from the front (oldest messages first)
            while self._messages and self._total_tokens() > budget:
                removed = self._messages.pop(0)
                logger.debug("Short-term memory: removed oldest message",
                             role=removed.role, tokens=removed.tokens)

        elif self.strategy == TruncationStrategy.HEAD:
            # Remove from the back (newest messages first)
            while self._messages and self._total_tokens() > budget:
                self._messages.pop()

        elif self.strategy == TruncationStrategy.SUMMARY and self.generator:
            self._summarise_and_truncate()

    def _summarise_and_truncate(self) -> None:
        """
        Summarise the oldest half of messages and replace them with the summary.
        This preserves the gist of earlier context while freeing up token budget.
        """
        if len(self._messages) < 4:
            return

        # Take the oldest half to summarise
        half      = len(self._messages) // 2
        old_msgs  = self._messages[:half]
        keep_msgs = self._messages[half:]

        # Build a summary prompt
        conversation = "\n".join(
            f"{m.role}: {m.content}" for m in old_msgs
        )
        summary_prompt = (
            f"Summarise this conversation in 2-3 sentences:\n{conversation}\n\nSummary:"
        )

        try:
            summary = self.generator.generate(summary_prompt, max_new_tokens=200, strategy="greedy")
            summary_msg = ContextMessage(
                role="system",
                content=f"[Previous context summary]: {summary}",
                tokens=self._estimate_tokens(summary),
            )
            self._messages = [summary_msg] + keep_msgs
        except Exception:
            # If summarisation fails, fall back to TAIL truncation
            self._messages = keep_msgs

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _total_tokens(self) -> int:
        """Estimated total tokens in current context."""
        system_tokens = self._system_message.tokens if self._system_message else 0
        return system_tokens + sum(m.tokens for m in self._messages)

    def _token_budget(self) -> int:
        """Available tokens for context (excluding output reserve)."""
        if self.reserve_for_output >= self.max_tokens:
            return self.max_tokens
        return self.max_tokens - self.reserve_for_output

    def _estimate_tokens(self, text: str) -> int:
        """Quick token count estimate: number of characters / 4."""
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ShortTermMemory(messages={len(self)}, tokens={self._total_tokens()}/{self._token_budget()})"
