"""
InsightSerenity AI Engine — Prompt Templates
=============================================
Standardised prompt templates for different use cases.
A prompt template wraps a user's input in the correct format for the model.

Why templates matter:
    A model fine-tuned on ChatML will only work well when the input follows
    the ChatML format. The template ensures the model sees the same format
    at inference time that it saw during fine-tuning.

Templates:
    SystemPromptTemplate    — single-turn with system prompt
    ChatTemplate            — multi-turn conversation
    FewShotTemplate         — few-shot examples prepended
    ChainOfThoughtTemplate  — encourages step-by-step reasoning
    CodeTemplate            — formatted for code generation tasks

Usage:
    from src.llm.prompting.templates import ChatTemplate

    template = ChatTemplate()
    prompt   = template.build(
        user_message="Summarise this article: ...",
        system="You are a concise summariser.",
    )
    response = generator.generate(prompt)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.tokenizer.special_tokens import SpecialTokens as ST


# ─────────────────────────────────────────────────────────────────────────────
# Base template
# ─────────────────────────────────────────────────────────────────────────────

class BaseTemplate:
    """Abstract base for all prompt templates."""

    DEFAULT_SYSTEM = (
        "You are InsightSerenity, a helpful, harmless, and honest AI assistant. "
        "You provide accurate, thoughtful responses."
    )

    def build(self, **kwargs) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Single-turn templates
# ─────────────────────────────────────────────────────────────────────────────

class SystemPromptTemplate(BaseTemplate):
    """
    Simple single-turn template with a configurable system prompt.

    Output format:
        <|system|>
        {system}
        <|user|>
        {user}
        <|assistant|>

    The model completes from the final <|assistant|> token.
    """

    def build(
        self,
        user:   str,
        system: Optional[str] = None,
    ) -> str:
        system_text = system or self.DEFAULT_SYSTEM
        return (
            f"{ST.SYSTEM}\n{system_text}\n"
            f"{ST.USER}\n{user}\n"
            f"{ST.ASSISTANT}\n"
        )


class ChatTemplate(BaseTemplate):
    """
    Multi-turn conversation template.

    Builds a prompt from a history of (role, content) pairs.
    Appends an empty assistant turn at the end to trigger completion.

    Args:
        system: System prompt. Uses default if None.
    """

    def __init__(self, system: Optional[str] = None) -> None:
        self.system = system or self.DEFAULT_SYSTEM

    def build(
        self,
        messages: List[Dict[str, str]],
        system:   Optional[str] = None,
    ) -> str:
        """
        Build a prompt from a message history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            system:   Optional system prompt override.

        Returns:
            Formatted prompt string ending with <|assistant|>.
        """
        sys_text = system or self.system
        parts    = [f"{ST.SYSTEM}\n{sys_text}\n"]

        for msg in messages:
            role    = msg["role"]
            content = msg["content"].strip()

            if role == "user":
                parts.append(f"{ST.USER}\n{content}\n")
            elif role == "assistant":
                parts.append(f"{ST.ASSISTANT}\n{content}\n{ST.END_TURN}\n")

        # Always end with the assistant delimiter to prompt completion
        if not messages or messages[-1]["role"] != "assistant":
            parts.append(f"{ST.ASSISTANT}\n")

        return "".join(parts)

    def add_turn(self, prompt: str, user_message: str) -> str:
        """
        Append a new user turn to an existing prompt string.
        Used for multi-turn interactive generation.
        """
        return prompt + f"{ST.USER}\n{user_message}\n{ST.ASSISTANT}\n"


class FewShotTemplate(BaseTemplate):
    """
    Few-shot prompting: prepend examples before the actual query.

    Example format:
        <|system|>
        You are a helpful assistant.
        <|user|>
        What is 2+2?
        <|assistant|>
        2+2 equals 4.
        <|end_turn|>
        ... (more examples) ...
        <|user|>
        {actual question}
        <|assistant|>

    Args:
        examples: List of {"user": "...", "assistant": "..."} dicts.
        system:   Optional system prompt.
    """

    def __init__(
        self,
        examples: List[Dict[str, str]],
        system:   Optional[str] = None,
    ) -> None:
        self.examples = examples
        self.system   = system or self.DEFAULT_SYSTEM

    def build(self, user: str, system: Optional[str] = None) -> str:
        """Build a few-shot prompt."""
        sys_text = system or self.system
        parts    = [f"{ST.SYSTEM}\n{sys_text}\n"]

        for ex in self.examples:
            parts.append(f"{ST.USER}\n{ex['user']}\n")
            parts.append(f"{ST.ASSISTANT}\n{ex['assistant']}\n{ST.END_TURN}\n")

        parts.append(f"{ST.USER}\n{user}\n{ST.ASSISTANT}\n")
        return "".join(parts)


class ChainOfThoughtTemplate(BaseTemplate):
    """
    Chain-of-Thought prompting: instructs the model to reason step-by-step
    before giving a final answer.

    Format:
        <|system|>
        {system} Think step by step before answering.
        <|user|>
        {user}
        <|assistant|>
        Let me think through this step by step.

    The "Let me think through this step by step." prefix is added to the
    assistant turn to encourage the model to reason before answering.
    """

    COT_TRIGGER = "Let me think through this step by step.\n\n"

    def build(
        self,
        user:   str,
        system: Optional[str] = None,
    ) -> str:
        sys_text = (system or self.DEFAULT_SYSTEM) + " Think step by step."
        return (
            f"{ST.SYSTEM}\n{sys_text}\n"
            f"{ST.USER}\n{user}\n"
            f"{ST.ASSISTANT}\n{self.COT_TRIGGER}"
        )


class CodeTemplate(BaseTemplate):
    """
    Template optimised for code generation tasks.

    Includes a system prompt that emphasises code quality and correctness.
    Wraps the user's request in a coding context.

    Args:
        language: Programming language hint (e.g. "Python", "TypeScript").
    """

    CODE_SYSTEM = (
        "You are an expert software engineer. Write clean, correct, and well-commented "
        "code. Follow best practices for the requested language. "
        "Only output code and brief explanations, no unnecessary prose."
    )

    def __init__(self, language: str = "Python") -> None:
        self.language = language

    def build(
        self,
        user:   str,
        system: Optional[str] = None,
    ) -> str:
        sys_text = system or f"{self.CODE_SYSTEM} Language: {self.language}."
        return (
            f"{ST.SYSTEM}\n{sys_text}\n"
            f"{ST.USER}\n{user}\n"
            f"{ST.ASSISTANT}\n```{self.language.lower()}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Template registry — lookup by name
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_REGISTRY = {
    "system":    SystemPromptTemplate,
    "chat":      ChatTemplate,
    "few_shot":  FewShotTemplate,
    "cot":       ChainOfThoughtTemplate,
    "code":      CodeTemplate,
}


def get_template(name: str, **kwargs) -> BaseTemplate:
    """
    Retrieve a prompt template by name.

    Args:
        name:    Template name. One of: system, chat, few_shot, cot, code.
        **kwargs: Forwarded to template constructor.

    Returns:
        An instantiated template object.
    """
    cls = _TEMPLATE_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown template '{name}'. "
            f"Available: {sorted(_TEMPLATE_REGISTRY.keys())}"
        )
    return cls(**kwargs)
