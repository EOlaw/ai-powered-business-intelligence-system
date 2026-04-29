"""
InsightSerenity AI Engine — Instruction Formatter
==================================================
Converts raw instruction/response pairs into formatted text strings
that the model sees during supervised fine-tuning (SFT).

The template defines the structure the model learns to follow. During
inference, we feed only the instruction part and the model completes the
assistant response in the learned format.

Templates implemented:

1. ChatMLTemplate (our primary format)
   <|system|>
   You are a helpful AI assistant.
   <|user|>
   Write a haiku about autumn.
   <|assistant|>
   Leaves fall silently,
   ...
   <|end_turn|>

2. AlpacaTemplate
   ### Instruction:
   Summarise the following text.
   ### Input:
   (optional context)
   ### Response:
   ...

3. ShareGPTTemplate
   Multi-turn conversation in ShareGPT format (used by many open-source SFT datasets).

Usage:
    from src.llm.finetuning.instruction_formatter import ChatMLFormatter

    formatter = ChatMLFormatter()
    text = formatter.format_example({
        "system":    "You are a helpful assistant.",
        "user":      "What is the capital of France?",
        "assistant": "Paris is the capital of France.",
    })
    # → Full formatted conversation string
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.tokenizer.special_tokens import SpecialTokens as ST


# ─────────────────────────────────────────────────────────────────────────────
# Message types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single turn in a conversation."""
    role:    str   # "system", "user", or "assistant"
    content: str


@dataclass
class Conversation:
    """A multi-turn conversation."""
    messages: List[Message] = field(default_factory=list)
    system:   Optional[str] = None

    def add(self, role: str, content: str) -> "Conversation":
        self.messages.append(Message(role=role, content=content.strip()))
        return self


# ─────────────────────────────────────────────────────────────────────────────
# ChatML formatter (our primary format)
# ─────────────────────────────────────────────────────────────────────────────

class ChatMLFormatter:
    """
    Formats conversations using the ChatML template.

    ChatML was popularised by OpenAI's Chat Markup Language and is now
    widely used in open-source instruction-tuned models. We use our own
    special tokens (defined in SpecialTokens) which match the ChatML style.

    Format:
        <|system|>
        {system_message}
        <|user|>
        {user_message}
        <|assistant|>
        {assistant_message}
        <|end_turn|>

    For SFT training: only the assistant turns contribute to the loss.
    The instruction/system/user portions are masked out (label = -100).

    Args:
        default_system: Default system prompt inserted when none is provided.
        add_eos_at_end: Append <eos> token at the very end. Default True.
    """

    def __init__(
        self,
        default_system: str = "You are InsightSerenity, a helpful and harmless AI assistant.",
        add_eos_at_end: bool = True,
    ) -> None:
        self.default_system = default_system
        self.add_eos_at_end = add_eos_at_end

        # Special tokens used as role delimiters
        self.SYSTEM    = ST.SYSTEM      # "<|system|>"
        self.USER      = ST.USER        # "<|user|>"
        self.ASSISTANT = ST.ASSISTANT   # "<|assistant|>"
        self.END_TURN  = ST.END_TURN    # "<|end_turn|>"
        self.EOS       = ST.EOS         # "<eos>"

    def format_conversation(self, conversation: Conversation) -> str:
        """
        Format a full conversation into a training string.

        Args:
            conversation: Conversation object with messages.

        Returns:
            Formatted string with all turns concatenated.
        """
        parts: List[str] = []

        # System prompt
        system = conversation.system or self.default_system
        parts.append(f"{self.SYSTEM}\n{system}\n")

        for message in conversation.messages:
            if message.role == "user":
                parts.append(f"{self.USER}\n{message.content}\n")
            elif message.role == "assistant":
                parts.append(f"{self.ASSISTANT}\n{message.content}\n{self.END_TURN}\n")

        if self.add_eos_at_end:
            parts.append(self.EOS)

        return "".join(parts)

    def format_example(self, example: Dict[str, str]) -> str:
        """
        Convenience method for single-turn instruction/response pairs.

        Args:
            example: Dict with keys "system" (optional), "user", "assistant".

        Returns:
            Formatted string.
        """
        conv = Conversation(system=example.get("system"))
        conv.add("user",      example["user"])
        conv.add("assistant", example["assistant"])
        return self.format_conversation(conv)

    def format_prompt_only(self, user_message: str, system: Optional[str] = None) -> str:
        """
        Format only the prompt portion (no assistant response).
        Used during inference: feed this to the model to generate the response.

        Args:
            user_message: The user's input.
            system:       Optional system prompt override.

        Returns:
            Formatted prompt string ending just before the assistant's response.
        """
        system = system or self.default_system
        return (
            f"{self.SYSTEM}\n{system}\n"
            f"{self.USER}\n{user_message}\n"
            f"{self.ASSISTANT}\n"
        )

    def get_response_mask_positions(
        self,
        tokens: List[int],
        assistant_token_id: int,
        end_turn_token_id:  int,
    ) -> List[bool]:
        """
        Compute a boolean mask indicating which token positions belong to
        assistant responses (True = include in loss, False = mask out).

        This allows SFT training to compute loss only on the model's own
        responses, not on the system prompt or user turns.

        Args:
            tokens:              Tokenised full conversation.
            assistant_token_id:  Token ID for <|assistant|>.
            end_turn_token_id:   Token ID for <|end_turn|>.

        Returns:
            List of bool, same length as tokens.
            True at positions inside assistant turns.
        """
        mask = [False] * len(tokens)
        in_assistant = False

        for i, token_id in enumerate(tokens):
            if token_id == assistant_token_id:
                in_assistant = True
                continue   # Don't include the delimiter itself
            if token_id == end_turn_token_id:
                in_assistant = False
                continue
            if in_assistant:
                mask[i] = True

        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca formatter
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaFormatter:
    """
    Formats instruction data in the Alpaca template.

    Format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}   (optional context)

        ### Response:
        {output}

    Args:
        response_prefix: The text that appears before the model's response.
                         Used to identify where to start measuring loss.
    """

    INSTRUCTION_PREFIX = "### Instruction:\n"
    INPUT_PREFIX       = "\n### Input:\n"
    RESPONSE_PREFIX    = "\n### Response:\n"

    def __init__(self, add_eos: bool = True) -> None:
        self.add_eos = add_eos

    def format_example(self, example: Dict[str, str]) -> str:
        """
        Format a single instruction/response example.

        Args:
            example: Dict with keys "instruction", "input" (optional), "output".

        Returns:
            Formatted string.
        """
        instruction = example.get("instruction", "")
        context     = example.get("input", "").strip()
        response    = example.get("output", "")

        text = self.INSTRUCTION_PREFIX + instruction

        if context:
            text += self.INPUT_PREFIX + context

        text += self.RESPONSE_PREFIX + response

        if self.add_eos:
            text += ST.EOS

        return text

    def format_prompt_only(self, instruction: str, context: str = "") -> str:
        """Format only the instruction portion (no response). Used for inference."""
        text = self.INSTRUCTION_PREFIX + instruction
        if context:
            text += self.INPUT_PREFIX + context
        text += self.RESPONSE_PREFIX
        return text
