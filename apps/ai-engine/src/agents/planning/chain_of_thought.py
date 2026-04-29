"""
InsightSerenity AI Engine — Chain-of-Thought Planning
======================================================
Chain-of-Thought (CoT) prompting (Wei et al., 2022) elicits step-by-step
reasoning from the LLM before it produces its final answer. Instead of:
    Q: "What is 17 × 24?" → A: "408"

CoT produces:
    Q: "What is 17 × 24? Let's think step by step."
    A: "17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408"

This dramatically improves performance on multi-step reasoning tasks,
especially arithmetic, commonsense reasoning, and complex Q&A.

Why it works:
    The LLM is an autoregressive model — each token is conditioned on all
    previous tokens. By forcing the model to write out its reasoning, it
    creates intermediate context that guides the rest of the generation.

Zero-shot CoT: Just add "Let's think step by step." — no examples needed.
Few-shot CoT: Provide worked examples with step-by-step solutions.

This module provides:
    ChainOfThoughtPlanner — wraps any generator with CoT prompting
    zero_shot_cot(question) → prompted string
    few_shot_cot(question, examples) → prompted string
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def zero_shot_cot(question: str, trigger: str = "Let's think step by step.") -> str:
    """
    Apply zero-shot CoT trigger to a question.

    The "Let's think step by step." trigger was found by Kojima et al. (2022)
    to significantly improve zero-shot reasoning across diverse tasks.

    Args:
        question: The question or task to reason about.
        trigger:  The CoT trigger phrase appended to the question.

    Returns:
        Formatted prompt string.
    """
    return f"{question}\n\n{trigger}"


def few_shot_cot(
    question:  str,
    examples:  List[Dict[str, str]],
    system:    str = "Solve problems step by step.",
) -> str:
    """
    Apply few-shot CoT with worked examples.

    Args:
        question:  The question to answer.
        examples:  List of {"question": ..., "reasoning": ..., "answer": ...} dicts.
        system:    System/instruction prefix.

    Returns:
        Formatted few-shot CoT prompt.
    """
    parts = [system, ""]

    for ex in examples:
        parts.append(f"Q: {ex['question']}")
        parts.append(f"A: {ex['reasoning']}")
        parts.append(f"Therefore, the answer is: {ex['answer']}")
        parts.append("")

    parts.append(f"Q: {question}")
    parts.append("A: Let's think step by step.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# ChainOfThoughtPlanner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoTResult:
    """Result of a CoT reasoning pass."""
    question:   str
    reasoning:  str     # The full step-by-step reasoning
    answer:     str     # The extracted final answer
    prompt:     str     # The prompt that was sent to the LLM


class ChainOfThoughtPlanner:
    """
    Wraps a TextGenerator with Chain-of-Thought prompting.

    The planner handles:
        1. Constructing the CoT prompt (zero-shot or few-shot)
        2. Generating the reasoning
        3. Extracting the final answer from the reasoning

    Args:
        generator:   TextGenerator (our LLM).
        examples:    Optional few-shot examples.
        max_tokens:  Max tokens for the reasoning generation.
    """

    ANSWER_TRIGGERS = [
        "therefore, the answer is",
        "the answer is",
        "final answer:",
        "so the answer is",
        "in conclusion,",
    ]

    def __init__(
        self,
        generator:  Any,
        examples:   Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
    ) -> None:
        self.generator  = generator
        self.examples   = examples or []
        self.max_tokens = max_tokens

    def reason(self, question: str) -> CoTResult:
        """
        Apply CoT reasoning to a question.

        Args:
            question: The question or problem to solve.

        Returns:
            CoTResult with the full reasoning chain and extracted answer.
        """
        # Choose zero-shot or few-shot based on whether examples are configured
        if self.examples:
            prompt = few_shot_cot(question, self.examples)
        else:
            prompt = zero_shot_cot(question)

        logger.debug("CoT reasoning", question=question[:80])

        # Generate the step-by-step reasoning
        reasoning = self.generator.generate(
            prompt=prompt,
            max_new_tokens=self.max_tokens,
            strategy="greedy",
            temperature=0.0,
        )

        answer = self._extract_answer(reasoning)

        return CoTResult(
            question=question,
            reasoning=reasoning,
            answer=answer,
            prompt=prompt,
        )

    def _extract_answer(self, reasoning: str) -> str:
        """
        Extract the final answer from the generated reasoning chain.

        Looks for common answer indicator phrases. Falls back to
        the last non-empty sentence if no indicator is found.
        """
        reasoning_lower = reasoning.lower()

        for trigger in self.ANSWER_TRIGGERS:
            idx = reasoning_lower.rfind(trigger)
            if idx >= 0:
                answer_part = reasoning[idx + len(trigger):].strip()
                # Take up to the first sentence end
                for end in [".", "\n"]:
                    end_idx = answer_part.find(end)
                    if end_idx > 0:
                        return answer_part[:end_idx].strip()
                return answer_part[:200].strip()

        # Fallback: last non-empty sentence
        sentences = [s.strip() for s in reasoning.split(".") if s.strip()]
        return sentences[-1] if sentences else reasoning[:200]
