"""
InsightSerenity AI Engine — Self-Critique and Self-Correction
=============================================================
Self-reflection allows the agent to evaluate and improve its own outputs
without human feedback. The model critiques its initial answer and then
produces a revised, improved version.

This implements the "Constitutional AI" / "Self-Critique Chain" pattern:
    1. Generate initial answer
    2. Ask the same LLM: "What is wrong or could be improved about this answer?"
    3. Ask the same LLM: "Now give a better answer based on the critique."

Why this works:
    The LLM is better at evaluating text than generating it in one shot.
    A second pass with explicit critique context often produces significantly
    better answers. This is especially effective for:
        - Factual accuracy checks
        - Logical consistency
        - Completeness of the answer
        - Clarity and conciseness

CritiqueChain:
    Applies multiple critique dimensions in sequence:
    1. Factuality: Are the claims accurate and grounded?
    2. Completeness: Does the answer address all parts of the question?
    3. Clarity: Is the answer clear and well-organised?

Each critique generates a specific improvement, and the final answer
integrates all improvements.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CritiqueResult:
    """Result of one critique-and-improve cycle."""
    original_answer:  str
    critique:         str
    improved_answer:  str
    improved:         bool   # Whether the answer actually changed


class SelfCritic:
    """
    Self-critique and self-correction loop.

    Asks the LLM to critique its own answer and then improve it.
    Uses only our own LLM — no external oracle.

    Args:
        generator:    TextGenerator (our LLM).
        max_tokens:   Max tokens for critique and improvement generation.
        critique_prompt: Template for the critique prompt.
        improve_prompt:  Template for the improvement prompt.
    """

    DEFAULT_CRITIQUE_PROMPT = (
        "Question: {question}\n\n"
        "Proposed answer: {answer}\n\n"
        "Critique this answer. What is incorrect, incomplete, or unclear? "
        "Be specific and concise.\n\nCritique:"
    )

    DEFAULT_IMPROVE_PROMPT = (
        "Question: {question}\n\n"
        "Original answer: {answer}\n\n"
        "Critique: {critique}\n\n"
        "Based on this critique, write an improved answer that addresses "
        "the issues identified. If the original was already correct, say so.\n\n"
        "Improved answer:"
    )

    def __init__(
        self,
        generator:       Any,
        max_tokens:      int          = 300,
        critique_prompt: Optional[str] = None,
        improve_prompt:  Optional[str] = None,
    ) -> None:
        self.generator       = generator
        self.max_tokens      = max_tokens
        self.critique_prompt = critique_prompt or self.DEFAULT_CRITIQUE_PROMPT
        self.improve_prompt  = improve_prompt  or self.DEFAULT_IMPROVE_PROMPT

    def critique_and_improve(
        self,
        task:   str,
        answer: str,
    ) -> str:
        """
        Apply one critique-and-improve cycle to an answer.

        Args:
            task:   The original question or task.
            answer: The initial answer to critique.

        Returns:
            Improved answer string (may be the same if no issues found).
        """
        result = self.critique(task, answer)
        return result.improved_answer

    def critique(self, task: str, answer: str) -> CritiqueResult:
        """
        Full critique cycle: generate critique + improved answer.

        Args:
            task:   The original question.
            answer: The initial answer.

        Returns:
            CritiqueResult with critique text and improved answer.
        """
        # Step 1: Generate critique
        critique_prompt = self.critique_prompt.format(
            question=task,
            answer=answer,
        )
        critique = self.generator.generate(
            prompt=critique_prompt,
            max_new_tokens=self.max_tokens,
            strategy="greedy",
        ).strip()

        # Check if the critique indicates the answer was already good
        if self._is_positive_critique(critique):
            return CritiqueResult(
                original_answer=answer,
                critique=critique,
                improved_answer=answer,
                improved=False,
            )

        # Step 2: Generate improved answer
        improve_prompt = self.improve_prompt.format(
            question=task,
            answer=answer,
            critique=critique,
        )
        improved = self.generator.generate(
            prompt=improve_prompt,
            max_new_tokens=self.max_tokens * 2,
            strategy="greedy",
        ).strip()

        # Don't use the "improved" answer if it's significantly shorter
        # (often a sign the model regressed)
        if len(improved) < len(answer) * 0.3:
            improved = answer

        return CritiqueResult(
            original_answer=answer,
            critique=critique,
            improved_answer=improved,
            improved=(improved != answer),
        )

    def _is_positive_critique(self, critique: str) -> bool:
        """Return True if the critique says the answer was already good."""
        positive_phrases = [
            "the answer is correct", "no issues", "well written",
            "already accurate", "nothing to improve", "looks good",
            "the answer is complete", "satisfactory",
        ]
        lower = critique.lower()
        return any(phrase in lower for phrase in positive_phrases)


class CritiqueChain:
    """
    Multi-dimensional critique chain applying several critique passes.

    Each dimension targets a specific aspect of answer quality:
        factuality, completeness, clarity

    The output of each pass becomes the input for the next, progressively
    improving the answer across all dimensions.

    Args:
        generator:   TextGenerator.
        dimensions:  List of critique dimension names to apply.
        max_tokens:  Max tokens per critique pass.
    """

    DIMENSION_PROMPTS = {
        "factuality": (
            "Critique this answer focusing ONLY on factual accuracy. "
            "Identify any claims that might be incorrect or unverified."
        ),
        "completeness": (
            "Critique this answer focusing ONLY on completeness. "
            "What important aspects of the question are not addressed?"
        ),
        "clarity": (
            "Critique this answer focusing ONLY on clarity and organisation. "
            "Is the answer clearly structured? Is anything confusing?"
        ),
    }

    def __init__(
        self,
        generator:   Any,
        dimensions:  Optional[List[str]] = None,
        max_tokens:  int                 = 200,
    ) -> None:
        self.generator  = generator
        self.dimensions = dimensions or ["factuality", "completeness", "clarity"]
        self.max_tokens = max_tokens

    def improve(self, task: str, answer: str) -> Tuple[str, List[CritiqueResult]]:
        """
        Apply all critique dimensions sequentially.

        Args:
            task:   The original question.
            answer: Initial answer.

        Returns:
            Tuple (final_answer, list of CritiqueResult per dimension).
        """
        current_answer = answer
        results: List[CritiqueResult] = []

        for dimension in self.dimensions:
            dim_prompt = self.DIMENSION_PROMPTS.get(
                dimension,
                f"Critique this answer with respect to {dimension}."
            )

            critic = SelfCritic(
                generator=self.generator,
                max_tokens=self.max_tokens,
                critique_prompt=(
                    f"Question: {{question}}\n\nAnswer: {{answer}}\n\n"
                    f"{dim_prompt}\n\nCritique:"
                ),
            )

            result = critic.critique(task, current_answer)
            results.append(result)

            if result.improved:
                current_answer = result.improved_answer
                logger.debug("Answer improved", dimension=dimension)

        return current_answer, results
