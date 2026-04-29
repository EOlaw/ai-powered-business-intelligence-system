"""
InsightSerenity AI Engine — Token Sampler
==========================================
Decoding strategies that convert the model's raw output logits into the
next token to generate. The choice of strategy critically affects the
quality and diversity of generated text.

Strategies implemented:

1. greedy_sample(logits)
   Always pick the highest-probability token.
   Deterministic — same input always gives same output.
   Pro: coherent, on-topic.  Con: repetitive, no creativity.

2. top_k_sample(logits, k)
   Sample from the top-K tokens (zero out the rest first).
   Controls vocabulary diversity. k=50 is a common default.
   Pro: diverse but constrained.  Con: fixed k may be too loose/tight.

3. top_p_sample(logits, p)
   Nucleus sampling: sample from the smallest set of tokens whose
   cumulative probability ≥ p. The set size adapts to the model's
   confidence — when the model is confident, fewer tokens are in the set.
   p=0.9 is the standard. This is the best general-purpose strategy.

4. temperature_scale(logits, T)
   Divide logits by temperature T before applying any sampling strategy.
   T=1.0: unmodified (default).
   T<1.0: sharper distribution, more deterministic (conservative).
   T>1.0: flatter distribution, more random (creative).

Combining strategies:
    temperature + top_p:  the most commonly used combination
    temperature + top_k:  good for structured output (code, JSON)
    greedy:               best for factual Q&A where accuracy matters most

Reference:
    "The Curious Case of Neural Text Degeneration" — Holtzman et al., 2020.
    This paper introduced nucleus (top-p) sampling.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def temperature_scale(logits: Tensor, temperature: float) -> Tensor:
    """
    Scale logits by temperature before sampling.

    Lower temperature → sharper (more deterministic) distribution.
    Higher temperature → flatter (more random) distribution.

    Args:
        logits:      (V,) or (B, V) — raw model output logits.
        temperature: Scaling factor. Must be > 0.

    Returns:
        Scaled logits of the same shape.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def greedy_sample(logits: Tensor) -> int:
    """
    Select the token with the highest probability.

    Args:
        logits: (V,) — logits for a single position.

    Returns:
        Integer token ID.
    """
    return logits.argmax(dim=-1).item()


def top_k_sample(logits: Tensor, k: int) -> int:
    """
    Sample from the top-K tokens by probability.

    Tokens outside the top-K are masked to -infinity before softmax,
    so they receive 0 probability. Sampling then picks from only the
    remaining K tokens.

    Args:
        logits: (V,) — logits for a single position.
        k:      Number of top tokens to keep. k=1 is equivalent to greedy.

    Returns:
        Integer token ID.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    k = min(k, logits.size(-1))

    # Find the k-th largest value and mask everything below it
    top_k_values, _ = torch.topk(logits, k)
    threshold        = top_k_values[..., -1, None]   # k-th largest value

    # Set logits below threshold to -inf
    filtered = logits.masked_fill(logits < threshold, float("-inf"))

    # Sample from the filtered distribution
    probs  = F.softmax(filtered, dim=-1)
    sample = torch.multinomial(probs, num_samples=1)
    return sample.item()


def top_p_sample(logits: Tensor, p: float) -> int:
    """
    Nucleus (top-p) sampling.

    Sort tokens by probability, then keep adding to the nucleus until
    cumulative probability ≥ p. Sample from the nucleus.

    Args:
        logits: (V,) — logits for a single position.
        p:      Cumulative probability threshold. 0.9 is a good default.

    Returns:
        Integer token ID.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs  = F.softmax(sorted_logits, dim=-1)
    cumulative    = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative probability exceeds p
    # Shift right by 1 to INCLUDE the token that pushes cumulative over p
    remove_mask = cumulative - sorted_probs > p
    sorted_logits[remove_mask] = float("-inf")

    # Sample from the filtered distribution (in sorted space)
    filtered_probs = F.softmax(sorted_logits, dim=-1)
    sample_idx     = torch.multinomial(filtered_probs, num_samples=1)

    # Map back to original vocabulary index
    return sorted_indices[sample_idx].item()


def repetition_penalty(
    logits:    Tensor,
    input_ids: Tensor,
    penalty:   float = 1.0,
) -> Tensor:
    """
    Apply repetition penalty to discourage generating already-seen tokens.

    Tokens that have already appeared in input_ids have their logits divided
    by `penalty` if they are positive, or multiplied if negative.
    penalty=1.0 has no effect; penalty>1.0 penalises repetition.

    Args:
        logits:    (V,) — logits for the next token.
        input_ids: (T,) — token IDs seen so far.
        penalty:   Repetition penalty factor. Default 1.0.

    Returns:
        Modified logits of the same shape.
    """
    if penalty == 1.0:
        return logits

    # Score already-seen tokens
    score = logits[input_ids]
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits[input_ids] = score
    return logits


class Sampler:
    """
    Unified sampling interface that combines all strategies.

    Supports:
        "greedy"  — deterministic, no randomness
        "top_k"   — sample from top-K tokens
        "top_p"   — nucleus sampling
        "beam"    — placeholder (beam search not implemented here; use HuggingFace for beam)

    Args:
        strategy:   Sampling strategy name.
        temperature: Temperature for logit scaling. Default 1.0.
        top_k:      K for top-k sampling. Default 50.
        top_p:      p for nucleus sampling. Default 0.9.
        rep_penalty: Repetition penalty factor. Default 1.0 (off).
    """

    def __init__(
        self,
        strategy:    str   = "top_p",
        temperature: float = 1.0,
        top_k:       int   = 50,
        top_p:       float = 0.9,
        rep_penalty: float = 1.0,
    ) -> None:
        self.strategy    = strategy
        self.temperature = temperature
        self.top_k       = top_k
        self.top_p       = top_p
        self.rep_penalty = rep_penalty

    def sample(
        self,
        logits:    Tensor,
        input_ids: Optional[Tensor] = None,
    ) -> int:
        """
        Sample the next token from logits using the configured strategy.

        Args:
            logits:    (V,) — raw logits for the next position.
            input_ids: Optional (T,) of previously generated IDs
                       (used for repetition penalty).

        Returns:
            Integer token ID.
        """
        # Apply repetition penalty
        if self.rep_penalty != 1.0 and input_ids is not None:
            logits = repetition_penalty(logits.clone(), input_ids, self.rep_penalty)

        # Apply temperature
        if self.temperature != 1.0:
            logits = temperature_scale(logits, self.temperature)

        # Sample
        if self.strategy == "greedy":
            return greedy_sample(logits)
        elif self.strategy == "top_k":
            return top_k_sample(logits, self.top_k)
        elif self.strategy == "top_p":
            return top_p_sample(logits, self.top_p)
        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Expected: greedy | top_k | top_p"
            )
