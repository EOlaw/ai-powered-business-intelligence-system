"""
InsightSerenity AI Engine — Mixture of Experts (MoE)
=====================================================
Sparse Mixture of Experts replaces the dense FFN in a transformer block with
a set of E expert FFNs, activating only the top-K experts for each token.

Key insight: each forward pass computes only K/E of the total expert capacity.
This allows the total parameter count to grow (more experts = more knowledge)
without a proportional increase in FLOPs per token.

Architecture:
    token x → TopKRouter(x) → {expert_1, ..., expert_K} → weighted sum

Router output: probabilities for each expert
Top-K selection: only K experts process each token
Load balancing: auxiliary loss prevents all tokens routing to the same expert

Used by: Mixtral 8×7B, Mixtral 8×22B, GPT-4 (suspected), Switch Transformer.

Implementation:
    Expert:    Standard FFN or SwiGLU FFN
    Router:    Linear(d_model, num_experts) → softmax → top-k
    MoELayer:  Dispatches tokens to selected experts, aggregates results
    SparseMoETransformerLayer: Drop-in replacement for GPTDecoderLayer's FFN

Load balancing auxiliary loss (Switch Transformer):
    L_aux = num_experts × sum_i(f_i × P_i)
    f_i = fraction of tokens dispatched to expert i
    P_i = average router probability for expert i
    This penalises routing collapse (all tokens going to one expert).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.nn.layers.feedforward import FeedForward, GatedFeedForward
from src.nn.normalization.normalization import RMSNorm


class Expert(nn.Module):
    """
    A single expert — just a feedforward network.

    Each expert specialises in a different aspect of the input distribution.
    In practice, experts tend to specialise by topic, syntax, or language
    patterns after training.

    Args:
        d_model:  Input and output dimension.
        d_ff:     Hidden dimension. Default None (auto-computed by FFN).
        ffn_type: "standard" (GELU FFN) or "swiglu" (SwiGLU FFN).
        dropout:  Dropout inside the FFN.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        ffn_type: str = "swiglu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if ffn_type == "swiglu":
            self.ffn: nn.Module = GatedFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        else:
            self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (tokens, d_model) — flattened token representations.

        Returns:
            (tokens, d_model)
        """
        return self.ffn(x)


class TopKRouter(nn.Module):
    """
    Gating network that routes each token to the top-K experts.

    Computes a probability distribution over experts for each token,
    selects the K highest-probability experts, and returns:
        - The indices of selected experts
        - The normalised routing weights (sum to 1 across K)
        - An auxiliary load-balancing loss (to prevent routing collapse)

    Args:
        d_model:     Token representation dimension.
        num_experts: Total number of experts.
        top_k:       Number of experts to activate per token. Default 2.
        noise_std:   Standard deviation of Gaussian noise added to logits
                     during training to encourage exploration. Default 0.01.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k       = top_k
        self.noise_std   = noise_std

        # Linear gate: maps each token to a score for each expert
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute expert routing for all tokens in the input.

        Args:
            x: (B×T, d_model) — flattened token representations.

        Returns:
            Tuple of:
                expert_indices: (B×T, top_k) LongTensor — selected expert indices.
                routing_weights: (B×T, top_k) FloatTensor — normalised weights.
                aux_loss: scalar — load balancing auxiliary loss.
        """
        # Compute routing logits
        logits = self.gate(x)   # (B×T, num_experts)

        # Add noise during training (encourages exploration and load balancing)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Select top-K experts per token
        # topk_weights: (B×T, top_k), topk_indices: (B×T, top_k)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Normalise selected weights via softmax over the K chosen logits
        # This ensures routing_weights.sum(dim=-1) == 1 for each token
        routing_weights = F.softmax(top_k_logits, dim=-1)

        # ── Load balancing auxiliary loss (Switch Transformer) ──────────────
        # Penalises routing collapse: encourages even distribution across experts.
        # L_aux = num_experts × sum_i(f_i × P_i)
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability for expert i
        all_probs = F.softmax(logits, dim=-1)   # (B×T, num_experts)

        # f_i: expert load fractions (how many tokens go to each expert)
        # Use a one-hot / binary selection for f_i
        # We use straight-through: binary selection but gradient through probs
        one_hot_mask = torch.zeros_like(all_probs)
        one_hot_mask.scatter_(-1, top_k_indices, 1.0)
        f = one_hot_mask.mean(dim=0)         # (num_experts,) — load fractions

        # P_i: average router probability per expert
        P = all_probs.mean(dim=0)            # (num_experts,)

        # Aux loss: encourages f and P to be uniform (equal load per expert)
        aux_loss = self.num_experts * (f * P).sum()

        return top_k_indices, routing_weights, aux_loss


class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer.

    Replaces the FFN in a transformer block. Each token is routed to K
    experts based on the router's scores, and the outputs are linearly
    combined using the routing weights.

    Dispatch strategy: we use token-choice routing (each token selects its
    preferred experts) rather than expert-choice routing (each expert selects
    preferred tokens). Token-choice is the standard in practice.

    Args:
        d_model:     Model dimension.
        num_experts: Total number of expert FFNs.
        top_k:       Experts activated per token. Default 2.
        d_ff:        Expert FFN hidden dimension. Default None (auto).
        ffn_type:    "standard" or "swiglu".
        dropout:     Expert FFN dropout.
        noise_std:   Router noise during training.
        aux_loss_coeff: Multiplier for load-balancing loss. Default 0.01.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        ffn_type: str = "swiglu",
        dropout: float = 0.0,
        noise_std: float = 0.01,
        aux_loss_coeff: float = 0.01,
    ) -> None:
        super().__init__()

        self.d_model        = d_model
        self.num_experts    = num_experts
        self.top_k          = top_k
        self.aux_loss_coeff = aux_loss_coeff

        # Create all experts
        self.experts = nn.ModuleList([
            Expert(d_model=d_model, d_ff=d_ff, ffn_type=ffn_type, dropout=dropout)
            for _ in range(num_experts)
        ])

        self.router = TopKRouter(d_model, num_experts, top_k, noise_std)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Route tokens to experts and combine results.

        Args:
            x: (B, T, D) — input token representations.

        Returns:
            Tuple of:
                output:   (B, T, D) — expert-processed representations.
                aux_loss: scalar — load balancing auxiliary loss to add to training loss.
        """
        B, T, D = x.shape

        # Flatten tokens for routing: (B×T, D)
        x_flat = x.reshape(B * T, D)

        # Get routing decisions
        expert_indices, routing_weights, aux_loss = self.router(x_flat)
        # expert_indices: (B×T, top_k), routing_weights: (B×T, top_k)

        # Collect expert outputs
        # We process each expert's assigned tokens in one batch
        output = torch.zeros_like(x_flat)   # (B×T, D)

        for k in range(self.top_k):
            # expert_indices[:, k]: which expert each token chose at position k
            # routing_weights[:, k]: weight to apply to that expert's output

            for e, expert in enumerate(self.experts):
                # Find tokens that chose expert e at selection position k
                mask = (expert_indices[:, k] == e)  # (B×T,) bool
                if not mask.any():
                    continue

                # Process this expert's tokens
                expert_input  = x_flat[mask]             # (n_tokens, D)
                expert_output = expert(expert_input)      # (n_tokens, D)

                # Weight and accumulate
                weight = routing_weights[mask, k].unsqueeze(-1)  # (n_tokens, 1)
                output[mask] += expert_output * weight

        # Reshape back to (B, T, D)
        output = output.reshape(B, T, D)

        return output, aux_loss * self.aux_loss_coeff

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}"
        )


class SparseMoETransformerLayer(nn.Module):
    """
    Transformer decoder layer with MoE replacing the dense FFN.

    Drop-in replacement for GPTDecoderLayer when you want to add MoE
    to a transformer. The attention sublayer is identical; only the FFN
    is replaced by a MoELayer.

    Args:
        d_model:           Model dimension.
        num_heads:         Attention heads.
        num_experts:       Number of MoE experts.
        top_k:             Active experts per token.
        max_seq_len:       For causal mask.
        dropout:           Residual dropout.
        attention_dropout: Attention weight dropout.
        norm_type:         "rms" or "layer".
        use_rope:          Apply RoPE to Q and K.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        norm_type: str = "rms",
        use_rope: bool = True,
    ) -> None:
        super().__init__()

        from src.architectures.transformer.attention.multi_head_attention import CausalSelfAttention
        NormClass = RMSNorm if norm_type == "rms" else RMSNorm   # Always RMSNorm for MoE

        self.norm_1 = NormClass(d_model)
        self.norm_2 = NormClass(d_model)

        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=attention_dropout,
            use_rope=use_rope,
        )

        self.moe  = MoELayer(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.drop = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x:                (B, T, D)
            key_padding_mask: (B, T)

        Returns:
            Tuple (output (B, T, D), attn_weights (B, H, T, T), aux_loss scalar).
        """
        # Pre-norm causal self-attention
        residual = x
        x, attn_weights = self.attn(self.norm_1(x), key_padding_mask=key_padding_mask)
        x = self.drop(x) + residual

        # Pre-norm MoE FFN
        residual = x
        moe_out, aux_loss = self.moe(self.norm_2(x))
        x = self.drop(moe_out) + residual

        return x, attn_weights, aux_loss
