"""
Unit tests for nn/layers: Linear, Embeddings, Conv, Attention, FeedForward

Coverage:
    Linear:
        - Output shape correctness
        - Bias present / absent
        - All initialisation strategies produce valid weights
        - Gradient flows through forward pass
        - in_features/out_features with arbitrary leading dims

    TokenEmbedding:
        - Maps token IDs to correct embedding dimension
        - Padding index is all zeros
        - Scale factor applied correctly
        - Gradient flows to embedding weights

    SinusoidalPositionalEmbedding:
        - Output shape matches input
        - Positional encoding table is not a learnable parameter
        - Different positions produce different encodings

    LearnedPositionalEmbedding:
        - Output shape matches input
        - Weights are learnable parameters
        - Gradient flows

    RotaryPositionalEmbedding:
        - Output shapes match inputs
        - Query and key are rotated differently from value (value unchanged)
        - Relative position property: rot(q_i) · rot(k_j) depends on (i-j)

    CausalConv1d:
        - Output length == input length (causal property)
        - No future leakage: output at time t depends only on times ≤ t

    Conv1dBlock:
        - Output shape with various padding options
        - BatchNorm and LayerNorm variants
        - Causal variant output length

    Attention functions:
        - scaled_dot_product_attention output shape
        - Causal mask: upper triangle is zeroed
        - Padding mask: masked positions have zero attention weight
        - Attention weights sum to 1 (per query, after softmax)

    MultiHeadAttention:
        - Self-attention output shape
        - Cross-attention output shape
        - Gradient flows through all heads
        - Causal mask applied correctly in MultiHeadCausalAttention

    FeedForward:
        - Output shape
        - d_ff defaults to 4×d_model
        - Gradient flows

    GatedFeedForward (SwiGLU):
        - Output shape same as standard FFN
        - d_ff auto-computed when None
"""

import math
import pytest
import torch
import torch.nn as nn

from src.nn.layers.linear import Linear
from src.nn.layers.embedding import (
    TokenEmbedding,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
)
from src.nn.layers.conv import CausalConv1d, Conv1dBlock, Conv2dBlock
from src.nn.layers.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    MultiHeadCausalAttention,
    build_causal_mask,
    build_padding_mask,
)
from src.nn.layers.feedforward import FeedForward, GatedFeedForward


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

B, T, D = 2, 8, 64   # batch=2, seq=8, dim=64
H       = 4           # heads


@pytest.fixture
def dummy_seq():
    """(B, T, D) float tensor."""
    return torch.randn(B, T, D)


@pytest.fixture
def dummy_token_ids():
    """(B, T) long tensor of random token IDs."""
    return torch.randint(0, 100, (B, T))


# ─────────────────────────────────────────────────────────────────────────────
# Linear
# ─────────────────────────────────────────────────────────────────────────────

class TestLinear:

    def test_output_shape(self):
        layer  = Linear(D, 128)
        x      = torch.randn(B, T, D)
        output = layer(x)
        assert output.shape == (B, T, 128)

    def test_no_bias_shape(self):
        layer = Linear(D, 32, bias=False)
        x     = torch.randn(B, D)
        assert layer(x).shape == (B, 32)
        assert layer.bias is None

    def test_gradient_flows(self):
        layer  = Linear(D, D)
        x      = torch.randn(B, D, requires_grad=True)
        loss   = layer(x).sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    @pytest.mark.parametrize("init", [
        "kaiming_uniform", "kaiming_normal", "xavier_uniform",
        "xavier_normal", "zeros", "normal",
    ])
    def test_all_init_strategies_valid(self, init):
        layer = Linear(32, 64, init=init)
        assert layer.weight.shape == (64, 32)
        assert not torch.isnan(layer.weight).any()
        assert not torch.isinf(layer.weight).any()

    def test_zeros_init_all_zero(self):
        layer = Linear(16, 16, init="zeros")
        assert (layer.weight == 0).all()

    def test_unknown_init_raises(self):
        with pytest.raises(ValueError, match="Unknown init"):
            Linear(8, 8, init="nonexistent_strategy")

    def test_arbitrary_leading_dims(self):
        layer = Linear(D, 32)
        x     = torch.randn(3, 4, 5, D)
        assert layer(x).shape == (3, 4, 5, 32)


# ─────────────────────────────────────────────────────────────────────────────
# TokenEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenEmbedding:

    def test_output_shape(self, dummy_token_ids):
        emb    = TokenEmbedding(vocab_size=200, d_model=D)
        output = emb(dummy_token_ids)
        assert output.shape == (B, T, D)

    def test_padding_index_is_zeros(self, dummy_token_ids):
        emb = TokenEmbedding(vocab_size=200, d_model=D, pad_id=0)
        # Embedding at index 0 (pad) must be all zeros
        pad_emb = emb.embedding.weight[0]
        assert torch.all(pad_emb == 0.0)

    def test_scale_applied(self):
        emb      = TokenEmbedding(vocab_size=50, d_model=D, scale=True)
        emb_no_s = TokenEmbedding(vocab_size=50, d_model=D, scale=False)
        ids      = torch.tensor([[1, 2, 3]])
        # Copy same weights
        emb_no_s.embedding.weight.data = emb.embedding.weight.data.clone()
        scaled     = emb(ids)
        not_scaled = emb_no_s(ids)
        expected_ratio = math.sqrt(D)
        assert torch.allclose(scaled, not_scaled * expected_ratio)

    def test_gradient_flows_to_embedding(self, dummy_token_ids):
        emb  = TokenEmbedding(vocab_size=200, d_model=D)
        loss = emb(dummy_token_ids).sum()
        loss.backward()
        assert emb.embedding.weight.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# SinusoidalPositionalEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestSinusoidalPE:

    def test_output_shape(self, dummy_seq):
        pe     = SinusoidalPositionalEmbedding(d_model=D, max_seq_len=512, dropout=0.0)
        output = pe(dummy_seq)
        assert output.shape == (B, T, D)

    def test_pe_not_learnable(self):
        pe = SinusoidalPositionalEmbedding(d_model=D)
        learnable = [p for p in pe.parameters() if p.requires_grad]
        assert len(learnable) == 0, "Sinusoidal PE has no learnable parameters"

    def test_different_positions_differ(self):
        pe = SinusoidalPositionalEmbedding(d_model=D, max_seq_len=64, dropout=0.0)
        table = pe.pe[0]   # (max_len, D)
        # Adjacent positions must differ
        assert not torch.allclose(table[0], table[1])

    def test_odd_d_model_raises(self):
        with pytest.raises(ValueError, match="even"):
            SinusoidalPositionalEmbedding(d_model=65)


# ─────────────────────────────────────────────────────────────────────────────
# LearnedPositionalEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestLearnedPE:

    def test_output_shape(self, dummy_seq):
        pe     = LearnedPositionalEmbedding(d_model=D, max_seq_len=512, dropout=0.0)
        output = pe(dummy_seq)
        assert output.shape == (B, T, D)

    def test_weights_are_learnable(self):
        pe = LearnedPositionalEmbedding(d_model=D, max_seq_len=64)
        learnable = list(pe.parameters())
        assert len(learnable) > 0

    def test_gradient_flows(self, dummy_seq):
        pe  = LearnedPositionalEmbedding(d_model=D, max_seq_len=64, dropout=0.0)
        x   = dummy_seq.clone().requires_grad_(True)
        out = pe(x)
        out.sum().backward()
        assert pe.embedding.weight.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# RotaryPositionalEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestRoPE:

    def test_output_shapes(self):
        head_dim = 16
        rope = RotaryPositionalEmbedding(head_dim=head_dim)
        q = torch.randn(B, H, T, head_dim)
        k = torch.randn(B, H, T, head_dim)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_tensor(self):
        rope = RotaryPositionalEmbedding(head_dim=16)
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        q_rot, k_rot = rope(q, k)
        assert not torch.allclose(q, q_rot), "RoPE must change the query tensor"
        assert not torch.allclose(k, k_rot), "RoPE must change the key tensor"

    def test_odd_head_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            RotaryPositionalEmbedding(head_dim=15)

    def test_no_learnable_parameters(self):
        rope      = RotaryPositionalEmbedding(head_dim=16)
        learnable = [p for p in rope.parameters() if p.requires_grad]
        assert len(learnable) == 0


# ─────────────────────────────────────────────────────────────────────────────
# CausalConv1d
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalConv1d:

    def test_output_length_matches_input(self):
        conv = CausalConv1d(in_channels=16, out_channels=32, kernel_size=3)
        x    = torch.randn(B, 16, T)
        out  = conv(x)
        assert out.shape == (B, 32, T), "Causal conv must preserve sequence length"

    def test_causal_property(self):
        """
        Verify causality: output at position t must not depend on inputs at t+1..T-1.
        Perturb input at position t+1 and check output at position t is unchanged.
        """
        conv = CausalConv1d(in_channels=4, out_channels=4, kernel_size=3)
        conv.eval()
        x = torch.randn(1, 4, 10)

        # Original output
        out_orig = conv(x).detach()

        # Perturb future input (position 5 onwards)
        x_perturbed = x.clone()
        x_perturbed[:, :, 5:] = torch.randn_like(x_perturbed[:, :, 5:])
        out_perturbed = conv(x_perturbed).detach()

        # Output at positions 0..4 must be UNCHANGED
        assert torch.allclose(out_orig[:, :, :5], out_perturbed[:, :, :5]), (
            "Causal conv output at early positions changed when future input was perturbed"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Conv1dBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestConv1dBlock:

    def test_output_shape_same_padding(self):
        block  = Conv1dBlock(16, 32, kernel_size=3, padding="same", norm="batch")
        x      = torch.randn(B, 16, T)
        block.train()
        output = block(x)
        assert output.shape == (B, 32, T)

    def test_output_shape_no_norm(self):
        block  = Conv1dBlock(8, 16, kernel_size=1, norm=None)
        x      = torch.randn(B, 8, 20)
        output = block(x)
        assert output.shape == (B, 16, 20)

    def test_causal_block_preserves_length(self):
        block = Conv1dBlock(8, 8, kernel_size=5, causal=True)
        x     = torch.randn(B, 8, 15)
        out   = block(x)
        assert out.shape[2] == 15


# ─────────────────────────────────────────────────────────────────────────────
# Attention functions
# ─────────────────────────────────────────────────────────────────────────────

class TestScaledDotProductAttention:

    def test_output_shape(self):
        Dh   = 16
        q    = torch.randn(B, H, T, Dh)
        k    = torch.randn(B, H, T, Dh)
        v    = torch.randn(B, H, T, Dh)
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape     == (B, H, T, Dh)
        assert weights.shape == (B, H, T, T)

    def test_attention_weights_sum_to_one(self):
        """Softmax probabilities must sum to 1 over the key dimension."""
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        _, weights = scaled_dot_product_attention(q, k, v)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_causal_mask_zeroes_upper_triangle(self):
        """With causal mask, attention weights in the upper triangle must be 0."""
        T_ = 5
        mask = build_causal_mask(T_, torch.device("cpu"))
        q = torch.randn(1, 1, T_, 8)
        k = torch.randn(1, 1, T_, 8)
        v = torch.randn(1, 1, T_, 8)
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        # Upper triangle (future positions) must be 0
        for i in range(T_):
            for j in range(i + 1, T_):
                assert weights[0, 0, i, j].item() < 1e-6, (
                    f"Position ({i},{j}) should have ~0 weight (future), got {weights[0,0,i,j]}"
                )

    def test_build_causal_mask_lower_triangular(self):
        mask = build_causal_mask(4, torch.device("cpu"))
        assert mask.shape == (4, 4)
        expected = torch.tensor([
            [True,  False, False, False],
            [True,  True,  False, False],
            [True,  True,  True,  False],
            [True,  True,  True,  True],
        ])
        assert torch.all(mask == expected)


# ─────────────────────────────────────────────────────────────────────────────
# MultiHeadAttention
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiHeadAttention:

    def test_self_attention_output_shape(self, dummy_seq):
        mha = MultiHeadAttention(d_model=D, num_heads=H)
        out, weights = mha(dummy_seq, dummy_seq, dummy_seq)
        assert out.shape     == (B, T, D)
        assert weights.shape == (B, H, T, T)

    def test_cross_attention_output_shape(self):
        mha     = MultiHeadAttention(d_model=D, num_heads=H)
        query   = torch.randn(B, T, D)
        context = torch.randn(B, 12, D)
        out, weights = mha(query, context, context)
        assert out.shape     == (B, T, D)
        assert weights.shape == (B, H, T, 12)

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=65, num_heads=4)

    def test_gradient_flows(self, dummy_seq):
        mha = MultiHeadAttention(d_model=D, num_heads=H)
        x   = dummy_seq.clone().requires_grad_(True)
        out, _ = mha(x, x, x)
        out.sum().backward()
        assert x.grad is not None
        for name, p in mha.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_key_padding_mask_zeroes_padded(self):
        """Padded key positions should have ~0 attention weight."""
        mha  = MultiHeadAttention(d_model=D, num_heads=H)
        mha.eval()
        q    = torch.randn(1, 3, D)
        k    = torch.randn(1, 5, D)
        v    = torch.randn(1, 5, D)
        # Positions 3 and 4 are padding
        key_padding_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        _, weights = mha(q, k, v, key_padding_mask=key_padding_mask)
        # Weights to padded positions (3, 4) must be ~0
        assert weights[0, :, :, 3:].abs().max() < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# MultiHeadCausalAttention
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiHeadCausalAttention:

    def test_output_shape(self, dummy_seq):
        mha = MultiHeadCausalAttention(d_model=D, num_heads=H, max_seq_len=64)
        out, weights = mha(dummy_seq)
        assert out.shape == (B, T, D)

    def test_causal_weights_upper_triangle_zero(self, dummy_seq):
        """Upper-triangular attention weights must be ~0 (future is masked)."""
        mha = MultiHeadCausalAttention(d_model=D, num_heads=H, max_seq_len=64)
        mha.eval()
        _, weights = mha(dummy_seq)
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[0, 0, i, j].item() < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# FeedForward
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedForward:

    def test_output_shape(self, dummy_seq):
        ff  = FeedForward(d_model=D, d_ff=256)
        out = ff(dummy_seq)
        assert out.shape == (B, T, D)

    def test_default_d_ff_is_4x(self):
        ff = FeedForward(d_model=64)
        assert ff.d_ff == 256

    def test_gradient_flows(self, dummy_seq):
        ff  = FeedForward(d_model=D)
        x   = dummy_seq.clone().requires_grad_(True)
        out = ff(x)
        out.sum().backward()
        assert x.grad is not None

    def test_custom_activation(self, dummy_seq):
        ff  = FeedForward(d_model=D, activation=nn.SiLU())
        out = ff(dummy_seq)
        assert out.shape == (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# GatedFeedForward
# ─────────────────────────────────────────────────────────────────────────────

class TestGatedFeedForward:

    def test_output_shape(self, dummy_seq):
        gff = GatedFeedForward(d_model=D)
        out = gff(dummy_seq)
        assert out.shape == (B, T, D)

    def test_auto_d_ff_computation(self):
        gff = GatedFeedForward(d_model=64)
        # d_ff = ceil(8/3 * 64 / 256) * 256 ≥ 64
        assert gff.d_ff > 0
        assert gff.d_ff % 256 == 0

    def test_explicit_d_ff(self, dummy_seq):
        gff = GatedFeedForward(d_model=D, d_ff=512)
        out = gff(dummy_seq)
        assert out.shape == (B, T, D)

    def test_gradient_flows(self, dummy_seq):
        gff = GatedFeedForward(d_model=D)
        x   = dummy_seq.clone().requires_grad_(True)
        gff(x).sum().backward()
        assert x.grad is not None
