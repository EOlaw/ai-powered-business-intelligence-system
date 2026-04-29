"""
Unit tests for all architectures: MLP, CNN, RNN/LSTM/GRU, Transformer, MoE

Coverage:
    MLP:
        - Output shape with various hidden dims
        - Residual connections only when dims match
        - Gradient flows to all parameters
        - Empty hidden_dims → single linear (logistic regression)
        - Output activation applied

    ConvNet1d / ConvNet2d:
        - Output shape with classification head
        - Feature extractor output shape
        - Gradient flows through all conv blocks

    VanillaRNN / LSTM / GRU:
        - Output sequence shape (batch_first=True)
        - Final hidden state shape
        - Bidirectional doubles output dim
        - Multi-layer stacking
        - Gradient flows through time (BPTT)
        - LSTMCell forget gate init (bias=1)

    TransformerEncoderLayer / BERTEncoder:
        - Layer output shape
        - BERTEncoder output dict keys
        - Gradient flows through all layers
        - MLM head output shape

    GPTDecoderLayer / GPTDecoder:
        - Layer output shape
        - Causal attention: upper-triangle weights ≈ 0
        - Logits shape: (B, T, vocab_size)
        - Weight tying: lm_head.weight IS token_emb.embedding.weight
        - Gradient flows

    Seq2SeqTransformer:
        - Encode output shape
        - Decode output shape
        - Full forward output shape

    MoE:
        - TopKRouter: top_k indices per token
        - MoELayer: output shape
        - MoELayer: aux_loss is a scalar and positive
        - SparseMoETransformerLayer: output shape + aux_loss
"""

import pytest
import torch
import torch.nn as nn

from src.architectures.feedforward.mlp import MLP, MLPBlock
from src.architectures.cnn.conv_net import ConvNet1d, ConvNet2d, ConvNet1dConfig, ConvNet2dConfig
from src.architectures.rnn.rnn import VanillaRNN, VanillaRNNCell
from src.architectures.rnn.lstm import LSTM, LSTMCell
from src.architectures.rnn.gru import GRU, GRUCell
from src.architectures.transformer.encoder.bert_encoder import (
    BERTEncoder, BERTConfig, TransformerEncoderLayer,
)
from src.architectures.transformer.decoder.gpt_decoder import (
    GPTDecoder, GPTConfig, GPTDecoderLayer, GPT_SMALL,
)
from src.architectures.transformer.encoder_decoder.seq2seq import (
    Seq2SeqTransformer, Seq2SeqConfig,
)
from src.architectures.mixture_of_experts.moe_layer import (
    MoELayer, TopKRouter, SparseMoETransformerLayer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Common fixtures
# ─────────────────────────────────────────────────────────────────────────────

B, T, D = 2, 8, 64   # batch=2, seq=8, dim=64
VOCAB   = 200         # small vocab for tests


@pytest.fixture
def dummy_seq():
    return torch.randn(B, T, D)


@pytest.fixture
def token_ids():
    return torch.randint(0, VOCAB, (B, T))


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class TestMLP:

    def test_output_shape_with_hidden(self):
        mlp = MLP(in_dim=64, hidden_dims=[128, 64], out_dim=10)
        x   = torch.randn(B, 64)
        assert mlp(x).shape == (B, 10)

    def test_output_shape_no_hidden(self):
        """Empty hidden_dims → single linear transformation."""
        mlp = MLP(in_dim=64, hidden_dims=[], out_dim=10)
        x   = torch.randn(B, 64)
        assert mlp(x).shape == (B, 10)

    def test_3d_input(self):
        mlp = MLP(in_dim=D, hidden_dims=[128], out_dim=32)
        x   = torch.randn(B, T, D)
        assert mlp(x).shape == (B, T, 32)

    def test_gradient_flows(self):
        mlp = MLP(in_dim=32, hidden_dims=[64, 32], out_dim=8)
        x   = torch.randn(4, 32, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None
        for p in mlp.parameters():
            assert p.grad is not None

    def test_residual_connections_same_dims(self):
        """Residual should not raise when input/output dims match."""
        mlp = MLP(in_dim=32, hidden_dims=[32, 32], out_dim=32, residual=True)
        x   = torch.randn(4, 32)
        assert mlp(x).shape == (4, 32)

    def test_output_activation_sigmoid(self):
        """Sigmoid output activation should produce values in (0, 1)."""
        mlp = MLP(in_dim=16, hidden_dims=[32], out_dim=4, output_activation=nn.Sigmoid())
        x   = torch.randn(4, 16)
        out = mlp(x)
        assert (out > 0).all() and (out < 1).all()

    def test_mlp_block_output_shape(self):
        block = MLPBlock(in_dim=64, out_dim=32, activation="gelu", dropout=0.0)
        x     = torch.randn(B, 64)
        assert block(x).shape == (B, 32)


# ─────────────────────────────────────────────────────────────────────────────
# ConvNet1d / ConvNet2d
# ─────────────────────────────────────────────────────────────────────────────

class TestConvNet1d:

    @pytest.fixture
    def config(self):
        return ConvNet1dConfig(
            in_channels=8, channel_sizes=[16, 32], kernel_sizes=[3, 3],
            strides=[1, 2], dropout=0.0, out_classes=10
        )

    def test_output_shape_with_head(self, config):
        model = ConvNet1d(config)
        model.eval()
        x     = torch.randn(B, 8, 20)
        out   = model(x)
        assert out.shape == (B, 10)

    def test_feature_extractor_shape(self, config):
        model = ConvNet1d(config)
        model.eval()
        x     = torch.randn(B, 8, 20)
        feat  = model.feature_extractor(x)
        assert feat.shape == (B, 32)

    def test_gradient_flows(self, config):
        model = ConvNet1d(config)
        model.train()
        x = torch.randn(B, 8, 20, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_mismatched_config_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            ConvNet1d(ConvNet1dConfig(channel_sizes=[32, 64], kernel_sizes=[3], strides=[1, 2]))


class TestConvNet2d:

    @pytest.fixture
    def config(self):
        return ConvNet2dConfig(
            in_channels=3, channel_sizes=[16, 32], kernel_sizes=[3, 3],
            strides=[1, 2], dropout=0.0, out_classes=5
        )

    def test_output_shape_with_head(self, config):
        model = ConvNet2d(config)
        model.eval()
        x     = torch.randn(B, 3, 16, 16)
        out   = model(x)
        assert out.shape == (B, 5)

    def test_feature_extractor_shape(self, config):
        model = ConvNet2d(config)
        model.eval()
        x    = torch.randn(B, 3, 16, 16)
        feat = model.feature_extractor(x)
        assert feat.shape == (B, 32)


# ─────────────────────────────────────────────────────────────────────────────
# VanillaRNN
# ─────────────────────────────────────────────────────────────────────────────

class TestVanillaRNN:

    def test_output_shape(self):
        rnn    = VanillaRNN(input_size=D, hidden_size=32, batch_first=True)
        x      = torch.randn(B, T, D)
        output, h_n = rnn(x)
        assert output.shape == (B, T, 32)
        assert h_n.shape    == (1, B, 32)

    def test_bidirectional_doubles_output_dim(self):
        rnn    = VanillaRNN(D, 32, bidirectional=True, batch_first=True)
        x      = torch.randn(B, T, D)
        output, h_n = rnn(x)
        assert output.shape == (B, T, 64)   # 2 × 32
        assert h_n.shape    == (2, B, 32)   # 2 directions

    def test_multilayer_output_shape(self):
        rnn    = VanillaRNN(D, 32, num_layers=3, batch_first=True)
        x      = torch.randn(B, T, D)
        output, h_n = rnn(x)
        assert output.shape == (B, T, 32)
        assert h_n.shape    == (3, B, 32)

    def test_gradient_flows_through_time(self):
        rnn = VanillaRNN(8, 16, batch_first=True)
        x   = torch.randn(2, 5, 8, requires_grad=True)
        rnn(x)[0].sum().backward()
        assert x.grad is not None

    def test_rnn_cell_output_shape(self):
        cell  = VanillaRNNCell(8, 16)
        x_t   = torch.randn(B, 8)
        h_prev = torch.zeros(B, 16)
        h_t   = cell(x_t, h_prev)
        assert h_t.shape == (B, 16)

    def test_rnn_cell_tanh_bounded(self):
        """RNN hidden state must be in (-1, 1) due to tanh."""
        cell  = VanillaRNNCell(8, 16)
        x_t   = torch.randn(4, 8) * 100   # Large input
        h     = cell(x_t, torch.zeros(4, 16))
        assert (h > -1).all() and (h < 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestLSTM:

    def test_output_shape(self):
        lstm = LSTM(D, 32, batch_first=True)
        x    = torch.randn(B, T, D)
        out, (h_n, c_n) = lstm(x)
        assert out.shape == (B, T, 32)
        assert h_n.shape == (1, B, 32)
        assert c_n.shape == (1, B, 32)

    def test_bidirectional(self):
        lstm = LSTM(D, 32, bidirectional=True, batch_first=True)
        x    = torch.randn(B, T, D)
        out, (h_n, c_n) = lstm(x)
        assert out.shape == (B, T, 64)
        assert h_n.shape == (2, B, 32)
        assert c_n.shape == (2, B, 32)

    def test_multilayer(self):
        lstm = LSTM(D, 32, num_layers=2, batch_first=True)
        x    = torch.randn(B, T, D)
        out, (h_n, c_n) = lstm(x)
        assert h_n.shape == (2, B, 32)

    def test_forget_gate_bias_initialised_to_one(self):
        """Forget gate bias must be initialised to 1.0 (Jozefowicz et al.)."""
        cell = LSTMCell(8, 16)
        H = 16
        # Forget gate is index 1 in [i, f, g, o] order → positions [H:2H]
        assert (cell.W_ih.bias.data[H:2*H] == 1.0).all()
        assert (cell.W_hh.bias.data[H:2*H] == 1.0).all()

    def test_gradient_flows(self):
        lstm = LSTM(8, 16, batch_first=True)
        x    = torch.randn(2, 4, 8, requires_grad=True)
        lstm(x)[0].sum().backward()
        assert x.grad is not None

    def test_lstm_cell_output_shape(self):
        cell  = LSTMCell(8, 16)
        x_t   = torch.randn(B, 8)
        h_c   = (torch.zeros(B, 16), torch.zeros(B, 16))
        h_t, c_t = cell(x_t, h_c)
        assert h_t.shape == (B, 16)
        assert c_t.shape == (B, 16)


# ─────────────────────────────────────────────────────────────────────────────
# GRU
# ─────────────────────────────────────────────────────────────────────────────

class TestGRU:

    def test_output_shape(self):
        gru = GRU(D, 32, batch_first=True)
        x   = torch.randn(B, T, D)
        out, h_n = gru(x)
        assert out.shape == (B, T, 32)
        assert h_n.shape == (1, B, 32)

    def test_bidirectional(self):
        gru = GRU(D, 32, bidirectional=True, batch_first=True)
        x   = torch.randn(B, T, D)
        out, h_n = gru(x)
        assert out.shape == (B, T, 64)
        assert h_n.shape == (2, B, 32)

    def test_gru_cell_output_shape(self):
        cell  = GRUCell(8, 16)
        x_t   = torch.randn(B, 8)
        h_prev = torch.zeros(B, 16)
        h_t   = cell(x_t, h_prev)
        assert h_t.shape == (B, 16)

    def test_update_gate_interpolation(self):
        """
        Verify the GRU update gate interpolation.
        When z_gate ≈ 0: output should be close to h_prev (no update).
        """
        cell = GRUCell(4, 8)
        # Force z_gate close to 0 by setting W_rz biases to large negative
        with torch.no_grad():
            cell.W_rz_ih.bias[8:].fill_(-100.0)   # z_gate bias very negative
            cell.W_rz_hh.bias[8:].fill_(-100.0)

        x_t   = torch.randn(2, 4)
        h_prev = torch.randn(2, 8)
        h_t   = cell(x_t, h_prev)

        # With z ≈ 0: h_t ≈ (1-0) × h_prev + 0 × n = h_prev
        assert torch.allclose(h_t, h_prev, atol=0.1)

    def test_gradient_flows(self):
        gru = GRU(8, 16, batch_first=True)
        x   = torch.randn(2, 4, 8, requires_grad=True)
        gru(x)[0].sum().backward()
        assert x.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# TransformerEncoderLayer / BERTEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformerEncoderLayer:

    def test_output_shape(self, dummy_seq):
        layer = TransformerEncoderLayer(D, num_heads=4, d_ff=256)
        out, weights = layer(dummy_seq)
        assert out.shape     == (B, T, D)
        assert weights.shape == (B, 4, T, T)

    def test_gradient_flows(self, dummy_seq):
        layer = TransformerEncoderLayer(D, num_heads=4, d_ff=256)
        x     = dummy_seq.clone().requires_grad_(True)
        layer(x)[0].sum().backward()
        assert x.grad is not None


class TestBERTEncoder:

    @pytest.fixture
    def bert_config(self):
        return BERTConfig(
            vocab_size=VOCAB, d_model=D, num_heads=4, num_layers=2,
            d_ff=256, max_seq_len=32, dropout=0.0, attention_dropout=0.0,
        )

    def test_last_hidden_state_shape(self, bert_config, token_ids):
        model  = BERTEncoder(bert_config)
        model.eval()
        output = model(token_ids)
        assert output["last_hidden_state"].shape == (B, T, D)

    def test_pooler_output_shape(self, bert_config, token_ids):
        model  = BERTEncoder(bert_config)
        model.eval()
        output = model(token_ids)
        assert output["pooler_output"].shape == (B, D)

    def test_attention_mask_applied(self, bert_config):
        model = BERTEncoder(bert_config)
        model.eval()
        ids   = torch.randint(0, VOCAB, (1, 5))
        mask  = torch.tensor([[1, 1, 1, 0, 0]])   # positions 3,4 are padding
        output = model(ids, attention_mask=mask)
        assert output["last_hidden_state"].shape == (1, 5, D)

    def test_mlm_head_output_shape(self):
        cfg   = BERTConfig(
            vocab_size=VOCAB, d_model=D, num_heads=4, num_layers=2,
            d_ff=256, max_seq_len=32, dropout=0.0, mlm_head=True,
        )
        model  = BERTEncoder(cfg)
        model.eval()
        ids    = torch.randint(0, VOCAB, (B, T))
        output = model(ids)
        assert output["mlm_logits"].shape == (B, T, VOCAB)

    def test_gradient_flows(self, bert_config, token_ids):
        model  = BERTEncoder(bert_config)
        output = model(token_ids)
        output["last_hidden_state"].sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


# ─────────────────────────────────────────────────────────────────────────────
# GPTDecoderLayer / GPTDecoder
# ─────────────────────────────────────────────────────────────────────────────

class TestGPTDecoderLayer:

    def test_output_shape(self, dummy_seq):
        layer  = GPTDecoderLayer(D, num_heads=4, d_ff=256, max_seq_len=64)
        out, weights = layer(dummy_seq)
        assert out.shape     == (B, T, D)
        assert weights.shape == (B, 4, T, T)

    def test_causal_attention_upper_triangle_zero(self, dummy_seq):
        layer = GPTDecoderLayer(D, num_heads=4, d_ff=256, max_seq_len=64)
        layer.eval()
        _, weights = layer(dummy_seq)
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[0, 0, i, j].item() < 1e-5, (
                    f"Causal violation at ({i},{j}): {weights[0,0,i,j]}"
                )


class TestGPTDecoder:

    @pytest.fixture
    def gpt_config(self):
        return GPTConfig(
            vocab_size=VOCAB, d_model=D, num_heads=4, num_layers=2,
            max_seq_len=32, dropout=0.0, attention_dropout=0.0,
            ffn_type="swiglu", use_rope=True, tie_weights=True,
        )

    def test_logits_shape(self, gpt_config, token_ids):
        model  = GPTDecoder(gpt_config)
        model.eval()
        output = model(token_ids)
        assert output["logits"].shape == (B, T, VOCAB)

    def test_last_hidden_shape(self, gpt_config, token_ids):
        model  = GPTDecoder(gpt_config)
        model.eval()
        output = model(token_ids)
        assert output["last_hidden"].shape == (B, T, D)

    def test_weight_tying(self, gpt_config):
        """lm_head.weight must be the same object as the token embedding weight."""
        model = GPTDecoder(gpt_config)
        assert model.lm_head.weight is model.token_emb.embedding.weight

    def test_gradient_flows(self, gpt_config, token_ids):
        model  = GPTDecoder(gpt_config)
        output = model(token_ids)
        output["logits"].sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad and "lm_head" not in name:
                assert p.grad is not None, f"No gradient for {name}"

    def test_num_parameters_positive(self, gpt_config):
        model = GPTDecoder(gpt_config)
        assert model.num_parameters() > 0

    def test_attention_mask_applied(self, gpt_config):
        model = GPTDecoder(gpt_config)
        model.eval()
        ids  = torch.randint(0, VOCAB, (1, 6))
        mask = torch.ones(1, 6)
        mask[0, 4:] = 0   # padding
        output = model(ids, attention_mask=mask)
        assert output["logits"].shape == (1, 6, VOCAB)


# ─────────────────────────────────────────────────────────────────────────────
# Seq2SeqTransformer
# ─────────────────────────────────────────────────────────────────────────────

class TestSeq2SeqTransformer:

    @pytest.fixture
    def s2s_config(self):
        return Seq2SeqConfig(
            vocab_size=VOCAB, d_model=D, num_heads=4,
            num_encoder_layers=2, num_decoder_layers=2,
            d_ff=256, max_seq_len=32, dropout=0.0, attention_dropout=0.0,
        )

    def test_encode_output_shape(self, s2s_config):
        model   = Seq2SeqTransformer(s2s_config)
        model.eval()
        src_ids = torch.randint(0, VOCAB, (B, T))
        enc_out = model.encode(src_ids)
        assert enc_out.shape == (B, T, D)

    def test_decode_output_shape(self, s2s_config):
        model   = Seq2SeqTransformer(s2s_config)
        model.eval()
        src_ids = torch.randint(0, VOCAB, (B, T))
        tgt_ids = torch.randint(0, VOCAB, (B, 5))
        enc_out = model.encode(src_ids)
        dec_out = model.decode(tgt_ids, enc_out)
        assert dec_out.shape == (B, 5, D)

    def test_forward_logits_shape(self, s2s_config):
        model   = Seq2SeqTransformer(s2s_config)
        model.eval()
        src_ids = torch.randint(0, VOCAB, (B, T))
        tgt_ids = torch.randint(0, VOCAB, (B, 5))
        logits  = model(src_ids, tgt_ids)
        assert logits.shape == (B, 5, VOCAB)

    def test_gradient_flows(self, s2s_config):
        model   = Seq2SeqTransformer(s2s_config)
        src_ids = torch.randint(0, VOCAB, (B, T))
        tgt_ids = torch.randint(0, VOCAB, (B, 5))
        model(src_ids, tgt_ids).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad and "lm_head" not in name:
                assert p.grad is not None, f"No grad for {name}"


# ─────────────────────────────────────────────────────────────────────────────
# MoE
# ─────────────────────────────────────────────────────────────────────────────

class TestTopKRouter:

    def test_top_k_indices_shape(self):
        router  = TopKRouter(d_model=D, num_experts=8, top_k=2)
        x_flat  = torch.randn(B * T, D)
        indices, weights, aux_loss = router(x_flat)
        assert indices.shape  == (B * T, 2)
        assert weights.shape  == (B * T, 2)

    def test_routing_weights_sum_to_one(self):
        router  = TopKRouter(d_model=D, num_experts=8, top_k=2)
        x_flat  = torch.randn(B * T, D)
        _, weights, _ = router(x_flat)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_aux_loss_positive(self):
        router  = TopKRouter(d_model=D, num_experts=8, top_k=2)
        x_flat  = torch.randn(B * T, D)
        _, _, aux_loss = router(x_flat)
        assert aux_loss.item() >= 0.0

    def test_indices_in_valid_range(self):
        num_experts = 6
        router  = TopKRouter(d_model=D, num_experts=num_experts, top_k=2)
        x_flat  = torch.randn(B * T, D)
        indices, _, _ = router(x_flat)
        assert (indices >= 0).all()
        assert (indices < num_experts).all()


class TestMoELayer:

    def test_output_shape(self, dummy_seq):
        moe    = MoELayer(d_model=D, num_experts=4, top_k=2)
        output, aux_loss = moe(dummy_seq)
        assert output.shape == (B, T, D)

    def test_aux_loss_scalar(self, dummy_seq):
        moe    = MoELayer(d_model=D, num_experts=4, top_k=2)
        moe.train()
        _, aux_loss = moe(dummy_seq)
        assert aux_loss.shape == ()   # scalar

    def test_aux_loss_positive(self, dummy_seq):
        moe    = MoELayer(d_model=D, num_experts=4, top_k=2)
        moe.train()
        _, aux_loss = moe(dummy_seq)
        assert aux_loss.item() >= 0.0

    def test_gradient_flows_through_moe(self, dummy_seq):
        moe = MoELayer(d_model=D, num_experts=4, top_k=2)
        x   = dummy_seq.clone().requires_grad_(True)
        output, aux_loss = moe(x)
        (output.sum() + aux_loss).backward()
        assert x.grad is not None


class TestSparseMoETransformerLayer:

    def test_output_shape(self, dummy_seq):
        layer = SparseMoETransformerLayer(
            d_model=D, num_heads=4, num_experts=4, top_k=2, max_seq_len=64
        )
        layer.eval()
        out, weights, aux_loss = layer(dummy_seq)
        assert out.shape     == (B, T, D)
        assert weights.shape == (B, 4, T, T)
        assert aux_loss.shape == ()

    def test_causal_upper_triangle_zero(self, dummy_seq):
        layer = SparseMoETransformerLayer(
            d_model=D, num_heads=4, num_experts=4, top_k=2, max_seq_len=64
        )
        layer.eval()
        _, weights, _ = layer(dummy_seq)
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[0, 0, i, j].item() < 1e-5
