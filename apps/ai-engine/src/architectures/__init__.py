"""
InsightSerenity AI Engine — Architectures Package
==================================================
Single import surface for all model architectures.

Usage:
    from src.architectures import MLP, GPTDecoder, BERTEncoder
    from src.architectures import VanillaRNN, LSTM, GRU
    from src.architectures import ConvNet1d, ConvNet2d
    from src.architectures import Seq2SeqTransformer, MoELayer
"""

from src.architectures.feedforward.mlp import MLP, MLPBlock

from src.architectures.cnn.conv_net import (
    ConvNet1d, ConvNet2d, ConvNet1dConfig, ConvNet2dConfig,
)

from src.architectures.rnn.rnn import VanillaRNN, VanillaRNNCell
from src.architectures.rnn.lstm import LSTM, LSTMCell
from src.architectures.rnn.gru import GRU, GRUCell

from src.architectures.transformer.attention.multi_head_attention import (
    SelfAttention, CausalSelfAttention, CrossAttention,
)
from src.architectures.transformer.encoder.bert_encoder import (
    BERTEncoder, BERTConfig, TransformerEncoderLayer, BERTMLMHead,
)
from src.architectures.transformer.decoder.gpt_decoder import (
    GPTDecoder, GPTConfig, GPTDecoderLayer,
    GPT_SMALL, GPT_MEDIUM, GPT_LARGE,
)
from src.architectures.transformer.encoder_decoder.seq2seq import (
    Seq2SeqTransformer, Seq2SeqConfig,
    Seq2SeqEncoderLayer, Seq2SeqDecoderLayer,
)
from src.architectures.mixture_of_experts.moe_layer import (
    MoELayer, Expert, TopKRouter, SparseMoETransformerLayer,
)

__all__ = [
    # Feedforward
    "MLP", "MLPBlock",
    # CNN
    "ConvNet1d", "ConvNet2d", "ConvNet1dConfig", "ConvNet2dConfig",
    # RNN family
    "VanillaRNN", "VanillaRNNCell",
    "LSTM", "LSTMCell",
    "GRU", "GRUCell",
    # Transformer attention
    "SelfAttention", "CausalSelfAttention", "CrossAttention",
    # BERT encoder
    "BERTEncoder", "BERTConfig", "TransformerEncoderLayer", "BERTMLMHead",
    # GPT decoder
    "GPTDecoder", "GPTConfig", "GPTDecoderLayer",
    "GPT_SMALL", "GPT_MEDIUM", "GPT_LARGE",
    # Seq2Seq
    "Seq2SeqTransformer", "Seq2SeqConfig",
    "Seq2SeqEncoderLayer", "Seq2SeqDecoderLayer",
    # MoE
    "MoELayer", "Expert", "TopKRouter", "SparseMoETransformerLayer",
]
