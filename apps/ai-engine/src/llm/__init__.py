"""
InsightSerenity AI Engine — LLM Package
========================================
Single import surface for the complete LLM stack.

Pretraining:
    from src.llm import CLMDataset, CLMDataCollator
    from src.llm import MLMDataset, MLMDataCollator

Fine-tuning:
    from src.llm import ChatMLFormatter, AlpacaFormatter
    from src.llm import SFTTrainer, SFTConfig, SFTDataset

Alignment:
    from src.llm import RewardModel, PreferenceDataset, RewardModelTrainer
    from src.llm import RLHFTrainer, RLHFConfig

Inference:
    from src.llm import TextGenerator, Sampler
    from src.llm import KVCache

Prompting:
    from src.llm import ChatTemplate, FewShotTemplate, ChainOfThoughtTemplate
    from src.llm import get_template

Quantization:
    from src.llm import quantize_dynamic_int8, quantize_model_int4
    from src.llm import Int4Linear, estimate_memory_reduction
"""

# ── Pretraining ────────────────────────────────────────────────────────────────
from src.llm.pretraining.clm_objective import (
    CLMDataset, CLMDataCollator, CLMExample,
)
from src.llm.pretraining.mlm_objective import (
    MLMDataset, MLMDataCollator,
)

# ── Fine-tuning ────────────────────────────────────────────────────────────────
from src.llm.finetuning.instruction_formatter import (
    ChatMLFormatter, AlpacaFormatter,
    Conversation, Message,
)
from src.llm.finetuning.sft_trainer import (
    SFTTrainer, SFTConfig, SFTDataset,
)

# ── Alignment (RLHF) ───────────────────────────────────────────────────────────
from src.llm.alignment.reward_model import (
    RewardModel, PreferenceDataset, RewardModelTrainer,
)
from src.llm.alignment.rlhf_trainer import (
    RLHFTrainer, RLHFConfig, PPOBuffer, ValueHead,
)

# ── Inference ──────────────────────────────────────────────────────────────────
from src.llm.inference.kv_cache import KVCache, LayerKVCache
from src.llm.inference.sampler import (
    Sampler,
    greedy_sample, top_k_sample, top_p_sample,
    temperature_scale, repetition_penalty,
)
from src.llm.inference.generator import TextGenerator

# ── Prompting ──────────────────────────────────────────────────────────────────
from src.llm.prompting.templates import (
    SystemPromptTemplate, ChatTemplate, FewShotTemplate,
    ChainOfThoughtTemplate, CodeTemplate, get_template,
)

# ── Quantization ───────────────────────────────────────────────────────────────
from src.llm.quantization.quantizer import (
    quantize_dynamic_int8, quantize_model_int4,
    Int4Linear, estimate_memory_reduction,
)

__all__ = [
    # Pretraining
    "CLMDataset", "CLMDataCollator", "CLMExample",
    "MLMDataset", "MLMDataCollator",
    # Fine-tuning
    "ChatMLFormatter", "AlpacaFormatter", "Conversation", "Message",
    "SFTTrainer", "SFTConfig", "SFTDataset",
    # Alignment
    "RewardModel", "PreferenceDataset", "RewardModelTrainer",
    "RLHFTrainer", "RLHFConfig", "PPOBuffer", "ValueHead",
    # Inference
    "KVCache", "LayerKVCache",
    "Sampler", "greedy_sample", "top_k_sample", "top_p_sample",
    "temperature_scale", "repetition_penalty",
    "TextGenerator",
    # Prompting
    "SystemPromptTemplate", "ChatTemplate", "FewShotTemplate",
    "ChainOfThoughtTemplate", "CodeTemplate", "get_template",
    # Quantization
    "quantize_dynamic_int8", "quantize_model_int4",
    "Int4Linear", "estimate_memory_reduction",
]
