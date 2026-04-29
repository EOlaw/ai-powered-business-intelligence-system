"""
InsightSerenity AI Engine — Supervised Fine-Tuning Trainer
============================================================
Supervised Fine-Tuning (SFT) adapts a pretrained language model to follow
instructions. The model already knows language from pretraining; SFT teaches
it the FORMAT of question → answer conversations.

Key difference from pretraining:
    - Pretraining: compute loss on ALL tokens (predict next token everywhere)
    - SFT:         compute loss ONLY on assistant response tokens
                   (the model should learn to generate responses, not mimic
                    the entire conversation including the human's questions)

The masking of non-response tokens is critical. Without it, the model would
treat reproducing the system prompt and user messages as part of the task,
which wastes gradient signal and can make the model echo the user.

Dataset format (JSONL):
    {"system": "...", "user": "...", "assistant": "..."}
    or multi-turn:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    from src.llm.finetuning.sft_trainer import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=pretrained_gpt,
        tokenizer=tokenizer,
        config=SFTConfig(max_epochs=3, lr=1e-5),
        train_data_path="storage/datasets/instructions.jsonl",
    )
    trainer.train()
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from src.training.trainer.language_model_trainer import LanguageModelTrainer, LMTrainerConfig
from src.llm.finetuning.instruction_formatter import ChatMLFormatter, AlpacaFormatter
from src.tokenizer.special_tokens import SpecialTokens as ST
from src.utils.file_io import iter_jsonl
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SFTConfig(LMTrainerConfig):
    """Configuration for supervised fine-tuning."""
    # Template: "chatml" (our primary) or "alpaca"
    template:          str   = "chatml"
    # Maximum sequence length including prompt + response
    max_seq_len:       int   = 2048
    # Loss on user/system turns? False = only assistant turns contribute
    loss_on_prompt:    bool  = False
    # Shuffle the training data
    shuffle:           bool  = True


class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning from JSONL instruction data.

    Reads instruction/response pairs, formats them using the configured
    template, tokenises the full conversation, and creates a loss mask
    that is True only at assistant response positions.

    Args:
        data_path:    Path to JSONL instruction dataset.
        tokenizer:    Trained tokenizer.
        formatter:    ChatMLFormatter or AlpacaFormatter instance.
        max_seq_len:  Maximum token length (longer examples are truncated).
        loss_on_prompt: If False, mask out non-response tokens from loss.
        shuffle:      Shuffle examples. Default True.
    """

    def __init__(
        self,
        data_path:      str,
        tokenizer:      Any,
        formatter:      Any,
        max_seq_len:    int  = 2048,
        loss_on_prompt: bool = False,
        shuffle:        bool = True,
    ) -> None:
        self._tokenizer      = tokenizer
        self._formatter      = formatter
        self._max_seq_len    = max_seq_len
        self._loss_on_prompt = loss_on_prompt

        logger.info("Loading SFT dataset", path=data_path)
        raw_examples = list(iter_jsonl(data_path))
        if shuffle:
            random.shuffle(raw_examples)

        self._examples: List[Dict[str, Tensor]] = []
        skipped = 0

        for example in raw_examples:
            processed = self._process(example)
            if processed is not None:
                self._examples.append(processed)
            else:
                skipped += 1

        logger.info(
            "SFT dataset ready",
            examples=len(self._examples),
            skipped=skipped,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return self._examples[idx]

    def _process(self, example: Dict[str, Any]) -> Optional[Dict[str, Tensor]]:
        """
        Tokenise and mask one instruction example.

        Returns None if the example is empty or too short to be useful.
        """
        # Format the conversation into a string
        if isinstance(self._formatter, ChatMLFormatter):
            if "user" in example and "assistant" in example:
                text = self._formatter.format_example(example)
            elif "messages" in example:
                from src.llm.finetuning.instruction_formatter import Conversation, Message
                conv = Conversation(system=example.get("system"))
                for msg in example["messages"]:
                    conv.messages.append(Message(role=msg["role"], content=msg["content"]))
                text = self._formatter.format_conversation(conv)
            else:
                return None
        else:
            text = self._formatter.format_example(example)

        if not text.strip():
            return None

        # Tokenise (without adding BOS/EOS — the template already has them)
        ids = self._tokenizer.encode(text, add_special_tokens=False)

        if len(ids) < 4:
            return None

        # Truncate to max_seq_len
        ids = ids[:self._max_seq_len]
        seq_len = len(ids)

        # Pad to max_seq_len
        pad_len    = self._max_seq_len - seq_len
        padded_ids = ids + [ST.PAD_ID] * pad_len
        attn_mask  = [1] * seq_len + [0] * pad_len

        # Build labels: by default, all positions predict the next token
        # But we may mask out prompt/system/user tokens
        labels = padded_ids[1:] + [ST.EOS_ID]

        if not self._loss_on_prompt:
            # Get the assistant token ID
            assistant_id = self._tokenizer.vocab.token_to_id(ST.ASSISTANT)
            end_turn_id  = self._tokenizer.vocab.token_to_id(ST.END_TURN)

            response_mask = self._formatter.get_response_mask_positions(
                ids, assistant_id, end_turn_id
            )

            # Mask out non-response tokens in labels (set to -100)
            masked_labels = []
            for i, (label, is_response) in enumerate(zip(labels, response_mask + [False])):
                if is_response and padded_ids[i] != ST.PAD_ID:
                    masked_labels.append(label)
                else:
                    masked_labels.append(-100)

            labels = masked_labels

        # Pad labels
        if len(labels) < self._max_seq_len:
            labels += [-100] * (self._max_seq_len - len(labels))
        labels = labels[:self._max_seq_len]

        return {
            "input_ids":      torch.tensor(padded_ids, dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
        }


class SFTTrainer:
    """
    Complete supervised fine-tuning setup.

    Wraps the full pipeline: dataset loading → collation → training.
    Uses LanguageModelTrainer internally for the training loop.

    Args:
        model:           Pretrained GPTDecoder.
        tokenizer:       Trained tokenizer.
        config:          SFTConfig.
        train_data_path: Path to training JSONL.
        val_data_path:   Optional path to validation JSONL.
        callbacks:       Optional list of training callbacks.
        checkpoint_mgr:  Optional CheckpointManager.
    """

    def __init__(
        self,
        model:           nn.Module,
        tokenizer:       Any,
        config:          SFTConfig,
        train_data_path: str,
        val_data_path:   Optional[str] = None,
        callbacks:       Optional[list] = None,
        checkpoint_mgr:  Optional[Any]  = None,
    ) -> None:
        self.config    = config
        self.tokenizer = tokenizer

        # Select formatter
        if config.template == "chatml":
            formatter = ChatMLFormatter()
        elif config.template == "alpaca":
            formatter = AlpacaFormatter()
        else:
            raise ValueError(f"Unknown template '{config.template}'. Expected: chatml | alpaca")

        # Build datasets
        self._train_dataset = SFTDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            formatter=formatter,
            max_seq_len=config.max_seq_len,
            loss_on_prompt=config.loss_on_prompt,
            shuffle=config.shuffle,
        )

        self._val_dataset = (
            SFTDataset(
                data_path=val_data_path,
                tokenizer=tokenizer,
                formatter=formatter,
                max_seq_len=config.max_seq_len,
                loss_on_prompt=config.loss_on_prompt,
                shuffle=False,
            )
            if val_data_path else None
        )

        # DataLoaders (SFT examples are already padded to max_seq_len)
        from torch.utils.data.dataloader import default_collate
        train_loader = DataLoader(
            self._train_dataset,
            batch_size=config.gradient_accumulation,
            shuffle=True,
            num_workers=2,
        )
        val_loader = (
            DataLoader(self._val_dataset, batch_size=config.gradient_accumulation, shuffle=False)
            if self._val_dataset else None
        )

        from src.nn.optimizers.optimizers import AdamW
        from src.nn.schedulers.schedulers import LinearWarmupCosineDecay

        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps  = len(train_loader) * config.max_epochs
        warmup_steps = max(50, total_steps // 10)

        scheduler = LinearWarmupCosineDecay(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self._trainer = LanguageModelTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            scheduler=scheduler,
            val_loader=val_loader,
            callbacks=callbacks or [],
            checkpoint_mgr=checkpoint_mgr,
        )

    def train(self, resume_from: Optional[str] = None):
        """Start the SFT training run."""
        logger.info(
            "SFT training started",
            train_examples=len(self._train_dataset),
            val_examples=len(self._val_dataset) if self._val_dataset else 0,
        )
        return self._trainer.train(resume_from=resume_from)
