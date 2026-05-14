#!/usr/bin/env python3
"""
InsightSerenity AI Engine — Main Training Script
=================================================
Entry point for pretraining and fine-tuning the InsightSerenity LLM.

This script wires together:
    - Model construction (GPTDecoder or BERTEncoder)
    - Tokenizer loading
    - Dataset creation (streaming or in-memory)
    - Optimizer + scheduler setup
    - Trainer configuration
    - Checkpoint management
    - Callbacks (logging, early stopping, progress)
    - Distributed training (if --distributed flag is set)

Usage — single GPU:
    python scripts/training/train.py \
        --model gpt-small \
        --tokenizer storage/tokenizers/bpe-32k \
        --data storage/datasets/05_corpus.jsonl \
        --output storage/checkpoints \
        --run-name pretrain-v1 \
        --epochs 3 \
        --batch-size 32 \
        --lr 3e-4

Usage — multi-GPU (4 GPUs via torchrun):
    torchrun --nproc_per_node=4 scripts/training/train.py \
        --model gpt-small \
        --tokenizer storage/tokenizers/bpe-32k \
        --data storage/datasets/05_corpus.jsonl \
        --distributed

Usage — resume from checkpoint:
    python scripts/training/train.py \
        --resume storage/checkpoints/pretrain-v1/step_00010000 \
        [other args...]
"""

import argparse
import hashlib
import json
import math
import random
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add the ai-engine root to sys.path so imports work when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.config.settings import settings
from src.architectures.transformer.decoder.gpt_decoder import (
    GPTDecoder, GPTConfig, GPT_SMALL, GPT_MEDIUM, GPT_LARGE,
)
from src.architectures.transformer.encoder.bert_encoder import BERTEncoder, BERTConfig
from src.tokenizer import load_tokenizer, BPETokenizer
from src.data.datasets.streaming_dataset import StreamingTextDataset
from src.data.datasets.text_dataset import JSONLTextDataset
from src.data.datasets.data_collator import LanguageModelCollator
from src.training import (
    LanguageModelTrainer, LMTrainerConfig,
    CheckpointManager,
    LoggingCallback, EarlyStoppingCallback, ProgressCallback,
    count_parameters, format_param_count,
    setup_distributed, cleanup_distributed, wrap_model_ddp,
    is_main_process, get_rank,
)
from src.nn.optimizers.optimizers import AdamW
from src.nn.schedulers.schedulers import LinearWarmupCosineDecay
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training validation helpers
# ─────────────────────────────────────────────────────────────────────────────

MIN_CORPUS_DOCS = 1_000


def count_corpus_lines(path: str) -> int:
    """Count non-empty lines in a JSONL file without loading it into memory."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def split_corpus(
    data_path:  str,
    val_split:  float,
    test_split: float,
    seed:       int = 42,
) -> tuple:
    """
    Deterministically split a JSONL corpus into train / val / test files.

    Reads the corpus once, shuffles with a fixed seed, and writes three JSONL
    files alongside the source file.  Only creates val/test files if the
    corresponding split fraction is > 0.

    Returns (train_path, val_path_or_None, test_path_or_None).
    """
    source = Path(data_path)
    with open(source, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    rng = random.Random(seed)
    rng.shuffle(lines)

    n        = len(lines)
    n_test   = math.ceil(n * test_split)  if test_split  > 0 else 0
    n_val    = math.ceil(n * val_split)   if val_split   > 0 else 0
    n_train  = n - n_test - n_val

    if n_train <= 0:
        sys.exit(
            f"ERROR: Corpus too small for the requested splits. "
            f"{n} docs → train={n_train}, val={n_val}, test={n_test}. "
            f"Reduce --val-split / --test-split or use a larger corpus."
        )

    stem = source.stem
    out  = source.parent

    test_lines  = lines[:n_test]
    val_lines   = lines[n_test:n_test + n_val]
    train_lines = lines[n_test + n_val:]

    train_path = str(out / f"{stem}.train.jsonl")
    val_path   = str(out / f"{stem}.val.jsonl")   if n_val  > 0 else None
    test_path  = str(out / f"{stem}.test.jsonl")  if n_test > 0 else None

    Path(train_path).write_text("".join(train_lines))
    if val_path:
        Path(val_path).write_text("".join(val_lines))
    if test_path:
        Path(test_path).write_text("".join(test_lines))

    logger.info(
        "Corpus split",
        total=n, train=len(train_lines), val=len(val_lines), test=len(test_lines),
        seed=seed,
    )
    return train_path, val_path, test_path


def write_dataset_card(
    source_path:  str,
    train_path:   str,
    val_path:     str | None,
    test_path:    str | None,
    val_split:    float,
    test_split:   float,
    seed:         int,
) -> None:
    """
    Write a dataset_card.json alongside the source corpus recording
    how the split was performed and the document counts for each split.
    """
    def n_lines(p):
        if not p or not Path(p).exists():
            return 0
        with open(p, encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())

    def md5(p):
        if not p or not Path(p).exists():
            return None
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    card = {
        "source_file":     source_path,
        "train_path":      train_path,
        "val_path":        val_path,
        "test_path":       test_path,
        "train_docs":      n_lines(train_path),
        "val_docs":        n_lines(val_path),
        "test_docs":       n_lines(test_path),
        "total_docs":      n_lines(train_path) + n_lines(val_path) + n_lines(test_path),
        "val_split":       val_split,
        "test_split":      test_split,
        "seed":            seed,
        "checksum_md5":    md5(source_path),
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }

    card_path = Path(source_path).parent / "dataset_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    logger.info("Dataset card written", path=str(card_path))


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gpt-tiny":   GPTConfig(d_model=128, num_heads=4, num_layers=2, max_seq_len=256),
    "gpt-small":  GPT_SMALL,
    "gpt-medium": GPT_MEDIUM,
    "gpt-large":  GPT_LARGE,
}


def build_model(args, vocab_size: int) -> torch.nn.Module:
    """Construct the model from CLI arguments."""
    if args.model in MODEL_REGISTRY:
        config          = MODEL_REGISTRY[args.model]
        config.vocab_size = vocab_size
        config.max_seq_len = args.seq_len
        model = GPTDecoder(config)
    elif args.model == "bert-base":
        config = BERTConfig(
            vocab_size=vocab_size,
            d_model=768, num_heads=12, num_layers=12,
            max_seq_len=args.seq_len,
        )
        model = BERTEncoder(config)
    else:
        raise ValueError(
            f"Unknown model '{args.model}'. "
            f"Available: {list(MODEL_REGISTRY.keys()) + ['bert-base']}"
        )

    n_params = count_parameters(model)
    logger.info(
        "Model built",
        model=args.model,
        params=format_param_count(n_params),
        vocab_size=vocab_size,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """End-to-end training setup and execution."""

    # ── Distributed setup ──────────────────────────────────────────────────
    if args.distributed:
        setup_distributed()

    main_process = is_main_process()

    # ── Tokenizer ──────────────────────────────────────────────────────────
    if main_process:
        logger.info("Loading tokenizer", path=args.tokenizer)
    tokenizer  = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size

    # ── Corpus minimum guard ───────────────────────────────────────────────
    # Refuse immediately if the training corpus is too small to be meaningful.
    # This prevents wasting GPU time on a misconfigured run.
    if main_process:
        doc_count = count_corpus_lines(args.data)
        if doc_count < args.min_corpus_docs:
            sys.exit(
                f"ERROR: Training corpus is too small: {doc_count} documents. "
                f"Minimum required: {args.min_corpus_docs}. "
                f"Add more data or rerun with --min-corpus-docs {doc_count} "
                f"for local experimentation."
            )
        logger.info("Corpus size verified", doc_count=doc_count)

    # ── Auto-split corpus if requested ────────────────────────────────────
    # If --val-split is provided and no explicit --val-data was given,
    # split the corpus deterministically and use the resulting files.
    train_data = args.data
    val_data   = args.val_data

    if main_process and args.val_split > 0 and val_data is None:
        train_data, val_data, _ = split_corpus(
            data_path=args.data,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
        )
        write_dataset_card(
            source_path=args.data,
            train_path=train_data,
            val_path=val_data,
            test_path=None,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
        )
        logger.info("Auto-split complete", train=train_data, val=val_data)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(args, vocab_size)

    # ── Vocab / model mismatch check ───────────────────────────────────────
    # The tokenizer and model embedding must agree on vocab size.
    # Catch this before any GPU memory is allocated.
    actual_embed_vocab = getattr(
        getattr(model, "token_emb", None),
        "num_embeddings",
        None,
    ) or getattr(
        getattr(model, "config", None), "vocab_size", None
    )
    if actual_embed_vocab is not None and actual_embed_vocab != vocab_size:
        sys.exit(
            f"ERROR: Vocab size mismatch.\n"
            f"  Tokenizer: {vocab_size} tokens\n"
            f"  Model embedding: {actual_embed_vocab} rows\n"
            f"  Ensure the tokenizer used here was the one used during model construction."
        )
    logger.info("Vocab/model match verified", vocab_size=vocab_size)

    device = settings.torch_device
    model.to(device)

    if args.distributed:
        model = wrap_model_ddp(model)

    # ── Dataset ────────────────────────────────────────────────────────────
    if main_process:
        logger.info("Loading dataset", path=train_data)

    collator = LanguageModelCollator(
        block_size=args.seq_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Use streaming for large corpora, in-memory for small ones
    if args.streaming:
        train_dataset = StreamingTextDataset(
            path=train_data,
            tokenizer=tokenizer,
            max_length=args.seq_len,
        )
    else:
        train_dataset = JSONLTextDataset(
            path=train_data,
            tokenizer=tokenizer,
            max_length=args.seq_len,
            max_examples=args.max_examples,
        )

    train_sampler = (
        DistributedSampler(train_dataset) if args.distributed and not args.streaming
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and not args.streaming),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Validation loader (either explicit --val-data or from auto-split)
    val_loader = None
    if val_data:
        val_dataset = JSONLTextDataset(
            path=val_data,
            tokenizer=tokenizer,
            max_length=args.seq_len,
            max_examples=args.eval_max_examples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )

    # ── Optimizer + scheduler ───────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Calculate total steps for scheduler
    steps_per_epoch = len(train_loader) // args.gradient_accumulation
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = args.warmup_steps or max(100, total_steps // 20)

    scheduler = LinearWarmupCosineDecay(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    if main_process:
        logger.info(
            "Training plan",
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            steps_per_epoch=steps_per_epoch,
        )

    # ── Trainer configuration ───────────────────────────────────────────────
    config = LMTrainerConfig(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        device=str(device),
        gradient_clip_norm=args.grad_clip,
        gradient_accumulation=args.gradient_accumulation,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        eval_every_n_steps=args.eval_every,
        log_every_n_steps=args.log_every,
        save_every_n_steps=args.save_every,
        output_dir=args.output,
        run_name=args.run_name,
        vocab_size=vocab_size,
        label_smoothing=args.label_smoothing,
    )

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = []

    if main_process:
        log_path = f"{args.output}/{args.run_name}/train_log.jsonl"
        callbacks.append(LoggingCallback(log_path=log_path, log_mlflow=args.mlflow))
        callbacks.append(ProgressCallback(total_steps=total_steps))

        if args.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(patience=args.patience, monitor="val_loss")
            )

    # ── Checkpoint manager ──────────────────────────────────────────────────
    checkpoint_mgr = (
        CheckpointManager(
            output_dir=args.output,
            run_name=args.run_name,
            keep_last_n=args.keep_checkpoints,
        )
        if main_process else None
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = LanguageModelTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=config,
        scheduler=scheduler,
        val_loader=val_loader,
        callbacks=callbacks,
        checkpoint_mgr=checkpoint_mgr,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    final_state = trainer.train(resume_from=args.resume)

    if main_process:
        logger.info(
            "Training complete",
            final_step=final_state.global_step,
            best_val_loss=round(final_state.best_metric, 4),
        )

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if args.distributed:
        cleanup_distributed()


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InsightSerenity LLM training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model", default="gpt-small",
                        help="Model architecture: gpt-tiny | gpt-small | gpt-medium | gpt-large | bert-base")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Maximum sequence length (tokens)")

    # Data
    parser.add_argument("--data", required=True,
                        help="Path to training JSONL corpus")
    parser.add_argument("--val-data", default=None,
                        help="Explicit validation JSONL corpus (skips auto-split)")
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Fraction of corpus to use as validation (0 = disabled). "
                             "Ignored if --val-data is provided.")
    parser.add_argument("--test-split", type=float, default=0.0,
                        help="Fraction of corpus to hold out as test set (default: 0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic corpus splitting.")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to trained tokenizer directory")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming dataset (recommended for large corpora)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap training examples (for debugging)")
    parser.add_argument("--eval-max-examples", type=int, default=10000)
    parser.add_argument("--min-corpus-docs", type=int, default=MIN_CORPUS_DOCS,
                        help="Minimum non-empty JSONL documents required before training")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_epochs with a fixed step count")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="LR warmup steps. Default: 5%% of total steps")
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable automatic mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", default="bfloat16",
                        choices=["float16", "bfloat16"])

    # Checkpointing
    parser.add_argument("--output", default="storage/checkpoints",
                        help="Root directory for checkpoints and logs")
    parser.add_argument("--run-name", default="pretrain-v1")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=5)

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=500)

    # Logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--mlflow", action="store_true",
                        help="Log to MLflow (must be installed and configured)")

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)

    # Distributed
    parser.add_argument("--distributed", action="store_true",
                        help="Enable DDP multi-GPU training (use with torchrun)")

    # DataLoader
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
