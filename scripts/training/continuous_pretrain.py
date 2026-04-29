"""
InsightSerenity — Continuous Pretraining Script
================================================
Incrementally pretrain the language model on newly crawled data.

This script is called by the worker's train job. It:
    1. Loads the base model checkpoint from the model registry
    2. Builds a DataLoader over the new JSONL dataset
    3. Runs CLM pretraining for a fixed number of steps (not epochs)
    4. Saves the updated checkpoint to the output directory
    5. Writes a JSON summary of training metrics

Usage:
    python scripts/training/continuous_pretrain.py \\
        --dataset-path storage/datasets/processed/latest \\
        --base-model   storage/models/insightserenity-1/latest \\
        --output-dir   storage/models/insightserenity-1/v1.3.0 \\
        --max-steps    2000 \\
        --mode         finetune
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.config.settings import settings
from src.utils.logger     import get_logger
from src.serving.registry.model_registry import ModelRegistry

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous pretraining for InsightSerenity")
    p.add_argument("--dataset-path", required=True, help="Path to preprocessed JSONL training data")
    p.add_argument("--base-model",   required=True, help="Path to the base model directory")
    p.add_argument("--output-dir",   required=True, help="Where to save the new model weights")
    p.add_argument("--max-steps",    type=int,  default=2000,   help="Number of gradient steps")
    p.add_argument("--batch-size",   type=int,  default=8,      help="Training batch size")
    p.add_argument("--lr",           type=float, default=5e-5,  help="Peak learning rate")
    p.add_argument("--warmup-steps", type=int,  default=100,    help="LR warmup steps")
    p.add_argument("--grad-accum",   type=int,  default=4,      help="Gradient accumulation steps")
    p.add_argument("--mode", choices=["pretrain", "finetune"], default="finetune")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> dict:
    device     = settings.torch_device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model", path=args.base_model)

    # Load the base model and tokenizer from disk
    registry = ModelRegistry(models_dir=str(settings.storage.models))
    model_name = Path(args.base_model).name
    try:
        model, tokenizer = registry.get_model(model_name)
    except KeyError:
        # Fall back to direct path load
        from src.architectures.transformer.decoder.gpt_decoder import GPTDecoder, GPT_SMALL
        model     = GPTDecoder(GPT_SMALL)
        tokenizer = None
        logger.warning("Model not in registry, using random weights", model=model_name)

    model.to(device)
    model.train()

    # ── Dataset ─────────────────────────────────────────────────────────────
    from src.data.datasets.text_dataset import JSONLTextDataset
    from src.data.datasets.data_collator import LanguageModelCollator
    from torch.utils.data import DataLoader

    dataset = JSONLTextDataset(
        path=args.dataset_path,
        tokenizer=tokenizer,
        max_length=settings.tokenizer.max_length,
    )

    if len(dataset) == 0:
        logger.warning("Dataset is empty — skipping training", path=args.dataset_path)
        return {"steps": 0, "final_loss": 0.0, "perplexity": float("inf"), "model_path": str(output_dir)}

    collator   = LanguageModelCollator(pad_token_id=0, max_length=settings.tokenizer.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collator, num_workers=2, pin_memory=True)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    from src.nn.optimizers.optimizers import AdamW
    from src.nn.schedulers.schedulers import LinearWarmupCosineDecay

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineDecay(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        min_lr_ratio=0.1,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    scaler        = torch.amp.GradScaler(enabled=device.type == "cuda")
    global_step   = 0
    total_loss    = 0.0
    log_interval  = 50
    start_time    = time.time()

    data_iter = iter(dataloader)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)   # Restart dataset
            batch = next(data_iter)

        input_ids  = batch["input_ids"].to(device)
        labels     = batch["labels"].to(device)
        attn_mask  = batch.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=device.type in ("cuda", "mps")):
            output = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = output["logits"]

            # Shift for CLM: predict token[t] from token[t-1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * args.grad_accum

        if (global_step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed  = time.time() - start_time
            logger.info(
                "Training progress",
                step=global_step,
                total=args.max_steps,
                loss=round(avg_loss, 4),
                elapsed_s=round(elapsed, 1),
                steps_per_s=round(global_step / elapsed, 1),
            )
            total_loss = 0.0

    # ── Save checkpoint ──────────────────────────────────────────────────────
    model.eval()
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Copy tokenizer and config if they exist
    base_dir = Path(args.base_model)
    for fname in ["config.json", "tokenizer_config.json"]:
        src = base_dir / fname
        if src.exists():
            (output_dir / fname).write_bytes(src.read_bytes())

    tok_src = base_dir / "tokenizer"
    if tok_src.exists():
        import shutil
        shutil.copytree(str(tok_src), str(output_dir / "tokenizer"), dirs_exist_ok=True)

    final_loss = total_loss / max(global_step % log_interval or log_interval, 1)
    perplexity = min(float(torch.exp(torch.tensor(final_loss)).item()), 10_000)

    summary = {
        "model_path": str(output_dir),
        "steps":      global_step,
        "final_loss": round(final_loss, 4),
        "perplexity": round(perplexity, 2),
        "mode":       args.mode,
        "elapsed_s":  round(time.time() - start_time, 1),
    }

    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Training complete", **summary)
    return summary


if __name__ == "__main__":
    args = parse_args()
    result = run_training(args)
    print(json.dumps(result))
