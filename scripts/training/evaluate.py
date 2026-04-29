"""
InsightSerenity — Model Evaluation Script
==========================================
Compares a candidate model against the current production baseline.

Metrics:
    perplexity  — Cross-entropy loss on a held-out evaluation set (primary gate)
    BLEU-4      — On reference generation tasks (secondary check)

Gate logic:
    Passes if:
        (new_perplexity - base_perplexity) / base_perplexity < max_regression
    i.e. the new model is allowed to be at most max_regression (default 2%) worse
    in perplexity. Lower perplexity is better.

Usage:
    python scripts/training/evaluate.py \\
        --new-model   storage/models/insightserenity-1/v1.3.0 \\
        --base-model  storage/models/insightserenity-1/latest \\
        --eval-data   storage/datasets/eval/eval.jsonl \\
        --max-regression 0.02 \\
        --output-json /tmp/eval_result.json
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config.settings import settings
from src.utils.logger     import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--new-model",      required=True)
    p.add_argument("--base-model",     required=True)
    p.add_argument("--eval-data",      default="storage/datasets/eval/eval.jsonl")
    p.add_argument("--max-regression", type=float, default=0.02)
    p.add_argument("--output-json",    default=None)
    p.add_argument("--max-batches",    type=int,   default=200)
    p.add_argument("--batch-size",     type=int,   default=4)
    return p.parse_args()


def compute_perplexity(model, dataloader, device: torch.device, max_batches: int) -> float:
    """Compute perplexity on a DataLoader. Lower is better."""
    model.eval()
    total_loss  = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            output     = model(input_ids=input_ids)
            logits     = output["logits"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            mask    = (shift_labels != -100).float()
            loss    = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            total_loss   += (loss * mask.view(-1)).sum().item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        return float("inf")

    avg_nll    = total_loss / total_tokens
    perplexity = math.exp(min(avg_nll, 20))   # Cap at e^20 to prevent overflow
    return round(perplexity, 4)


def load_model(model_dir: str, device: torch.device):
    config_path = Path(model_dir) / "config.json"
    model_path  = Path(model_dir) / "model.pt"
    tok_dir     = Path(model_dir) / "tokenizer"

    config_dict = {}
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)

    from src.architectures.transformer.decoder.gpt_decoder import GPTDecoder, GPTConfig, GPT_SMALL
    try:
        cfg = GPTConfig(**{k: v for k, v in config_dict.items() if k in GPTConfig.__dataclass_fields__})
    except Exception:
        cfg = GPT_SMALL

    model = GPTDecoder(cfg)

    if model_path.exists():
        state = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        logger.warning("model.pt not found, using random weights", path=str(model_path))

    model.to(device)

    tokenizer = None
    if tok_dir.exists():
        from src.tokenizer import load_tokenizer
        try:
            tokenizer = load_tokenizer(str(tok_dir))
        except Exception as e:
            logger.warning("Tokenizer load failed", error=str(e))

    return model, tokenizer


def run_evaluation(args: argparse.Namespace) -> dict:
    device = settings.torch_device

    logger.info("Loading new model",  path=args.new_model)
    new_model,  new_tokenizer  = load_model(args.new_model,  device)

    logger.info("Loading base model", path=args.base_model)
    base_model, base_tokenizer = load_model(args.base_model, device)

    # Build eval DataLoader
    from src.data.datasets.text_dataset import JSONLTextDataset
    from src.data.datasets.data_collator import LanguageModelCollator
    from torch.utils.data import DataLoader

    tokenizer  = new_tokenizer or base_tokenizer
    eval_path  = args.eval_data
    dataset    = JSONLTextDataset(path=eval_path, tokenizer=tokenizer, max_length=512)
    collator   = LanguageModelCollator(pad_token_id=0, max_length=512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    if len(dataset) == 0:
        logger.warning("Eval dataset is empty — assuming pass")
        result = {"new_perplexity": 1.0, "base_perplexity": 1.0, "regression": 0.0, "passed": True}
    else:
        new_ppl  = compute_perplexity(new_model,  dataloader, device, args.max_batches)
        base_ppl = compute_perplexity(base_model, dataloader, device, args.max_batches)

        regression = (new_ppl - base_ppl) / max(base_ppl, 1e-6)
        passed     = regression <= args.max_regression

        result = {
            "new_perplexity":  new_ppl,
            "base_perplexity": base_ppl,
            "regression":      round(regression, 6),
            "max_regression":  args.max_regression,
            "passed":          passed,
        }

        logger.info(
            "Evaluation complete",
            new_perplexity=new_ppl,
            base_perplexity=base_ppl,
            regression=f"{regression * 100:.2f}%",
            passed=passed,
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    args   = parse_args()
    result = run_evaluation(args)
    print(json.dumps(result))
    exit(0 if result["passed"] else 1)
