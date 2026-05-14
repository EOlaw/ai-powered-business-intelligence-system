# Large Model Runbook

This runbook is for `gpt-large`. In this codebase, `gpt-large` uses `d_model=1280`, `36` layers, and `20` attention heads. Treat this as a serious GPU training run, not a CPU experiment.

Commands assume Git Bash on Windows from the repo root:

```bash
cd ~/onedrive/desktop/ai-powered-business-intelligence-system/apps/ai-engine
```

## 1. Dataset Requirement

Do not train `gpt-large` on a few hundred documents. Build a much larger corpus first.

Recommended minimum for a real experiment:

```text
10000+ final documents
```

Example crawl cap:

```bash
CRAWLER_MAX_PAGES=15000 CRAWLER_MAX_DEPTH=1 python -m src.data.pipeline --seeds scripts/training/seed_urls.txt --output storage/datasets/ --force
```

Check the final count:

```bash
python -c "import json; print(json.load(open('storage/datasets/pipeline_stats.json')).get('final'))"
```

## 2. Build A Larger Tokenizer

Use a larger vocabulary for the large model.

```bash
python -c "from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer; BPETokenizer.from_corpus('storage/datasets/05_corpus.jsonl', vocab_size=8000, save_dir='storage/tokenizers/insightserenity-bpe-8k', min_frequency=2)"
```

Quick check:

```bash
python -c "from src.tokenizer import load_tokenizer; t=load_tokenizer('storage/tokenizers/insightserenity-bpe-8k'); print(t.vocab_size); print(t.encode('Artificial intelligence is')[:20])"
```

## 3. Large Smoke Train

Run this before any long job. If this fails, do not start the full run.

```bash
python scripts/training/train.py \
  --model gpt-large \
  --seq-len 128 \
  --tokenizer storage/tokenizers/insightserenity-bpe-8k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-large-smoke \
  --epochs 1 \
  --max-steps 5 \
  --batch-size 1 \
  --lr 3e-5 \
  --min-corpus-docs 10000 \
  --num-workers 0 \
  --save-every 5 \
  --log-every 1
```

If you are on CPU, this will likely be painfully slow. Add `--no-amp` if AMP causes issues.

## 4. Large Training Run

Use a GPU for this. Increase `seq-len` only when memory allows.

```bash
python scripts/training/train.py \
  --model gpt-large \
  --seq-len 256 \
  --tokenizer storage/tokenizers/insightserenity-bpe-8k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-large-bpe8k \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --lr 3e-5 \
  --min-corpus-docs 10000 \
  --num-workers 0 \
  --save-every 500 \
  --log-every 50
```

## 5. Multi-GPU Option

If you have multiple GPUs and PyTorch distributed is configured:

```bash
torchrun --nproc_per_node=2 scripts/training/train.py \
  --model gpt-large \
  --seq-len 256 \
  --tokenizer storage/tokenizers/insightserenity-bpe-8k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-large-bpe8k-ddp \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --lr 3e-5 \
  --min-corpus-docs 10000 \
  --num-workers 0 \
  --save-every 500 \
  --log-every 50 \
  --distributed
```

## 6. Promote Large

Use the latest checkpoint path from your run.

```bash
python scripts/models/promote.py \
  --checkpoint storage/checkpoints/insightserenity-large-bpe8k/step_00000500 \
  --tokenizer storage/tokenizers/insightserenity-bpe-8k \
  --name insightserenity-1 \
  --version v0.4.0-large
```

## 7. Serve And Test

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "insightserenity-1:latest",
    "prompt": "Artificial intelligence is",
    "max_tokens": 120,
    "temperature": 0.2
  }'
```

## Notes

`gpt-large` will not become useful just because it is large. It needs a large, clean corpus, a suitable tokenizer, validation, and enough compute. For local iteration, use tiny and small first.
