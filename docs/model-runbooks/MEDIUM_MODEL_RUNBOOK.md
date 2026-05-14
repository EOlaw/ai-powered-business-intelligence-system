# Medium Model Runbook

This runbook is for `gpt-medium`. In this codebase, `gpt-medium` uses `d_model=1024`, `24` layers, and `16` attention heads. Treat it as a GPU-oriented run.

Commands assume Git Bash on Windows from the repo root:

```bash
cd ~/onedrive/desktop/ai-powered-business-intelligence-system/apps/ai-engine
```

## 1. Dataset Requirement

Do not use `gpt-medium` on a tiny corpus. As a minimum local target, build at least `1000` final documents. More is better.

```bash
CRAWLER_MAX_PAGES=3000 CRAWLER_MAX_DEPTH=1 python -m src.data.pipeline --seeds scripts/training/seed_urls.txt --output storage/datasets/ --force
```

Check the final count:

```bash
python -c "import json; print(json.load(open('storage/datasets/pipeline_stats.json')).get('final'))"
```

## 2. Build A Medium Tokenizer

Use a larger vocabulary for this model size.

```bash
python -c "from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer; BPETokenizer.from_corpus('storage/datasets/05_corpus.jsonl', vocab_size=4000, save_dir='storage/tokenizers/insightserenity-bpe-4k', min_frequency=1)"
```

Quick check:

```bash
python -c "from src.tokenizer import load_tokenizer; t=load_tokenizer('storage/tokenizers/insightserenity-bpe-4k'); print(t.vocab_size); print(t.encode('Artificial intelligence is')[:20])"
```

## 3. Medium Smoke Train

Run this first. On CPU, expect this to be very slow.

```bash
python scripts/training/train.py \
  --model gpt-medium \
  --seq-len 128 \
  --tokenizer storage/tokenizers/insightserenity-bpe-4k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-medium-smoke \
  --epochs 1 \
  --max-steps 10 \
  --batch-size 1 \
  --lr 5e-5 \
  --min-corpus-docs 1000 \
  --no-amp \
  --num-workers 0 \
  --save-every 10 \
  --log-every 1
```

## 4. Medium Training Run

Use this only if the smoke run is stable and you have enough compute.

```bash
python scripts/training/train.py \
  --model gpt-medium \
  --seq-len 256 \
  --tokenizer storage/tokenizers/insightserenity-bpe-4k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-medium-bpe4k \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --lr 5e-5 \
  --min-corpus-docs 1000 \
  --num-workers 0 \
  --save-every 250 \
  --log-every 25
```

If you are on CPU, add:

```bash
--no-amp
```

## 5. Promote Medium

Use the latest checkpoint path from your run.

```bash
python scripts/models/promote.py \
  --checkpoint storage/checkpoints/insightserenity-medium-bpe4k/step_00000250 \
  --tokenizer storage/tokenizers/insightserenity-bpe-4k \
  --name insightserenity-1 \
  --version v0.3.0-medium
```

## 6. Serve And Test

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "insightserenity-1:latest",
    "prompt": "Artificial intelligence is",
    "max_tokens": 80,
    "temperature": 0.2
  }'
```

## Notes

`gpt-medium` is not a quick local test. Use `gpt-tiny` and `gpt-small` to validate workflow first, then move to medium when you have enough data and compute.
