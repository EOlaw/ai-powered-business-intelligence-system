# Tiny Model Runbook

This runbook trains and promotes a `gpt-tiny` model. Use this path first when you want to prove the whole AI engine workflow works on CPU.

Commands assume Git Bash on Windows from the repo root:

```bash
cd ~/onedrive/desktop/ai-powered-business-intelligence-system/apps/ai-engine
```

## 1. Build Or Reuse The Dataset

If `storage/datasets/05_corpus.jsonl` already exists and `pipeline_stats.json` shows a non-zero `final`, you can reuse it.

To rebuild from your seed URLs:

```bash
CRAWLER_MAX_PAGES=500 CRAWLER_MAX_DEPTH=1 python -m src.data.pipeline --seeds scripts/training/seed_urls.txt --output storage/datasets/ --force
```

## 2. Build A Better Test Tokenizer

Use a 2k vocabulary instead of the older 256-token test vocabulary. The 256-token tokenizer works technically, but it tends to produce repeated character junk.

```bash
python -c "from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer; BPETokenizer.from_corpus('storage/datasets/05_corpus.jsonl', vocab_size=2000, save_dir='storage/tokenizers/insightserenity-bpe-2k', min_frequency=1)"
```

Quick check:

```bash
python -c "from src.tokenizer import load_tokenizer; t=load_tokenizer('storage/tokenizers/insightserenity-bpe-2k'); print(t.vocab_size); print(t.encode('hello business intelligence')[:20])"
```

## 3. Train Tiny

This is the safest local CPU test.

```bash
python scripts/training/train.py \
  --model gpt-tiny \
  --seq-len 128 \
  --tokenizer storage/tokenizers/insightserenity-bpe-2k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-tiny-bpe2k \
  --epochs 1 \
  --batch-size 1 \
  --lr 1e-4 \
  --min-corpus-docs 10 \
  --no-amp \
  --num-workers 0 \
  --save-every 100 \
  --log-every 10
```

Expected checkpoint:

```text
storage/checkpoints/insightserenity-tiny-bpe2k/step_00000100
```

## 4. Promote Tiny

Use the latest checkpoint from the run. Adjust the step if your run stopped earlier or later.

```bash
python scripts/models/promote.py \
  --checkpoint storage/checkpoints/insightserenity-tiny-bpe2k/step_00000100 \
  --tokenizer storage/tokenizers/insightserenity-bpe-2k \
  --name insightserenity-1 \
  --version v0.1.0-tiny
```

## 5. Serve And Test

Restart the AI engine:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

In another terminal:

```bash
curl http://localhost:8001/v1/models
```

Ask a simple completion:

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "insightserenity-1:latest",
    "prompt": "Artificial intelligence is",
    "max_tokens": 40,
    "temperature": 0.2
  }'
```

## Notes

`gpt-tiny` is for plumbing tests, not quality. It should load, train, promote, and generate, but it will not answer like a polished assistant.
