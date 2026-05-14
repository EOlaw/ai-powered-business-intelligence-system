# Small Model Runbook

This runbook trains and promotes `gpt-small`. In this codebase, `gpt-small` is about 85M parameters. It can run on CPU, but it is slow and overfits quickly on small datasets.

Commands assume Git Bash on Windows from the repo root:

```bash
cd ~/onedrive/desktop/ai-powered-business-intelligence-system/apps/ai-engine
```

## 1. Build Or Reuse The Dataset

Check the final document count:

```bash
python -c "import json; print(json.load(open('storage/datasets/pipeline_stats.json')).get('final'))"
```

If you need to rebuild:

```bash
CRAWLER_MAX_PAGES=1000 CRAWLER_MAX_DEPTH=1 python -m src.data.pipeline --seeds scripts/training/seed_urls.txt --output storage/datasets/ --force
```

## 2. Build The Tokenizer

Use at least 2k vocabulary for small-model tests.

```bash
python -c "from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer; BPETokenizer.from_corpus('storage/datasets/05_corpus.jsonl', vocab_size=2000, save_dir='storage/tokenizers/insightserenity-bpe-2k', min_frequency=1)"
```

Quick check:

```bash
python -c "from src.tokenizer import load_tokenizer; t=load_tokenizer('storage/tokenizers/insightserenity-bpe-2k'); print(t.vocab_size); print(t.encode('Artificial intelligence is')[:20])"
```

## 3. Small Smoke Train

Use this to verify `gpt-small` can train and checkpoint.

```bash
python scripts/training/train.py \
  --model gpt-small \
  --seq-len 128 \
  --tokenizer storage/tokenizers/insightserenity-bpe-2k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-small-smoke \
  --epochs 1 \
  --max-steps 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --min-corpus-docs 10 \
  --no-amp \
  --num-workers 0 \
  --save-every 25 \
  --log-every 5
```

## 4. Small Local Train

Use this after the smoke train works. Keep an eye on loss. If loss falls near `0.0001`, the model is memorizing.

```bash
python scripts/training/train.py \
  --model gpt-small \
  --seq-len 128 \
  --tokenizer storage/tokenizers/insightserenity-bpe-2k \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-small-bpe2k \
  --epochs 2 \
  --batch-size 1 \
  --lr 1e-4 \
  --min-corpus-docs 10 \
  --no-amp \
  --num-workers 0 \
  --save-every 100 \
  --log-every 10
```

## 5. Promote Small

Use the latest checkpoint that was saved.

```bash
python scripts/models/promote.py \
  --checkpoint storage/checkpoints/insightserenity-small-bpe2k/step_00000100 \
  --tokenizer storage/tokenizers/insightserenity-bpe-2k \
  --name insightserenity-1 \
  --version v0.2.0-small
```

## 6. Serve And Test

Restart the AI engine:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

Ask with low temperature:

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "insightserenity-1:latest",
    "prompt": "Artificial intelligence is",
    "max_tokens": 60,
    "temperature": 0.2
  }'
```

## Notes

For better answers, do not simply add more epochs to a tiny dataset. Add more corpus data, use a larger tokenizer, and train with validation data.
