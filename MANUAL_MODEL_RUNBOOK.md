# Manual AI Engine Runbook

This file shows the manual steps to rebuild the dataset, tokenizer, and test model from a fresh terminal.

The commands below assume you are using Git Bash on Windows from this repo:

```bash
cd ~/onedrive/desktop/ai-powered-business-intelligence-system
```

## 1. Install Python Dependencies

The repo root has a `requirements.txt` that points to the AI engine requirements.

```bash
python -m pip install -r requirements.txt
```

If pip says packages are already installed, that is fine.

## 2. Move Into The AI Engine

Most Python commands must run from `apps/ai-engine`.

```bash
cd apps/ai-engine
```

## 3. Check Seed URLs

The crawler reads seed URLs from:

```text
scripts/training/seed_urls.txt
```

Each URL should be on its own line, for example:

```text
https://insightserenity.com/
https://insightserenity.com/about.html
https://insightserenity.com/services.html
```

More seed URLs usually means more final training documents.

## 4. Build The Dataset

Run the full crawl, extract, clean, dedup, and quality-filter pipeline:

```bash
python -m src.data.pipeline --seeds scripts/training/seed_urls.txt --output storage/datasets/
```

Expected output files:

```text
storage/datasets/01_raw_html.jsonl
storage/datasets/02_extracted.jsonl
storage/datasets/03_exact_deduped.jsonl
storage/datasets/04_near_deduped.jsonl
storage/datasets/05_corpus.jsonl
storage/datasets/pipeline_stats.json
```

The final training corpus is:

```text
storage/datasets/05_corpus.jsonl
```

If the pipeline says `final: 0`, the crawler did not collect usable text. Check the seed URLs, robots/access, and pipeline logs.

## 5. Build Or Rebuild The Tokenizer

If this folder is missing:

```text
storage/tokenizers/insightserenity-bpe-256
```

rebuild it from the corpus:

```bash
python -c "from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer; BPETokenizer.from_corpus('storage/datasets/05_corpus.jsonl', vocab_size=256, save_dir='storage/tokenizers/insightserenity-bpe-256', min_frequency=1)"
```

Expected tokenizer files:

```text
storage/tokenizers/insightserenity-bpe-256/vocab.json
storage/tokenizers/insightserenity-bpe-256/merges.txt
storage/tokenizers/insightserenity-bpe-256/tokenizer_config.json
```

Quick loader test:

```bash
python -c "from src.tokenizer import load_tokenizer; t=load_tokenizer('storage/tokenizers/insightserenity-bpe-256'); print(t.vocab_size); print(t.encode('hello business intelligence')[:10])"
```

## 6. Run A Fast CPU Training Test

Use `gpt-tiny` for local CPU testing. It is intentionally small and proves the full training path works.

```bash
python scripts/training/train.py \
  --model gpt-tiny \
  --seq-len 128 \
  --tokenizer storage/tokenizers/insightserenity-bpe-256 \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-test \
  --epochs 3 \
  --batch-size 1 \
  --lr 3e-4 \
  --min-corpus-docs 10 \
  --no-amp \
  --num-workers 0 \
  --save-every 10 \
  --log-every 1
```

Expected checkpoint folder:

```text
storage/checkpoints/insightserenity-test/
```

Example saved checkpoints:

```text
storage/checkpoints/insightserenity-test/step_00000010
storage/checkpoints/insightserenity-test/step_00000020
storage/checkpoints/insightserenity-test/step_00000030
```

## 7. Optional One-Step Smoke Test

If you only want to verify setup quickly:

```bash
python scripts/training/train.py \
  --model gpt-tiny \
  --seq-len 128 \
  --tokenizer storage/tokenizers/insightserenity-bpe-256 \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-smoke \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 1 \
  --lr 3e-4 \
  --min-corpus-docs 10 \
  --no-amp \
  --num-workers 0 \
  --save-every 1 \
  --log-every 1
```

Expected checkpoint:

```text
storage/checkpoints/insightserenity-smoke/step_00000001
```

## 8. Larger Training

Use `gpt-small` only when you have enough time or a GPU. In this repo, `gpt-small` is about 85M parameters, so it is very slow on CPU.

```bash
python scripts/training/train.py \
  --model gpt-small \
  --tokenizer storage/tokenizers/insightserenity-bpe-256 \
  --data storage/datasets/05_corpus.jsonl \
  --output storage/checkpoints \
  --run-name insightserenity-small \
  --epochs 3 \
  --batch-size 4 \
  --lr 3e-4 \
  --min-corpus-docs 1000
```

For serious training, add many more seed URLs first and rerun the data pipeline. The default training guard expects at least `1000` documents.

## 9. Promote A Checkpoint

After training, promote a checkpoint into model storage:

```bash
python scripts/models/promote.py \
  --checkpoint storage/checkpoints/insightserenity-test/step_00000030 \
  --name insightserenity-1 \
  --version v0.0.3
```

Promoted models live under:

```text
storage/models/
```

## 10. Start The AI Engine API

Run the FastAPI server:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

Health check:

```bash
curl http://localhost:8001/health
```

List models:

```bash
curl http://localhost:8001/v1/models
```

## Troubleshooting

`tokenizer_config.json not found`

Rebuild the tokenizer:

```bash
python -c "from src.tokenizer.bpe.bpe_tokenizer import BPETokenizer; BPETokenizer.from_corpus('storage/datasets/05_corpus.jsonl', vocab_size=256, save_dir='storage/tokenizers/insightserenity-bpe-256', min_frequency=1)"
```

`ModuleNotFoundError: No module named 'torch'`

Install requirements from the repo root:

```bash
cd ~/onedrive/desktop/ai-powered-business-intelligence-system
python -m pip install -r requirements.txt
```

Training starts but appears frozen on CPU

Use `gpt-tiny` instead of `gpt-small`, lower `--seq-len`, and keep `--batch-size 1`.

Pipeline creates `05_corpus.jsonl` but training rejects it as too small

For a tiny local test, lower the guard:

```bash
--min-corpus-docs 10
```

For real training, add more seed URLs and rebuild the dataset.
