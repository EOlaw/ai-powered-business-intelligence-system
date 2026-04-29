/**
 * InsightSerenity Worker — BullMQ Queue Definitions
 * ==================================================
 * Five queues for the data flywheel pipeline:
 *
 *   crawl      → fetch new web data and store raw HTML in S3
 *   preprocess → clean, deduplicate, quality-filter the raw crawl output
 *   train      → run incremental pretraining or fine-tuning on new data
 *   evaluate   → compare new model vs. current; reject regressions
 *   swap       → hot-swap the AI engine to the new model weights
 *
 * Pipeline flow (each job enqueues the next on completion):
 *   CronScheduler → crawl → preprocess → train → evaluate → swap
 *
 * Each stage is idempotent: re-running a failed stage produces the same output.
 * Jobs can be triggered individually or as a full pipeline run.
 */

import { Queue } from 'bullmq';
import IORedis   from 'ioredis';

const REDIS_URL = process.env['REDIS_URL'] ?? 'redis://localhost:6379';

export const connection = new IORedis(REDIS_URL, {
  maxRetriesPerRequest: null,
  enableReadyCheck:     true,
});

const defaultJobOptions = {
  attempts:  3,
  backoff:   { type: 'exponential', delay: 5_000 },
  removeOnComplete: { count: 20 },
  removeOnFail:     { count: 100 },
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Queue declarations
// ─────────────────────────────────────────────────────────────────────────────

/** Crawl job payload */
export interface CrawlJobData {
  runId:     string;
  seedUrls:  string[];
  maxPages:  number;
  outputKey: string;   // S3 key / local path prefix for output
}

/** Preprocess job payload */
export interface PreprocessJobData {
  runId:     string;
  inputKey:  string;   // Where raw crawl data lives
  outputKey: string;   // Where cleaned JSONL goes
}

/** Train job payload */
export interface TrainJobData {
  runId:      string;
  datasetKey: string;   // Path to preprocessed training data
  baseModel:  string;   // e.g. "insightserenity-1:latest"
  newVersion: string;   // e.g. "insightserenity-1:v1.3.0"
  mode:       'pretrain' | 'finetune';
  maxSteps:   number;
}

/** Evaluate job payload */
export interface EvaluateJobData {
  runId:          string;
  newModelPath:   string;
  baseModelPath:  string;
  maxRegression:  number;   // Max acceptable perplexity regression fraction (0.02 = 2%)
}

/** Swap job payload */
export interface SwapJobData {
  runId:    string;
  modelKey: string;   // S3 key or local path to new weights
  version:  string;
}

export const crawlQueue      = new Queue<CrawlJobData>('crawl',      { connection, defaultJobOptions });
export const preprocessQueue = new Queue<PreprocessJobData>('preprocess', { connection, defaultJobOptions });
export const trainQueue      = new Queue<TrainJobData>('train',      { connection, defaultJobOptions: { ...defaultJobOptions, attempts: 1 } });
export const evaluateQueue   = new Queue<EvaluateJobData>('evaluate', { connection, defaultJobOptions });
export const swapQueue       = new Queue<SwapJobData>('swap',        { connection, defaultJobOptions });
