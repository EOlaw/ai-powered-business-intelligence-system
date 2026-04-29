/**
 * InsightSerenity Worker — Evaluate Job
 * =======================================
 * Compares the new model against the current production model.
 * Rejects (dead-letters the swap) if perplexity regresses beyond the threshold.
 *
 * Evaluation metrics checked:
 *   - Perplexity on held-out eval set (primary gate)
 *   - BLEU-4 on a reference generation task (secondary check)
 *
 * If the new model passes, enqueues a swap job.
 * If it regresses, logs a warning and stops the pipeline — the current model
 * keeps running without interruption.
 */

import { Worker, Job }  from 'bullmq';
import { connection, swapQueue, type EvaluateJobData, type SwapJobData } from '../queues/index.js';
import { getLogger }    from '../logger.js';

const log             = getLogger('evaluate-job');
const AI_ENGINE_URL   = process.env['AI_ENGINE_URL']         ?? 'http://localhost:8001';
const INTERNAL_SECRET = process.env['SERVING_INTERNAL_API_SECRET'] ?? '';

async function processEvaluate(job: Job<EvaluateJobData>): Promise<void> {
  const { runId, newModelPath, baseModelPath, maxRegression } = job.data;
  log.info({ runId, newModelPath }, 'Evaluate job started');

  await job.updateProgress(10);

  const response = await fetch(`${AI_ENGINE_URL}/admin/evaluate`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${INTERNAL_SECRET}` },
    body: JSON.stringify({ new_model_path: newModelPath, base_model_path: baseModelPath }),
    signal: AbortSignal.timeout(1_800_000),  // 30 min
  });

  if (!response.ok) {
    throw new Error(`Evaluation call failed: HTTP ${response.status} — ${await response.text()}`);
  }

  const result = await response.json() as {
    new_perplexity:  number;
    base_perplexity: number;
    regression:      number;   // Fraction: (new - base) / base
    passed:          boolean;
  };

  log.info({
    runId,
    newPerplexity:  result.new_perplexity,
    basePerplexity: result.base_perplexity,
    regression:     `${(result.regression * 100).toFixed(2)}%`,
    passed:         result.passed,
  }, 'Evaluation complete');

  await job.updateProgress(100);

  if (!result.passed || result.regression > maxRegression) {
    log.warn({
      runId,
      regression: result.regression,
      maxRegression,
    }, 'Model evaluation FAILED — swap aborted, current model retained');
    return;   // Stop the pipeline; do not enqueue swap
  }

  const swapPayload: SwapJobData = {
    runId,
    modelKey: newModelPath,
    version:  newModelPath.split('/').slice(-2).join(':'),  // "insightserenity-1:v1.3.0"
  };

  await swapQueue.add('swap', swapPayload, { jobId: `swap-${runId}` });
  log.info({ runId }, 'Swap job enqueued — model passed evaluation');
}

export function startEvaluateWorker(): Worker<EvaluateJobData> {
  const worker = new Worker<EvaluateJobData>('evaluate', processEvaluate, { connection, concurrency: 1 });
  worker.on('failed', (job, err) => log.error({ jobId: job?.id, err }, 'Evaluate job failed'));
  return worker;
}
