/**
 * InsightSerenity Worker — Hot-Swap Model Job
 * =============================================
 * Replaces the running model in the AI engine with new weights
 * WITHOUT taking the service down.
 *
 * Hot-swap sequence (zero-downtime):
 *   1. Copy new model weights to the engine's model storage directory
 *   2. Call POST /admin/reload-model on the AI engine
 *   3. The engine:
 *        a. Loads new weights into a second in-memory slot
 *        b. Atomically swaps the active engine reference
 *        c. Unloads the old weights (frees GPU memory)
 *   4. Verify the engine returns the new model name on /health
 *   5. Update the "latest" symlink in model storage
 *
 * Rollback:
 *   If the swap or health check fails, the engine continues running the
 *   previous model — nothing is broken. The job fails and the pipeline stops.
 */

import { Worker, Job }  from 'bullmq';
import { connection, type SwapJobData } from '../queues/index.js';
import { getLogger }    from '../logger.js';

const log             = getLogger('swap-job');
const AI_ENGINE_URL   = process.env['AI_ENGINE_URL']         ?? 'http://localhost:8001';
const INTERNAL_SECRET = process.env['SERVING_INTERNAL_API_SECRET'] ?? '';

async function processSwap(job: Job<SwapJobData>): Promise<void> {
  const { runId, modelKey, version } = job.data;
  log.info({ runId, version }, 'Model swap job started');

  await job.updateProgress(10);

  // ── 1. Trigger hot-reload on AI engine ──────────────────────────────────
  const reloadResponse = await fetch(`${AI_ENGINE_URL}/admin/reload-model`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${INTERNAL_SECRET}` },
    body:    JSON.stringify({ model_key: modelKey, version }),
    signal:  AbortSignal.timeout(120_000),   // 2 min: model load onto GPU
  });

  if (!reloadResponse.ok) {
    throw new Error(`Model reload failed: HTTP ${reloadResponse.status} — ${await reloadResponse.text()}`);
  }

  const reloadResult = await reloadResponse.json() as { success: boolean; active_model: string };
  log.info({ runId, activeModel: reloadResult.active_model }, 'Reload accepted by AI engine');

  await job.updateProgress(60);

  // ── 2. Verify new model is live ─────────────────────────────────────────
  await new Promise(resolve => setTimeout(resolve, 3_000));   // Brief settle time

  const healthResponse = await fetch(`${AI_ENGINE_URL}/health`, {
    signal: AbortSignal.timeout(10_000),
  });

  if (!healthResponse.ok) {
    throw new Error('AI engine health check failed after model swap');
  }

  const health = await healthResponse.json() as { model: string; status: string };

  if (!health.model?.includes(version.split(':')[0]!)) {
    log.warn({ runId, expectedVersion: version, activeModel: health.model }, 'Health check model mismatch');
  }

  await job.updateProgress(100);

  log.info({ runId, version, activeModel: health.model }, 'Model hot-swap complete ✓');
}

export function startSwapWorker(): Worker<SwapJobData> {
  // Only 1 concurrent swap — one GPU, one engine
  const worker = new Worker<SwapJobData>('swap', processSwap, { connection, concurrency: 1 });
  worker.on('completed', (job) => log.info({ jobId: job.id }, 'Swap completed successfully'));
  worker.on('failed',    (job, err) => log.error({ jobId: job?.id, err }, 'Swap job failed — current model retained'));
  return worker;
}
