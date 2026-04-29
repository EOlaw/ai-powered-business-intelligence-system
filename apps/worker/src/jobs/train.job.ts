/**
 * InsightSerenity Worker — Train Job
 * ====================================
 * Triggers incremental model training on the AI engine.
 * Training runs on the engine's GPU; this worker just dispatches and waits.
 *
 * The AI engine /admin/train endpoint:
 *   - Accepts the dataset path and training configuration
 *   - Runs the Python training loop (async, long-running)
 *   - Returns the path to new model weights when done
 *
 * After training succeeds, enqueues an evaluate job.
 * If training fails, the job retries once before going to the dead-letter queue.
 */

import { Worker, Job }   from 'bullmq';
import { connection, evaluateQueue, type TrainJobData, type EvaluateJobData } from '../queues/index.js';
import { getLogger }     from '../logger.js';

const log             = getLogger('train-job');
const AI_ENGINE_URL   = process.env['AI_ENGINE_URL']         ?? 'http://localhost:8001';
const INTERNAL_SECRET = process.env['SERVING_INTERNAL_API_SECRET'] ?? '';

async function processTrain(job: Job<TrainJobData>): Promise<void> {
  const { runId, datasetKey, baseModel, newVersion, mode, maxSteps } = job.data;
  log.info({ runId, mode, maxSteps, newVersion }, 'Train job started');

  await job.updateProgress(5);

  const response = await fetch(`${AI_ENGINE_URL}/admin/train`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${INTERNAL_SECRET}` },
    body: JSON.stringify({
      dataset_key:  datasetKey,
      base_model:   baseModel,
      new_version:  newVersion,
      mode,
      max_steps:    maxSteps,
    }),
    signal: AbortSignal.timeout(14_400_000),   // 4-hour max training run
  });

  if (!response.ok) {
    throw new Error(`Training failed: HTTP ${response.status} — ${await response.text()}`);
  }

  const result = await response.json() as {
    model_path:  string;
    final_loss:  number;
    perplexity:  number;
    steps:       number;
  };

  log.info({
    runId,
    newVersion,
    modelPath:  result.model_path,
    finalLoss:  result.final_loss,
    perplexity: result.perplexity,
    steps:      result.steps,
  }, 'Training complete');

  await job.updateProgress(100);

  const evalPayload: EvaluateJobData = {
    runId,
    newModelPath:  result.model_path,
    baseModelPath: `storage/models/${baseModel.replace(':', '/')}`,
    maxRegression: parseFloat(process.env['MAX_PERPLEXITY_REGRESSION'] ?? '0.02'),
  };

  await evaluateQueue.add('evaluate', evalPayload, { jobId: `evaluate-${runId}` });
  log.info({ runId }, 'Evaluate job enqueued');
}

export function startTrainWorker(): Worker<TrainJobData> {
  // Only 1 concurrent training job — GPU is shared
  const worker = new Worker<TrainJobData>('train', processTrain, { connection, concurrency: 1 });
  worker.on('failed', (job, err) => log.error({ jobId: job?.id, runId: job?.data?.runId, err }, 'Train job failed'));
  return worker;
}
