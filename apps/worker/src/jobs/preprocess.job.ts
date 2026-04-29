/**
 * InsightSerenity Worker — Preprocess Job
 * =========================================
 * Calls the AI engine's data pipeline endpoint to clean, deduplicate,
 * and quality-filter raw crawl output, then enqueues a training job.
 */

import { Worker, Job }   from 'bullmq';
import { connection, trainQueue, type PreprocessJobData, type TrainJobData } from '../queues/index.js';
import { getLogger }     from '../logger.js';

const log                = getLogger('preprocess-job');
const AI_ENGINE_URL      = process.env['AI_ENGINE_URL']         ?? 'http://localhost:8001';
const INTERNAL_SECRET    = process.env['SERVING_INTERNAL_API_SECRET'] ?? '';
const BASE_MODEL         = process.env['BASE_MODEL_NAME']       ?? 'insightserenity-1';

async function processPreprocess(job: Job<PreprocessJobData>): Promise<void> {
  const { runId, inputKey, outputKey } = job.data;
  log.info({ runId, inputKey }, 'Preprocess job started');

  await job.updateProgress(10);

  const response = await fetch(`${AI_ENGINE_URL}/admin/preprocess`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${INTERNAL_SECRET}` },
    body:    JSON.stringify({ input_key: inputKey, output_key: outputKey }),
    signal:  AbortSignal.timeout(1_800_000),   // 30 min
  });

  if (!response.ok) {
    throw new Error(`Preprocess failed: HTTP ${response.status} — ${await response.text()}`);
  }

  const result = await response.json() as { output_path: string; doc_count: number };
  log.info({ runId, docCount: result.doc_count, outputPath: result.output_path }, 'Preprocess complete');

  await job.updateProgress(100);

  const version = `v${new Date().toISOString().slice(0, 10).replace(/-/g, '.')}.${runId.slice(-6)}`;

  const trainPayload: TrainJobData = {
    runId,
    datasetKey:  result.output_path,
    baseModel:   `${BASE_MODEL}:latest`,
    newVersion:  `${BASE_MODEL}:${version}`,
    mode:        'finetune',
    maxSteps:    parseInt(process.env['FLYWHEEL_TRAIN_STEPS'] ?? '2000', 10),
  };

  await trainQueue.add('train', trainPayload, { jobId: `train-${runId}` });
  log.info({ runId, version }, 'Train job enqueued');
}

export function startPreprocessWorker(): Worker<PreprocessJobData> {
  const worker = new Worker<PreprocessJobData>('preprocess', processPreprocess, { connection, concurrency: 3 });
  worker.on('failed', (job, err) => log.error({ jobId: job?.id, err }, 'Preprocess job failed'));
  return worker;
}
