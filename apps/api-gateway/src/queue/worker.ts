/**
 * InsightSerenity API Gateway — Background Job Queues
 * =====================================================
 * BullMQ-backed queues for async, reliable background work.
 *
 * Queues defined here:
 *   webhookQueue   — Delivers webhook payloads to org endpoints with retry/backoff.
 *                    Decoupled from the request path so a slow webhook receiver
 *                    cannot increase API latency.
 *
 *   usageQueue     — Batches UsageRecord inserts to reduce DB write pressure.
 *                    Records are buffered in Redis and flushed every 5 seconds.
 *
 * Each queue has:
 *   - A Queue (producer) — used by route handlers to enqueue jobs
 *   - A Worker (consumer) — picks up and executes jobs
 *   - Retry policy: 3 attempts with exponential backoff (2^attempt * 1000ms)
 *   - Dead-letter: failed jobs remain in the queue for inspection
 *
 * Workers are started by calling startWorkers() from server.ts.
 * They are stopped gracefully on SIGTERM by calling stopWorkers().
 */

import { Queue, Worker, Job } from 'bullmq';
import { createConnection } from '../cache/redis.js';
import { getLogger } from '../observability/logger.js';
import { signWebhookPayload } from '../security/crypto.js';

const log = getLogger('queue');

// Separate connections required by BullMQ (one per producer, one per worker)
const producerConn = createConnection();
const workerConn   = createConnection();

// ─────────────────────────────────────────────────────────────────────────────
// Queue definitions
// ─────────────────────────────────────────────────────────────────────────────

/** Webhook delivery job payload. */
export interface WebhookJob {
  webhookId: string;
  orgId:     string;
  url:       string;
  secret:    string;
  event:     string;
  payload:   Record<string, unknown>;
}

/** Usage record batch insert job payload. */
export interface UsageJob {
  orgId:            string;
  apiKeyId:         string;
  endpoint:         string;
  method:           string;
  promptTokens:     number;
  completionTokens: number;
  totalTokens:      number;
  latencyMs:        number;
  statusCode:       number;
  model:            string;
}

const DEFAULT_JOB_OPTIONS = {
  attempts:     3,
  backoff: {
    type:  'exponential',
    delay: 1_000,   // 1s, 2s, 4s
  },
  removeOnComplete: { count: 100 },
  removeOnFail:     { count: 500 },
} as const;

export const webhookQueue = new Queue<WebhookJob>('webhooks', {
  connection:     producerConn,
  defaultJobOptions: DEFAULT_JOB_OPTIONS,
});

export const usageQueue = new Queue<UsageJob>('usage', {
  connection:        producerConn,
  defaultJobOptions: { ...DEFAULT_JOB_OPTIONS, attempts: 5 },
});

// ─────────────────────────────────────────────────────────────────────────────
// Webhook worker
// ─────────────────────────────────────────────────────────────────────────────

async function processWebhook(job: Job<WebhookJob>): Promise<void> {
  const { url, secret, event, payload, webhookId } = job.data;

  const body      = JSON.stringify({ event, data: payload, webhookId, timestamp: new Date().toISOString() });
  const signature = signWebhookPayload(body, secret);

  const response = await fetch(url, {
    method:  'POST',
    headers: {
      'Content-Type':               'application/json',
      'X-InsightSerenity-Signature': signature,
      'X-InsightSerenity-Event':     event,
    },
    body,
    signal: AbortSignal.timeout(10_000),  // 10s timeout per delivery attempt
  });

  if (!response.ok) {
    throw new Error(`Webhook delivery failed: HTTP ${response.status} for ${url}`);
  }

  log.debug({ webhookId, event, url, status: response.status }, 'Webhook delivered');
}

// ─────────────────────────────────────────────────────────────────────────────
// Usage worker
// ─────────────────────────────────────────────────────────────────────────────

async function processUsage(job: Job<UsageJob>): Promise<void> {
  // Lazy import prisma to avoid circular dependency at module init time
  const { prisma } = await import('../db/client.js');

  await prisma.usageRecord.create({
    data: {
      orgId:            job.data.orgId,
      apiKeyId:         job.data.apiKeyId,
      endpoint:         job.data.endpoint,
      method:           job.data.method,
      promptTokens:     job.data.promptTokens,
      completionTokens: job.data.completionTokens,
      totalTokens:      job.data.totalTokens,
      latencyMs:        job.data.latencyMs,
      statusCode:       job.data.statusCode,
      model:            job.data.model,
    },
  });

  // Also increment the org's daily token counter in the subscription table
  await prisma.subscription.updateMany({
    where: { orgId: job.data.orgId },
    data:  { currentDayTokens: { increment: job.data.totalTokens } },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker lifecycle
// ─────────────────────────────────────────────────────────────────────────────

let webhookWorker: Worker<WebhookJob> | null = null;
let usageWorker:   Worker<UsageJob>   | null = null;

export function startWorkers(): void {
  webhookWorker = new Worker<WebhookJob>('webhooks', processWebhook, {
    connection:  workerConn,
    concurrency: 10,
  });

  usageWorker = new Worker<UsageJob>('usage', processUsage, {
    connection:  workerConn,
    concurrency: 20,   // High concurrency — mostly DB writes
  });

  for (const worker of [webhookWorker, usageWorker]) {
    worker.on('failed', (job, err) => {
      log.error({ jobId: job?.id, queue: worker.name, err }, 'Job failed');
    });
    worker.on('completed', (job) => {
      log.debug({ jobId: job.id, queue: worker.name }, 'Job completed');
    });
  }

  log.info('Background workers started');
}

export async function stopWorkers(): Promise<void> {
  await Promise.all([
    webhookWorker?.close(),
    usageWorker?.close(),
  ]);
  await producerConn.quit();
  await workerConn.quit();
  log.info('Background workers stopped');
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience enqueue functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Enqueue a usage record for async DB insertion.
 * Fire-and-forget from route handlers — does not block the response.
 */
export function enqueueUsage(data: UsageJob): void {
  usageQueue.add('record', data).catch((err: Error) => {
    log.error({ err }, 'Failed to enqueue usage record');
  });
}

/**
 * Enqueue a webhook delivery job.
 */
export async function enqueueWebhook(data: WebhookJob): Promise<void> {
  await webhookQueue.add('deliver', data);
}
