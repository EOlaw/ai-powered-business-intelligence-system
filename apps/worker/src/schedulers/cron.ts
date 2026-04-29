/**
 * InsightSerenity Worker — Cron Scheduler
 * =========================================
 * Schedules the data flywheel pipeline on a recurring basis.
 *
 * Default schedules (configurable via env vars):
 *   CRAWL_CRON         = "0 1 * * *"   → daily at 01:00 UTC
 *   TRAIN_CRON         = "0 3 * * 0"   → weekly on Sunday at 03:00 UTC
 *   QUOTA_RESET_CRON   = "0 0 * * *"   → daily at midnight UTC (quota reset)
 *
 * Each schedule triggers the first job in its chain; subsequent jobs are
 * enqueued automatically by the previous job on success.
 */

import cron from 'node-cron';
import { crawlQueue, trainQueue, type CrawlJobData, type TrainJobData } from '../queues/index.js';
import { getLogger }  from '../logger.js';

const log = getLogger('scheduler');

function runId(): string {
  return `run-${new Date().toISOString().slice(0, 10)}-${Math.random().toString(36).slice(2, 8)}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Daily crawl → preprocess → (if weekly) train → evaluate → swap
// ─────────────────────────────────────────────────────────────────────────────

const CRAWL_CRON   = process.env['CRAWL_CRON']   ?? '0 1 * * *';
const TRAIN_CRON   = process.env['TRAIN_CRON']   ?? '0 3 * * 0';

const SEED_URLS    = (process.env['CRAWLER_SEED_URLS'] ?? '')
  .split(',').map(s => s.trim()).filter(Boolean);

const MAX_PAGES    = parseInt(process.env['CRAWLER_MAX_PAGES'] ?? '5000', 10);
const BASE_MODEL   = process.env['BASE_MODEL_NAME'] ?? 'insightserenity-1';

export function startScheduler(): void {
  // ── Daily crawl + preprocess ──────────────────────────────────────────────
  if (SEED_URLS.length > 0) {
    cron.schedule(CRAWL_CRON, async () => {
      const id = runId();
      log.info({ runId: id, seedUrls: SEED_URLS.length }, 'Scheduled crawl triggered');

      const jobData: CrawlJobData = {
        runId:     id,
        seedUrls:  SEED_URLS,
        maxPages:  MAX_PAGES,
        outputKey: `data/runs/${id}/raw`,
      };

      await crawlQueue.add('crawl', jobData, { jobId: `crawl-${id}` });
    });

    log.info({ cron: CRAWL_CRON }, 'Daily crawl scheduler registered');
  } else {
    log.warn('No CRAWLER_SEED_URLS configured — crawl scheduler disabled');
  }

  // ── Weekly training run (triggered independently, not from crawl chain) ──
  cron.schedule(TRAIN_CRON, async () => {
    const id = runId();
    const version = `auto-${new Date().toISOString().slice(0, 10).replace(/-/g, '.')}`;

    log.info({ runId: id, version }, 'Scheduled training run triggered');

    const jobData: TrainJobData = {
      runId:      id,
      datasetKey: `data/processed/latest`,
      baseModel:  `${BASE_MODEL}:latest`,
      newVersion: `${BASE_MODEL}:${version}`,
      mode:       'finetune',
      maxSteps:   parseInt(process.env['FLYWHEEL_TRAIN_STEPS'] ?? '2000', 10),
    };

    await trainQueue.add('train', jobData, { jobId: `train-${id}` });
  });

  log.info({ cron: TRAIN_CRON }, 'Weekly training scheduler registered');
}
