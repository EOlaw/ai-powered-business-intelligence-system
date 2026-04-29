/**
 * InsightSerenity Worker — Entry Point
 * ======================================
 * Starts all BullMQ workers for the data flywheel pipeline and registers
 * cron schedules for recurring jobs.
 *
 * Pipeline: crawl → preprocess → train → evaluate → swap
 *
 * Each stage is a separate Worker. They communicate by enqueuing the next
 * job on successful completion (job chaining), not by direct function calls.
 * This means each stage is independently retriable and observable.
 */

import { getLogger }          from './logger.js';
import { startCrawlWorker }   from './jobs/crawl.job.js';
import { startPreprocessWorker } from './jobs/preprocess.job.js';
import { startTrainWorker }   from './jobs/train.job.js';
import { startEvaluateWorker } from './jobs/evaluate.job.js';
import { startSwapWorker }    from './jobs/swap-model.job.js';
import { startScheduler }     from './schedulers/cron.js';
import { connection }         from './queues/index.js';

const log = getLogger('main');

async function main(): Promise<void> {
  log.info('InsightSerenity Worker starting');

  // ── Start all pipeline workers ────────────────────────────────────────────
  const workers = [
    startCrawlWorker(),
    startPreprocessWorker(),
    startTrainWorker(),
    startEvaluateWorker(),
    startSwapWorker(),
  ];

  log.info({ workerCount: workers.length }, 'All pipeline workers started');

  // ── Start cron scheduler ──────────────────────────────────────────────────
  startScheduler();
  log.info('Cron scheduler started');

  // ── Graceful shutdown ─────────────────────────────────────────────────────
  async function shutdown(signal: string): Promise<void> {
    log.info({ signal }, 'Shutdown signal received');
    await Promise.all(workers.map(w => w.close()));
    await connection.quit();
    log.info('Worker gracefully stopped');
    process.exit(0);
  }

  process.once('SIGTERM', () => shutdown('SIGTERM'));
  process.once('SIGINT',  () => shutdown('SIGINT'));

  process.on('unhandledRejection', (reason) => {
    log.error({ reason }, 'Unhandled rejection in worker');
  });

  log.info('Worker is running — waiting for jobs');
}

main().catch(err => {
  console.error('Worker startup failed:', err);
  process.exit(1);
});
