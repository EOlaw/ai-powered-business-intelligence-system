/**
 * InsightSerenity Worker — Crawl Job
 * ====================================
 * Invokes the Python AI-engine crawler via HTTP or subprocess to fetch
 * new web data, then enqueues a preprocess job on completion.
 *
 * Strategy:
 *   The crawler is implemented in Python (apps/ai-engine/src/data/crawler/).
 *   This worker triggers it via a REST call to the AI engine's admin API,
 *   which runs the crawler async and writes raw JSONL to the data store.
 *
 *   On completion the crawler endpoint returns the output path, which this
 *   worker passes to the preprocess queue.
 */

import { Worker, Job }    from 'bullmq';
import { connection, preprocessQueue, type CrawlJobData, type PreprocessJobData } from '../queues/index.js';
import { getLogger }      from '../logger.js';

const log = getLogger('crawl-job');

const AI_ENGINE_URL         = process.env['AI_ENGINE_URL']         ?? 'http://localhost:8001';
const INTERNAL_API_SECRET   = process.env['SERVING_INTERNAL_API_SECRET'] ?? '';

async function processCrawl(job: Job<CrawlJobData>): Promise<void> {
  const { runId, seedUrls, maxPages, outputKey } = job.data;
  log.info({ runId, seedUrls: seedUrls.length, maxPages }, 'Crawl job started');

  await job.updateProgress(5);

  // Trigger the Python crawler via the AI engine admin endpoint
  const response = await fetch(`${AI_ENGINE_URL}/admin/crawl`, {
    method:  'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${INTERNAL_API_SECRET}`,
    },
    body: JSON.stringify({ seed_urls: seedUrls, max_pages: maxPages, output_key: outputKey }),
    signal: AbortSignal.timeout(3_600_000),   // 1-hour max crawl
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Crawl trigger failed: HTTP ${response.status} — ${err}`);
  }

  const result = await response.json() as { output_path: string; pages_crawled: number };
  log.info({ runId, pagesCrawled: result.pages_crawled, outputPath: result.output_path }, 'Crawl complete');

  await job.updateProgress(100);

  // Chain to preprocess
  const preprocessPayload: PreprocessJobData = {
    runId,
    inputKey:  result.output_path,
    outputKey: outputKey.replace('/raw/', '/processed/'),
  };

  await preprocessQueue.add('preprocess', preprocessPayload, {
    jobId: `preprocess-${runId}`,
  });

  log.info({ runId }, 'Preprocess job enqueued');
}

export function startCrawlWorker(): Worker<CrawlJobData> {
  const worker = new Worker<CrawlJobData>('crawl', processCrawl, {
    connection,
    concurrency: 2,    // Max 2 concurrent crawl jobs (bandwidth limit)
  });

  worker.on('failed', (job, err) => {
    log.error({ jobId: job?.id, err }, 'Crawl job failed');
  });

  return worker;
}
