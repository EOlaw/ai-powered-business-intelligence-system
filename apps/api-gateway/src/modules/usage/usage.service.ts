/**
 * InsightSerenity API Gateway — Usage Service
 * =============================================
 * Per-key and per-org usage analytics.
 *
 * The usage data is written asynchronously (via BullMQ) so that recording
 * usage never adds latency to the primary request path.
 *
 * Query patterns supported:
 *   - Overview: total tokens, requests, avg latency for a date range
 *   - Timeline: daily/hourly breakdown for charting
 *   - Per-key breakdown: which API keys consumed the most tokens
 *   - Per-endpoint breakdown: completions vs. chat vs. embeddings
 *   - Recent requests: paginated log of the last N calls
 */

import { prisma }    from '../../db/client.js';
import { getLogger } from '../../observability/logger.js';

const log = getLogger('usage-service');

// ─────────────────────────────────────────────────────────────────────────────
// Overview
// ─────────────────────────────────────────────────────────────────────────────

export async function getUsageOverview(
  orgId:   string,
  startDate: Date,
  endDate:   Date,
) {
  const [agg, byKey, byEndpoint] = await Promise.all([
    prisma.usageRecord.aggregate({
      where: { orgId, createdAt: { gte: startDate, lte: endDate } },
      _sum: { promptTokens: true, completionTokens: true, totalTokens: true, latencyMs: true },
      _count: { id: true },
      _avg: { latencyMs: true },
    }),
    // Tokens per key
    prisma.usageRecord.groupBy({
      by:     ['apiKeyId'],
      where:  { orgId, createdAt: { gte: startDate, lte: endDate } },
      _sum:   { totalTokens: true },
      _count: { id: true },
      orderBy: { _sum: { totalTokens: 'desc' } },
      take:    10,
    }),
    // Tokens per endpoint
    prisma.usageRecord.groupBy({
      by:     ['endpoint'],
      where:  { orgId, createdAt: { gte: startDate, lte: endDate } },
      _sum:   { totalTokens: true },
      _count: { id: true },
    }),
  ]);

  return {
    period: { start: startDate, end: endDate },
    totals: {
      requests:         agg._count.id,
      promptTokens:     agg._sum.promptTokens     ?? 0,
      completionTokens: agg._sum.completionTokens ?? 0,
      totalTokens:      agg._sum.totalTokens      ?? 0,
      avgLatencyMs:     Math.round(agg._avg.latencyMs ?? 0),
    },
    byKey,
    byEndpoint,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Daily timeline
// ─────────────────────────────────────────────────────────────────────────────

export async function getUsageTimeline(
  orgId:     string,
  startDate: Date,
  endDate:   Date,
) {
  // Use raw SQL for date truncation — Prisma doesn't have a cross-DB trunc helper
  const rows = await prisma.$queryRaw<Array<{
    day:          Date;
    requests:     bigint;
    total_tokens: bigint;
    avg_latency:  number;
  }>>`
    SELECT
      date_trunc('day', "createdAt" AT TIME ZONE 'UTC') AS day,
      COUNT(*)                 AS requests,
      SUM("totalTokens")       AS total_tokens,
      AVG("latencyMs")         AS avg_latency
    FROM "UsageRecord"
    WHERE "orgId" = ${orgId}
      AND "createdAt" >= ${startDate}
      AND "createdAt" <= ${endDate}
    GROUP BY 1
    ORDER BY 1 ASC
  `;

  return rows.map((r) => ({
    day:         r.day,
    requests:    Number(r.requests),
    totalTokens: Number(r.total_tokens),
    avgLatencyMs: Math.round(r.avg_latency),
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Recent requests (paginated)
// ─────────────────────────────────────────────────────────────────────────────

export async function listRecentRequests(
  orgId:   string,
  page:    number,
  limit:   number,
  apiKeyId?: string,
) {
  const where = { orgId, ...(apiKeyId ? { apiKeyId } : {}) };
  const [records, total] = await Promise.all([
    prisma.usageRecord.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      skip:    (page - 1) * limit,
      take:    limit,
      select: {
        id:               true,
        endpoint:         true,
        method:           true,
        promptTokens:     true,
        completionTokens: true,
        totalTokens:      true,
        latencyMs:        true,
        statusCode:       true,
        model:            true,
        createdAt:        true,
        apiKey: { select: { id: true, name: true, keyPrefix: true } },
      },
    }),
    prisma.usageRecord.count({ where }),
  ]);

  return { data: records, total, page, limit, hasMore: total > page * limit };
}
