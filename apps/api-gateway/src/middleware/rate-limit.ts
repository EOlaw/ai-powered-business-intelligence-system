/**
 * InsightSerenity API Gateway — Rate Limiting Middleware
 * =======================================================
 * Per-API-key sliding window rate limiter backed by Redis.
 *
 * Two limits enforced:
 *   1. requests/minute  — controlled by the org's plan (FREE=10, PRO=300, …)
 *   2. tokens/day       — cumulative tokens consumed since midnight UTC
 *
 * Both checks are performed BEFORE forwarding to the AI engine.
 * This prevents quota evasion even if the engine accepts the request.
 *
 * Response headers (matching OpenAI's format):
 *   X-RateLimit-Limit-Requests:     {limit}
 *   X-RateLimit-Remaining-Requests: {remaining}
 *   X-RateLimit-Reset-Requests:     {ISO-8601 reset time}
 *   X-RateLimit-Limit-Tokens:       {daily token limit}
 *   X-RateLimit-Remaining-Tokens:   {remaining today}
 *
 * On 429: returns Retry-After header with seconds until the window resets.
 */

import type { FastifyRequest, FastifyReply } from 'fastify';
import { slidingWindowRateLimit, redis } from '../cache/redis.js';
import { PLAN_LIMITS }                   from '../config/settings.js';
import { getLogger }                     from '../observability/logger.js';
import { rateLimitedTotal }              from '../observability/metrics.js';

const log = getLogger('rate-limit');

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

export async function checkRateLimit(
  request: FastifyRequest,
  reply:   FastifyReply,
): Promise<void> {
  const ctx = request.apiKeyContext;
  if (!ctx) return;  // Should never happen — authenticate runs first

  const plan   = (ctx.org.plan as keyof typeof PLAN_LIMITS) ?? 'FREE';
  const limits = PLAN_LIMITS[plan] ?? PLAN_LIMITS.FREE;

  // ── 1. Requests-per-minute ──────────────────────────────────────────────
  const rpmKey    = `rate:rpm:${ctx.apiKey.id}`;
  const rpmResult = await slidingWindowRateLimit(rpmKey, limits.requestsPerMin, 60_000);

  reply.header('X-RateLimit-Limit-Requests',     limits.requestsPerMin.toString());
  reply.header('X-RateLimit-Remaining-Requests', rpmResult.remaining.toString());
  reply.header('X-RateLimit-Reset-Requests',     new Date(rpmResult.resetMs).toISOString());

  if (!rpmResult.allowed) {
    const retryAfter = Math.ceil((rpmResult.resetMs - Date.now()) / 1000);
    log.warn({ orgId: ctx.org.id, plan, limit: limits.requestsPerMin }, 'Rate limit exceeded');
    rateLimitedTotal.labels('rpm_exceeded', plan).inc();
    return reply
      .code(429)
      .header('Retry-After', retryAfter.toString())
      .send({
        success: false,
        error: {
          code:    'RATE_LIMIT_EXCEEDED',
          message: `Rate limit exceeded. ${limits.requestsPerMin} requests per minute allowed on the ${plan} plan.`,
        },
      });
  }

  // ── 2. Daily token quota ────────────────────────────────────────────────
  if (limits.tokensPerDay !== Infinity) {
    const todayKey      = `quota:tokens:${ctx.org.id}:${utcDateString()}`;
    const usedTokensRaw = await redis.get(todayKey);
    const usedTokens    = usedTokensRaw ? parseInt(usedTokensRaw, 10) : 0;
    const remaining     = Math.max(0, limits.tokensPerDay - usedTokens);

    reply.header('X-RateLimit-Limit-Tokens',     limits.tokensPerDay.toString());
    reply.header('X-RateLimit-Remaining-Tokens', remaining.toString());

    if (usedTokens >= limits.tokensPerDay) {
      log.warn({ orgId: ctx.org.id, plan, used: usedTokens, limit: limits.tokensPerDay }, 'Daily token quota exceeded');
      rateLimitedTotal.labels('token_quota_exceeded', plan).inc();
      return reply.code(429).send({
        success: false,
        error: {
          code:    'TOKEN_QUOTA_EXCEEDED',
          message: `Daily token quota (${limits.tokensPerDay.toLocaleString()} tokens) exceeded. Quota resets at midnight UTC.`,
        },
      });
    }
  }
}

/**
 * Increment the org's daily token counter after a successful generation.
 * Called fire-and-forget from the proxy handler.
 */
export async function incrementTokenQuota(orgId: string, tokens: number): Promise<void> {
  const key = `quota:tokens:${orgId}:${utcDateString()}`;
  const secondsUntilMidnight = getSecondsUntilMidnightUtc();

  const pipeline = redis.pipeline();
  pipeline.incrby(key, tokens);
  pipeline.expire(key, secondsUntilMidnight + 60); // +60s buffer
  await pipeline.exec();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function utcDateString(): string {
  return new Date().toISOString().slice(0, 10); // "YYYY-MM-DD"
}

function getSecondsUntilMidnightUtc(): number {
  const now      = new Date();
  const midnight = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
  return Math.ceil((midnight.getTime() - now.getTime()) / 1000);
}
