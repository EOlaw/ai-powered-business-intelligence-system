/**
 * InsightSerenity API Gateway — Billing Service
 * ===============================================
 * Manages subscription plans, quota resets, and plan upgrades.
 *
 * Plan hierarchy:
 *   FREE → STARTER → PRO → ENTERPRISE
 *
 * Quota reset:
 *   currentDayTokens is reset to 0 at midnight UTC each day.
 *   The rate-limit middleware reads the Redis quota counter, not the DB field,
 *   so the DB field is only used for billing analytics and invoicing.
 *
 * Plan changes:
 *   Upgrades take effect immediately.
 *   Downgrades take effect at the next billing cycle (endsAt).
 *   This file handles plan metadata — actual payment processing would integrate
 *   with Stripe here (omitted as InsightSerenity doesn't use third-party billing).
 */

import { prisma }      from '../../db/client.js';
import { PLAN_LIMITS } from '../../config/settings.js';
import { getLogger }   from '../../observability/logger.js';
import type { Plan }   from '@prisma/client';

const log = getLogger('billing-service');

// ─────────────────────────────────────────────────────────────────────────────
// Get subscription
// ─────────────────────────────────────────────────────────────────────────────

export async function getSubscription(orgId: string) {
  const sub = await prisma.subscription.findUnique({ where: { orgId } });
  if (!sub) {
    throw Object.assign(new Error('Subscription not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }
  return sub;
}

// ─────────────────────────────────────────────────────────────────────────────
// Upgrade / downgrade plan
// ─────────────────────────────────────────────────────────────────────────────

export async function changePlan(orgId: string, newPlan: Plan) {
  const limits = PLAN_LIMITS[newPlan] ?? PLAN_LIMITS.FREE;
  const tokensPerDay    = limits.tokensPerDay === Infinity ? 999_999_999 : limits.tokensPerDay;
  const requestsPerMin  = limits.requestsPerMin;

  const sub = await prisma.subscription.update({
    where: { orgId },
    data: {
      plan:           newPlan,
      status:         'ACTIVE',
      tokensPerDay:   tokensPerDay as number,
      requestsPerMin,
    },
  });

  // Also update the org's plan field for quick reads
  await prisma.organization.update({ where: { id: orgId }, data: { plan: newPlan } });

  log.info({ orgId, plan: newPlan }, 'Plan changed');
  return sub;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reset daily quota (called by cron job at midnight UTC)
// ─────────────────────────────────────────────────────────────────────────────

export async function resetDailyQuotas(): Promise<number> {
  const result = await prisma.subscription.updateMany({
    data: { currentDayTokens: 0, quotaResetAt: new Date() },
  });
  log.info({ count: result.count }, 'Daily token quotas reset');
  return result.count;
}

// ─────────────────────────────────────────────────────────────────────────────
// Usage summary for billing
// ─────────────────────────────────────────────────────────────────────────────

export async function getBillingSummary(orgId: string, year: number, month: number) {
  const start = new Date(Date.UTC(year, month - 1, 1));
  const end   = new Date(Date.UTC(year, month, 1));

  const [sub, aggregation] = await Promise.all([
    prisma.subscription.findUnique({ where: { orgId } }),
    prisma.usageRecord.aggregate({
      where: { orgId, createdAt: { gte: start, lt: end } },
      _sum: {
        promptTokens:     true,
        completionTokens: true,
        totalTokens:      true,
      },
      _count: { id: true },
    }),
  ]);

  return {
    period: { year, month, start, end },
    plan:   sub?.plan ?? 'FREE',
    status: sub?.status ?? 'ACTIVE',
    usage: {
      requests:         aggregation._count.id,
      promptTokens:     aggregation._sum.promptTokens     ?? 0,
      completionTokens: aggregation._sum.completionTokens ?? 0,
      totalTokens:      aggregation._sum.totalTokens      ?? 0,
    },
    limits: {
      tokensPerDay:   sub?.tokensPerDay   ?? PLAN_LIMITS.FREE.tokensPerDay,
      requestsPerMin: sub?.requestsPerMin ?? PLAN_LIMITS.FREE.requestsPerMin,
    },
  };
}
