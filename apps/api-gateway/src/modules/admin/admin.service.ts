/**
 * InsightSerenity API Gateway — Admin Service
 * ============================================
 * Platform-level admin operations. Accessible only to users with role=ADMIN.
 *
 * Capabilities:
 *   - List all organisations and users (paginated)
 *   - Lookup and impersonate any organisation
 *   - Override a subscription plan (manual billing adjustments)
 *   - Revoke any API key on the platform
 *   - View platform-wide usage aggregates
 *   - Trigger daily quota reset manually
 */

import { prisma }           from '../../db/client.js';
import { getLogger }        from '../../observability/logger.js';
import * as BillingService  from '../billing/billing.service.js';
import { invalidateApiKeyCache } from '../../cache/redis.js';
import type { Plan } from '@prisma/client';

const log = getLogger('admin-service');

// ─────────────────────────────────────────────────────────────────────────────
// Users
// ─────────────────────────────────────────────────────────────────────────────

export async function listAllUsers(page: number, limit: number) {
  const [users, total] = await Promise.all([
    prisma.user.findMany({
      orderBy: { createdAt: 'desc' },
      skip:    (page - 1) * limit,
      take:    limit,
      select: {
        id:          true,
        email:       true,
        role:        true,
        createdAt:   true,
        memberships: { select: { orgId: true, role: true } },
      },
    }),
    prisma.user.count(),
  ]);
  return { data: users, total, page, limit, hasMore: total > page * limit };
}

export async function promoteToAdmin(userId: string) {
  const user = await prisma.user.update({
    where: { id: userId },
    data:  { role: 'ADMIN' },
  });
  log.info({ userId }, 'User promoted to admin');
  return user;
}

// ─────────────────────────────────────────────────────────────────────────────
// Organisations
// ─────────────────────────────────────────────────────────────────────────────

export async function listAllOrgs(page: number, limit: number) {
  const [orgs, total] = await Promise.all([
    prisma.organization.findMany({
      orderBy: { createdAt: 'desc' },
      skip:    (page - 1) * limit,
      take:    limit,
      include: {
        subscription: { select: { plan: true, status: true, currentDayTokens: true } },
        _count:       { select: { apiKeys: true, members: true } },
      },
    }),
    prisma.organization.count(),
  ]);
  return { data: orgs, total, page, limit, hasMore: total > page * limit };
}

export async function adminSetPlan(orgId: string, plan: Plan, adminUserId: string) {
  const result = await BillingService.changePlan(orgId, plan);
  log.info({ orgId, plan, adminUserId }, 'Admin overrode plan');
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// API Keys
// ─────────────────────────────────────────────────────────────────────────────

export async function adminRevokeKey(keyId: string, adminUserId: string) {
  const key = await prisma.apiKey.findUnique({ where: { id: keyId } });
  if (!key) throw Object.assign(new Error('Key not found'), { statusCode: 404, code: 'NOT_FOUND' });

  await prisma.apiKey.update({ where: { id: keyId }, data: { isActive: false, revokedAt: new Date() } });
  await invalidateApiKeyCache(key.keyHash);

  log.warn({ keyId, orgId: key.orgId, adminUserId }, 'Admin revoked API key');
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform stats
// ─────────────────────────────────────────────────────────────────────────────

export async function getPlatformStats() {
  const [users, orgs, keys, todayUsage] = await Promise.all([
    prisma.user.count(),
    prisma.organization.count(),
    prisma.apiKey.count({ where: { isActive: true } }),
    prisma.usageRecord.aggregate({
      where: { createdAt: { gte: new Date(new Date().setUTCHours(0, 0, 0, 0)) } },
      _sum:   { totalTokens: true },
      _count: { id: true },
    }),
  ]);

  return {
    users,
    orgs,
    activeApiKeys:        keys,
    todayRequests:        todayUsage._count.id,
    todayTotalTokens:     todayUsage._sum.totalTokens ?? 0,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Quota management
// ─────────────────────────────────────────────────────────────────────────────

export async function triggerQuotaReset() {
  const count = await BillingService.resetDailyQuotas();
  log.info({ count }, 'Admin triggered quota reset');
  return { orgsReset: count };
}
