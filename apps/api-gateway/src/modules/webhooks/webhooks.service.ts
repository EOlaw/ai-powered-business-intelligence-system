/**
 * InsightSerenity API Gateway — Webhooks Service
 * ================================================
 * Manages webhook endpoints that receive platform events.
 *
 * Supported events:
 *   usage.limit_reached    — When 80% or 100% of daily token quota is consumed
 *   key.revoked            — When an API key is revoked
 *   subscription.changed   — When the org's plan changes
 *
 * Delivery:
 *   Webhook payloads are dispatched via BullMQ with 3 retry attempts.
 *   Each delivery is signed with HMAC-SHA256 so receivers can verify authenticity.
 *
 * Security:
 *   The webhook secret is generated randomly on creation.
 *   It is returned once and cannot be retrieved — only regenerated.
 */

import { prisma }                from '../../db/client.js';
import { generateWebhookSecret } from '../../security/crypto.js';
import { enqueueWebhook }        from '../../queue/worker.js';
import { getLogger }             from '../../observability/logger.js';

const log = getLogger('webhooks-service');

export const VALID_EVENTS = [
  'usage.limit_reached',
  'key.revoked',
  'subscription.changed',
] as const;

export type WebhookEvent = (typeof VALID_EVENTS)[number];

// ─────────────────────────────────────────────────────────────────────────────
// CRUD
// ─────────────────────────────────────────────────────────────────────────────

export async function createWebhook(orgId: string, url: string, events: string[]) {
  const secret = generateWebhookSecret();

  const webhook = await prisma.webhook.create({
    data: { orgId, url, events, secret },
  });

  log.info({ orgId, webhookId: webhook.id }, 'Webhook created');
  // Return secret once — it is stored but not retrievable after this
  return { ...webhook, secret };
}

export async function listWebhooks(orgId: string) {
  return prisma.webhook.findMany({
    where:   { orgId },
    select:  { id: true, url: true, events: true, isActive: true, createdAt: true },
    orderBy: { createdAt: 'desc' },
  });
}

export async function deleteWebhook(id: string, orgId: string) {
  const wh = await prisma.webhook.findFirst({ where: { id, orgId } });
  if (!wh) throw Object.assign(new Error('Webhook not found'), { statusCode: 404, code: 'NOT_FOUND' });
  await prisma.webhook.delete({ where: { id } });
}

export async function rotateWebhookSecret(id: string, orgId: string) {
  const wh = await prisma.webhook.findFirst({ where: { id, orgId } });
  if (!wh) throw Object.assign(new Error('Webhook not found'), { statusCode: 404, code: 'NOT_FOUND' });

  const secret = generateWebhookSecret();
  await prisma.webhook.update({ where: { id }, data: { secret } });
  return { id, secret };
}

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Dispatch a platform event to all active webhooks subscribed to it.
 * Called internally by services when events occur.
 */
export async function dispatchEvent(
  orgId:   string,
  event:   WebhookEvent,
  payload: Record<string, unknown>,
): Promise<void> {
  const webhooks = await prisma.webhook.findMany({
    where: { orgId, isActive: true, events: { has: event } },
  });

  for (const wh of webhooks) {
    await enqueueWebhook({
      webhookId: wh.id,
      orgId:     wh.orgId,
      url:       wh.url,
      secret:    wh.secret,
      event,
      payload,
    });
  }

  if (webhooks.length > 0) {
    log.debug({ orgId, event, count: webhooks.length }, 'Webhook event dispatched');
  }
}
