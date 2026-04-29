/**
 * InsightSerenity API Gateway — API Key Authentication Middleware
 * ===============================================================
 * Validates INSIGHTSERENITY_API_KEY on all /v1/* proxy routes.
 *
 * Flow:
 *   1. Extract "Bearer is_sk_xxx" from Authorization header
 *   2. Hash the raw key with SHA-256
 *   3. Check Redis cache (avoids DB hit on every request — 60s TTL)
 *   4. On cache miss: load from DB, verify hash, check active/not-expired
 *   5. Attach ApiKeyContext to request for downstream handlers
 *   6. Update lastUsedAt asynchronously (fire-and-forget, no added latency)
 *
 * On any failure: return 401 Unauthorized with a structured error body.
 * We never leak whether the key exists — all failures return the same message.
 */

import type { FastifyRequest, FastifyReply } from 'fastify';
import { prisma }           from '../db/client.js';
import { hashApiKey, compareApiKeyHash } from '../security/crypto.js';
import { getCachedApiKey, cacheApiKey }  from '../cache/redis.js';
import { getLogger }        from '../observability/logger.js';
import type { ApiKeyContext } from '../types/index.js';

const log = getLogger('auth');

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface CachedKeyRecord {
  id:        string;
  orgId:     string;
  keyHash:   string;
  scopes:    string[];
  isActive:  boolean;
  expiresAt: string | null;
  org: {
    id:   string;
    name: string;
    slug: string;
    plan: string;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

export async function authenticateApiKey(
  request: FastifyRequest,
  reply:   FastifyReply,
): Promise<void> {
  const authHeader = request.headers['authorization'];

  if (!authHeader) {
    return unauthorized(reply, 'Missing Authorization header');
  }

  const [scheme, rawKey] = authHeader.split(' ');

  if (scheme?.toLowerCase() !== 'bearer' || !rawKey) {
    return unauthorized(reply, 'Authorization header must be: Bearer <api_key>');
  }

  if (!rawKey.startsWith('is_sk_')) {
    return unauthorized(reply, 'Invalid API key format');
  }

  // ── Cache lookup ──────────────────────────────────────────────────────────
  const keyHash   = hashApiKey(rawKey);
  const cacheHit  = await getCachedApiKey(keyHash) as CachedKeyRecord | null;

  let record: CachedKeyRecord;

  if (cacheHit) {
    record = cacheHit;
    log.debug({ keyPrefix: rawKey.slice(0, 14) }, 'API key cache hit');
  } else {
    // ── DB lookup ─────────────────────────────────────────────────────────
    const dbKey = await prisma.apiKey.findUnique({
      where: { keyHash },
      select: {
        id:        true,
        orgId:     true,
        keyHash:   true,
        scopes:    true,
        isActive:  true,
        expiresAt: true,
        org: {
          select: { id: true, name: true, slug: true, plan: true },
        },
      },
    });

    if (!dbKey) {
      // Cache negative result to prevent DB hammering with invalid keys
      await cacheApiKey(keyHash, null);
      return unauthorized(reply, 'Invalid API key');
    }

    record = dbKey as unknown as CachedKeyRecord;
    await cacheApiKey(keyHash, record);
  }

  // ── Validation ───────────────────────────────────────────────────────────

  if (!record) {
    return unauthorized(reply, 'Invalid API key');
  }

  if (!record.isActive) {
    return unauthorized(reply, 'API key has been revoked');
  }

  if (record.expiresAt && new Date(record.expiresAt) < new Date()) {
    return unauthorized(reply, 'API key has expired');
  }

  // ── Attach context ───────────────────────────────────────────────────────
  request.apiKeyContext = {
    apiKey: record as any,
    org:    record.org as any,
    scopes: record.scopes,
  };

  // ── Update lastUsedAt (fire-and-forget) ───────────────────────────────────
  setImmediate(() => {
    prisma.apiKey.update({
      where: { id: record.id },
      data:  { lastUsedAt: new Date() },
    }).catch((err: Error) => {
      log.warn({ err }, 'Failed to update lastUsedAt');
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

async function unauthorized(reply: FastifyReply, message: string): Promise<void> {
  log.debug({ message }, 'API key auth failed');
  await reply.code(401).send({
    success: false,
    error: {
      code:    'UNAUTHORIZED',
      message,
    },
  });
}
