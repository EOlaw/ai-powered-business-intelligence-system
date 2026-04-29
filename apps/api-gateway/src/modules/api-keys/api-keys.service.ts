/**
 * InsightSerenity API Gateway — API Key Service
 * ===============================================
 * CRUD operations for INSIGHTSERENITY_API_KEY management.
 *
 * Key lifecycle:
 *   create   → generates raw key, stores SHA-256 hash + prefix, returns raw key ONCE
 *   list     → returns all keys for an org (prefix only — never the raw key)
 *   rotate   → revokes existing key, issues new one (atomic)
 *   revoke   → sets isActive=false, records revokedAt timestamp
 *   delete   → hard delete (preserves usage history via cascading FK)
 *
 * The raw API key is returned ONLY during create and rotate.
 * It is never stored and cannot be recovered — users must rotate to get a new one.
 */

import { prisma }                        from '../../db/client.js';
import { generateApiKey }                from '../../security/crypto.js';
import { invalidateApiKeyCache }         from '../../cache/redis.js';
import { getLogger }                     from '../../observability/logger.js';
import type { CreateApiKeyInput }        from './api-keys.schema.js';

const log = getLogger('api-keys-service');

// ─────────────────────────────────────────────────────────────────────────────
// Create
// ─────────────────────────────────────────────────────────────────────────────

export async function createApiKey(
  orgId:  string,
  input:  CreateApiKeyInput,
): Promise<{ id: string; rawKey: string; keyPrefix: string; scopes: string[] }> {
  const { rawKey, keyHash, keyPrefix } = generateApiKey();

  const key = await prisma.apiKey.create({
    data: {
      orgId,
      name:      input.name,
      keyHash,
      keyPrefix,
      scopes:    input.scopes,
      ...(input.expiresAt ? { expiresAt: new Date(input.expiresAt) } : {}),
    },
  });

  log.info({ orgId, keyId: key.id, name: input.name }, 'API key created');

  return { id: key.id, rawKey, keyPrefix, scopes: key.scopes };
}

// ─────────────────────────────────────────────────────────────────────────────
// List
// ─────────────────────────────────────────────────────────────────────────────

export async function listApiKeys(orgId: string) {
  return prisma.apiKey.findMany({
    where:  { orgId },
    select: {
      id:         true,
      name:       true,
      keyPrefix:  true,
      scopes:     true,
      isActive:   true,
      lastUsedAt: true,
      expiresAt:  true,
      createdAt:  true,
      revokedAt:  true,
    },
    orderBy: { createdAt: 'desc' },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Get single key (ownership check included)
// ─────────────────────────────────────────────────────────────────────────────

export async function getApiKey(id: string, orgId: string) {
  const key = await prisma.apiKey.findFirst({
    where: { id, orgId },
    select: {
      id:         true,
      name:       true,
      keyPrefix:  true,
      scopes:     true,
      isActive:   true,
      lastUsedAt: true,
      expiresAt:  true,
      createdAt:  true,
      revokedAt:  true,
    },
  });

  if (!key) {
    throw Object.assign(new Error('API key not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }

  return key;
}

// ─────────────────────────────────────────────────────────────────────────────
// Revoke
// ─────────────────────────────────────────────────────────────────────────────

export async function revokeApiKey(id: string, orgId: string): Promise<void> {
  const key = await prisma.apiKey.findFirst({ where: { id, orgId } });
  if (!key) {
    throw Object.assign(new Error('API key not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }

  await prisma.apiKey.update({
    where: { id },
    data:  { isActive: false, revokedAt: new Date() },
  });

  await invalidateApiKeyCache(key.keyHash);
  log.info({ keyId: id, orgId }, 'API key revoked');
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotate — atomic: revoke old, issue new
// ─────────────────────────────────────────────────────────────────────────────

export async function rotateApiKey(
  id:    string,
  orgId: string,
): Promise<{ id: string; rawKey: string; keyPrefix: string }> {
  const existing = await prisma.apiKey.findFirst({ where: { id, orgId } });
  if (!existing) {
    throw Object.assign(new Error('API key not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }

  const { rawKey, keyHash, keyPrefix } = generateApiKey();

  await prisma.$transaction([
    // Revoke the old key
    prisma.apiKey.update({
      where: { id },
      data:  { isActive: false, revokedAt: new Date() },
    }),
    // Create the replacement
    prisma.apiKey.create({
      data: {
        orgId,
        name:     `${existing.name} (rotated)`,
        keyHash,
        keyPrefix,
        scopes:   existing.scopes,
      },
    }),
  ]);

  await invalidateApiKeyCache(existing.keyHash);
  log.info({ oldKeyId: id, orgId }, 'API key rotated');

  const newKey = await prisma.apiKey.findFirst({ where: { keyHash } });
  return { id: newKey!.id, rawKey, keyPrefix };
}

// ─────────────────────────────────────────────────────────────────────────────
// Delete
// ─────────────────────────────────────────────────────────────────────────────

export async function deleteApiKey(id: string, orgId: string): Promise<void> {
  const key = await prisma.apiKey.findFirst({ where: { id, orgId } });
  if (!key) {
    throw Object.assign(new Error('API key not found'), { statusCode: 404, code: 'NOT_FOUND' });
  }

  await prisma.apiKey.delete({ where: { id } });
  await invalidateApiKeyCache(key.keyHash);
  log.info({ keyId: id, orgId }, 'API key deleted');
}
