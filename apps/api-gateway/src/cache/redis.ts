/**
 * InsightSerenity API Gateway — Redis Client
 * ===========================================
 * Singleton ioredis client shared by:
 *   - Rate limiter (sliding window counters)
 *   - API key validation cache (avoid DB hit on every request)
 *   - BullMQ job queues (webhook delivery, usage aggregation)
 *   - Token quota caching (current day token count)
 *
 * Connection retry:
 *   ioredis automatically reconnects on disconnect with exponential backoff.
 *   We cap the retry delay at 5 seconds to limit queue accumulation.
 *
 * Key naming convention:
 *   rate:{orgId}:{windowStart}  → sliding window request counter
 *   apikey:{keyHash}            → cached ApiKey+Org record (TTL 60s)
 *   quota:{orgId}:{date}        → daily token count (TTL until midnight)
 */

import IORedis from 'ioredis';
import { config } from '../config/settings.js';
import { getLogger } from '../observability/logger.js';

const log = getLogger('redis');

// ─────────────────────────────────────────────────────────────────────────────
// Client factory
// ─────────────────────────────────────────────────────────────────────────────

function createRedisClient(): IORedis {
  const client = new IORedis(config.REDIS_URL, {
    maxRetriesPerRequest: null,   // Required by BullMQ
    enableReadyCheck:     true,
    lazyConnect:          false,
    retryStrategy: (times) => {
      const delay = Math.min(times * 50, 5_000);
      log.warn({ attempt: times, delayMs: delay }, 'Redis reconnecting');
      return delay;
    },
  });

  client.on('connect',  () => log.info('Redis connected'));
  client.on('ready',    () => log.debug('Redis ready'));
  client.on('close',    () => log.warn('Redis connection closed'));
  client.on('error',    (err: Error) => log.error({ err }, 'Redis error'));
  client.on('reconnecting', () => log.warn('Redis reconnecting'));

  return client;
}

// ─────────────────────────────────────────────────────────────────────────────
// Singleton
// ─────────────────────────────────────────────────────────────────────────────

// Single shared instance — BullMQ workers must use separate connections
// because ioredis in subscriber mode cannot issue regular commands.
export const redis = createRedisClient();

/**
 * Create a new, independent Redis connection.
 * Use for BullMQ workers that need a dedicated subscriber connection.
 */
export function createConnection(): IORedis {
  return createRedisClient();
}

/**
 * Gracefully close the shared connection.
 * Called on SIGTERM/SIGINT to drain in-flight commands before exit.
 */
export async function closeRedis(): Promise<void> {
  await redis.quit();
  log.info('Redis connection closed gracefully');
}

// ─────────────────────────────────────────────────────────────────────────────
// Rate limiting helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sliding window rate limiter.
 *
 * Uses a Redis sorted set where each member is a unique request ID
 * and the score is the Unix timestamp in milliseconds.
 *
 * On each request:
 *   1. Remove all entries older than `windowMs`
 *   2. Count remaining entries
 *   3. If count < limit, add this request and allow
 *   4. If count >= limit, reject with 429
 *
 * @returns { allowed: boolean; remaining: number; resetMs: number }
 */
export async function slidingWindowRateLimit(
  key:      string,
  limit:    number,
  windowMs: number,
): Promise<{ allowed: boolean; remaining: number; resetMs: number }> {
  const now       = Date.now();
  const windowStart = now - windowMs;
  const requestId   = `${now}-${Math.random()}`;   // Unique member per request

  const pipeline = redis.pipeline();
  pipeline.zremrangebyscore(key, '-inf', windowStart); // Evict expired entries
  pipeline.zadd(key, now, requestId);                  // Record this request
  pipeline.zcard(key);                                 // Count entries in window
  pipeline.expire(key, Math.ceil(windowMs / 1000) + 1);

  const results = await pipeline.exec();
  const count   = (results?.[2]?.[1] as number) ?? 0;

  if (count > limit) {
    // Over limit — remove the entry we just added and return rejection
    await redis.zrem(key, requestId);
    return { allowed: false, remaining: 0, resetMs: windowStart + windowMs };
  }

  return {
    allowed:   true,
    remaining: Math.max(0, limit - count),
    resetMs:   now + windowMs,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// API key caching
// ─────────────────────────────────────────────────────────────────────────────

const KEY_CACHE_TTL = 60; // seconds — balance between freshness and DB load

/** Cache a serialised ApiKey record. Pass null to cache a "key not found". */
export async function cacheApiKey(keyHash: string, data: unknown): Promise<void> {
  await redis.setex(`apikey:${keyHash}`, KEY_CACHE_TTL, JSON.stringify(data));
}

/** Retrieve a cached ApiKey record, or null if not cached / expired. */
export async function getCachedApiKey(keyHash: string): Promise<unknown | null> {
  const raw = await redis.get(`apikey:${keyHash}`);
  return raw ? JSON.parse(raw) : null;
}

/** Invalidate the cache entry for a specific key hash (on revoke/rotate). */
export async function invalidateApiKeyCache(keyHash: string): Promise<void> {
  await redis.del(`apikey:${keyHash}`);
}
