/**
 * InsightSerenity API Gateway — Prisma Client Singleton
 * ======================================================
 * Exports a single PrismaClient instance shared across the entire app.
 *
 * In production, one connection pool is created at startup and reused.
 * In development (with tsx --watch), module hot-reload can create multiple
 * instances causing "too many connections" errors — the global guard below
 * prevents that by caching the instance on the global object.
 */

import { PrismaClient } from '@prisma/client';
import { config } from '../config/settings.js';
import { getLogger } from '../observability/logger.js';

const log = getLogger('prisma');

// ─────────────────────────────────────────────────────────────────────────────
// Client construction
// ─────────────────────────────────────────────────────────────────────────────

function buildPrismaClient(): PrismaClient {
  const client = new PrismaClient({
    log: config.NODE_ENV === 'development'
      ? [
          { emit: 'event', level: 'query' },
          { emit: 'event', level: 'warn'  },
          { emit: 'event', level: 'error' },
        ]
      : [
          { emit: 'event', level: 'warn'  },
          { emit: 'event', level: 'error' },
        ],
  });

  // Forward Prisma log events to Pino
  (client as any).$on('warn',  (e: { message: string }) => log.warn(e.message));
  (client as any).$on('error', (e: { message: string }) => log.error(e.message));

  if (config.NODE_ENV === 'development') {
    (client as any).$on('query', (e: { query: string; duration: number }) => {
      log.debug({ query: e.query, durationMs: e.duration }, 'SQL');
    });
  }

  return client;
}

// ─────────────────────────────────────────────────────────────────────────────
// Singleton (global guard for hot-reload safety)
// ─────────────────────────────────────────────────────────────────────────────

const globalForPrisma = globalThis as unknown as { prisma?: PrismaClient };

export const prisma: PrismaClient =
  globalForPrisma.prisma ?? buildPrismaClient();

if (config.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

/**
 * Gracefully disconnect Prisma.
 * Called on SIGTERM so in-flight queries complete before the process exits.
 */
export async function disconnectPrisma(): Promise<void> {
  await prisma.$disconnect();
  log.info('Prisma disconnected');
}
