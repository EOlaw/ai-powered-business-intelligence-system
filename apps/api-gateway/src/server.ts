/**
 * InsightSerenity API Gateway — Server Entry Point
 * =================================================
 * Boots the Fastify server, starts background workers, and handles
 * graceful shutdown on SIGTERM/SIGINT.
 *
 * Startup sequence:
 *   1. Build Fastify app (register all plugins + routes)
 *   2. Start BullMQ background workers (webhook delivery, usage recording)
 *   3. Bind to PORT:HOST from config
 *   4. Register SIGTERM/SIGINT handlers for graceful shutdown
 *
 * Graceful shutdown sequence (on SIGTERM):
 *   1. Stop accepting new connections (fastify.close())
 *   2. Wait for in-flight requests to complete (Fastify handles this)
 *   3. Close BullMQ workers (drain queues)
 *   4. Disconnect Prisma (flush DB connection pool)
 *   5. Close Redis connection
 *   6. Exit with code 0
 */

import { buildApp }         from './app.js';
import { config }           from './config/settings.js';
import { logger }           from './observability/logger.js';
import { startWorkers, stopWorkers } from './queue/worker.js';
import { disconnectPrisma } from './db/client.js';
import { closeRedis }       from './cache/redis.js';

async function start(): Promise<void> {
  const app = await buildApp();

  // ── Start background workers ──────────────────────────────────────────────
  startWorkers();

  // ── Start HTTP server ─────────────────────────────────────────────────────
  try {
    const address = await app.listen({
      port: config.PORT,
      host: config.HOST,
    });

    logger.info({ address, env: config.NODE_ENV }, 'InsightSerenity API Gateway started');
  } catch (err) {
    logger.fatal({ err }, 'Failed to start server');
    process.exit(1);
  }

  // ── Graceful shutdown ─────────────────────────────────────────────────────
  async function shutdown(signal: string): Promise<void> {
    logger.info({ signal }, 'Shutdown signal received');

    try {
      // 1. Stop accepting requests and wait for in-flight to finish
      await app.close();

      // 2. Drain background job workers
      await stopWorkers();

      // 3. Close DB connection pool
      await disconnectPrisma();

      // 4. Close Redis connection
      await closeRedis();

      logger.info('Graceful shutdown complete');
      process.exit(0);
    } catch (err) {
      logger.error({ err }, 'Error during shutdown');
      process.exit(1);
    }
  }

  process.once('SIGTERM', () => shutdown('SIGTERM'));
  process.once('SIGINT',  () => shutdown('SIGINT'));

  // ── Unhandled rejection safety net ───────────────────────────────────────
  process.on('unhandledRejection', (reason) => {
    logger.error({ reason }, 'Unhandled promise rejection');
  });

  process.on('uncaughtException', (err) => {
    logger.fatal({ err }, 'Uncaught exception — forcing shutdown');
    process.exit(1);
  });
}

start();
