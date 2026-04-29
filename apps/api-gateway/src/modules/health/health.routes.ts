/**
 * Health Routes
 * -------------
 * GET /health         — Liveness: server process is running
 * GET /health/ready   — Readiness: DB + Redis connections are alive
 */

import type { FastifyInstance } from 'fastify';
import { prisma }   from '../../db/client.js';
import { redis }    from '../../cache/redis.js';
import { getLogger } from '../../observability/logger.js';
import { config }    from '../../config/settings.js';

const log = getLogger('health');

let startTime = Date.now();

export async function healthRoutes(fastify: FastifyInstance): Promise<void> {
  startTime = Date.now();

  // ── GET /health ──────────────────────────────────────────────────────────
  fastify.get('/health', async (_req, reply) => {
    return reply.send({
      status:  'ok',
      service: 'api-gateway',
      version: config.NODE_ENV,
      uptime:  Math.round((Date.now() - startTime) / 1000),
    });
  });

  // ── GET /health/ready ────────────────────────────────────────────────────
  fastify.get('/health/ready', async (_req, reply) => {
    const checks: Record<string, 'ok' | 'error'> = {};
    let healthy = true;

    // Database check
    try {
      await prisma.$queryRaw`SELECT 1`;
      checks['database'] = 'ok';
    } catch (err) {
      log.error({ err }, 'Database health check failed');
      checks['database'] = 'error';
      healthy = false;
    }

    // Redis check
    try {
      await redis.ping();
      checks['redis'] = 'ok';
    } catch (err) {
      log.error({ err }, 'Redis health check failed');
      checks['redis'] = 'error';
      healthy = false;
    }

    return reply.code(healthy ? 200 : 503).send({
      status: healthy ? 'ready' : 'not_ready',
      checks,
    });
  });
}
