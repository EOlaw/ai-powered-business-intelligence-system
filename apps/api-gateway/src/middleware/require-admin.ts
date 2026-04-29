/**
 * InsightSerenity API Gateway — Admin Authorization Middleware
 * ============================================================
 * Checks that the authenticated JWT user holds the ADMIN role.
 * Applied to all /admin/* routes.
 *
 * Must be registered AFTER the JWT verification hook — relies on
 * request.jwtUser being populated.
 */

import type { FastifyRequest, FastifyReply } from 'fastify';
import { getLogger } from '../observability/logger.js';

const log = getLogger('require-admin');

export async function requireAdmin(
  request: FastifyRequest,
  reply:   FastifyReply,
): Promise<void> {
  const user = request.jwtUser;

  if (!user) {
    return reply.code(401).send({
      success: false,
      error: { code: 'UNAUTHORIZED', message: 'Authentication required' },
    });
  }

  if (user.role !== 'ADMIN') {
    log.warn({ userId: user.sub, role: user.role }, 'Admin access denied');
    return reply.code(403).send({
      success: false,
      error: { code: 'FORBIDDEN', message: 'Admin access required' },
    });
  }
}
