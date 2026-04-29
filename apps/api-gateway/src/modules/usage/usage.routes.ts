/**
 * Usage Routes
 * ------------
 * GET /orgs/:orgId/usage                  — Overview for a date range
 * GET /orgs/:orgId/usage/timeline         — Daily breakdown for charts
 * GET /orgs/:orgId/usage/requests         — Paginated request log
 */

import type { FastifyInstance } from 'fastify';
import { verifyAccessToken }    from '../auth/strategies/jwt.strategy.js';
import * as UsageService        from './usage.service.js';

export async function usageRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.addHook('preHandler', jwtGuard);

  fastify.get<{ Params: { orgId: string }; Querystring: { start?: string; end?: string } }>(
    '/orgs/:orgId/usage',
    async (req: any, reply) => {
      const { start, end } = req.query;
      const endDate   = end   ? new Date(end)   : new Date();
      const startDate = start ? new Date(start)  : new Date(Date.now() - 30 * 86_400_000); // 30-day default
      const overview  = await UsageService.getUsageOverview(req.params.orgId, startDate, endDate);
      return reply.send({ success: true, data: overview });
    },
  );

  fastify.get<{ Params: { orgId: string }; Querystring: { start?: string; end?: string } }>(
    '/orgs/:orgId/usage/timeline',
    async (req: any, reply) => {
      const { start, end } = req.query;
      const endDate   = end   ? new Date(end)   : new Date();
      const startDate = start ? new Date(start)  : new Date(Date.now() - 30 * 86_400_000);
      const timeline  = await UsageService.getUsageTimeline(req.params.orgId, startDate, endDate);
      return reply.send({ success: true, data: timeline });
    },
  );

  fastify.get<{ Params: { orgId: string }; Querystring: { page?: string; limit?: string; apiKeyId?: string } }>(
    '/orgs/:orgId/usage/requests',
    async (req: any, reply) => {
      const page     = Math.max(1,  parseInt(req.query.page  ?? '1',  10));
      const limit    = Math.min(100, parseInt(req.query.limit ?? '50', 10));
      const result   = await UsageService.listRecentRequests(req.params.orgId, page, limit, req.query.apiKeyId);
      return reply.send({ success: true, data: result });
    },
  );
}

async function jwtGuard(request: any, reply: any): Promise<void> {
  const [, token] = (request.headers['authorization'] ?? '').split(' ');
  if (!token) return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Authentication required' } });
  try { request.jwtUser = verifyAccessToken(token); }
  catch { return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Invalid or expired token' } }); }
}
