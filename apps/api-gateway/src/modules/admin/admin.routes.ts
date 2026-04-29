/**
 * Admin Routes — all require UserRole=ADMIN JWT
 * -----------------------------------------------
 * GET  /admin/stats                        — Platform-wide stats
 * GET  /admin/users                        — List all users
 * POST /admin/users/:userId/promote        — Promote to admin
 * GET  /admin/orgs                         — List all orgs
 * POST /admin/orgs/:orgId/plan             — Override subscription plan
 * POST /admin/keys/:keyId/revoke           — Revoke any API key
 * POST /admin/quota/reset                  — Manually reset daily quotas
 */

import type { FastifyInstance } from 'fastify';
import { z }                    from 'zod';
import { verifyAccessToken }    from '../auth/strategies/jwt.strategy.js';
import { requireAdmin }         from '../../middleware/require-admin.js';
import * as AdminService        from './admin.service.js';

export async function adminRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.addHook('preHandler', jwtGuard);
  fastify.addHook('preHandler', requireAdmin);

  fastify.get('/admin/stats', async (_req, reply) => {
    const stats = await AdminService.getPlatformStats();
    return reply.send({ success: true, data: stats });
  });

  fastify.get<{ Querystring: { page?: string; limit?: string } }>('/admin/users', async (req, reply) => {
    const page  = Math.max(1,  parseInt((req.query as any).page  ?? '1',  10));
    const limit = Math.min(100, parseInt((req.query as any).limit ?? '50', 10));
    const data  = await AdminService.listAllUsers(page, limit);
    return reply.send({ success: true, data });
  });

  fastify.post<{ Params: { userId: string } }>('/admin/users/:userId/promote', async (req, reply) => {
    const user = await AdminService.promoteToAdmin(req.params.userId);
    return reply.send({ success: true, data: user });
  });

  fastify.get<{ Querystring: { page?: string; limit?: string } }>('/admin/orgs', async (req, reply) => {
    const page  = Math.max(1,  parseInt((req.query as any).page  ?? '1',  10));
    const limit = Math.min(100, parseInt((req.query as any).limit ?? '50', 10));
    const data  = await AdminService.listAllOrgs(page, limit);
    return reply.send({ success: true, data });
  });

  const PlanSchema = z.object({ plan: z.enum(['FREE', 'STARTER', 'PRO', 'ENTERPRISE']) });

  fastify.post<{ Params: { orgId: string } }>('/admin/orgs/:orgId/plan', async (req: any, reply) => {
    const body = PlanSchema.safeParse(req.body);
    if (!body.success) return reply.code(400).send({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Invalid plan' } });
    const sub = await AdminService.adminSetPlan(req.params.orgId, body.data.plan, req.jwtUser.sub);
    return reply.send({ success: true, data: sub });
  });

  fastify.post<{ Params: { keyId: string } }>('/admin/keys/:keyId/revoke', async (req: any, reply) => {
    try {
      await AdminService.adminRevokeKey(req.params.keyId, req.jwtUser.sub);
      return reply.send({ success: true, data: { message: 'Key revoked' } });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.post('/admin/quota/reset', async (_req, reply) => {
    const result = await AdminService.triggerQuotaReset();
    return reply.send({ success: true, data: result });
  });
}

async function jwtGuard(request: any, reply: any): Promise<void> {
  const [, token] = (request.headers['authorization'] ?? '').split(' ');
  if (!token) return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Authentication required' } });
  try { request.jwtUser = verifyAccessToken(token); }
  catch { return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Invalid or expired token' } }); }
}
