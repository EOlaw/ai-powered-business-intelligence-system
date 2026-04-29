/**
 * Billing Routes
 * --------------
 * GET   /orgs/:orgId/billing            — Current subscription + limits
 * POST  /orgs/:orgId/billing/plan       — Upgrade or change plan
 * GET   /orgs/:orgId/billing/summary    — Monthly usage summary
 */

import type { FastifyInstance } from 'fastify';
import { z }                    from 'zod';
import { verifyAccessToken }    from '../auth/strategies/jwt.strategy.js';
import * as BillingService      from './billing.service.js';

const ChangePlanSchema = z.object({
  plan: z.enum(['FREE', 'STARTER', 'PRO', 'ENTERPRISE']),
});

export async function billingRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.addHook('preHandler', jwtGuard);

  fastify.get<{ Params: { orgId: string } }>('/orgs/:orgId/billing', async (req: any, reply) => {
    try {
      const sub = await BillingService.getSubscription(req.params.orgId);
      return reply.send({ success: true, data: sub });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.post<{ Params: { orgId: string } }>('/orgs/:orgId/billing/plan', async (req: any, reply) => {
    const body = ChangePlanSchema.safeParse(req.body);
    if (!body.success) {
      return reply.code(400).send({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Invalid plan' } });
    }
    try {
      const sub = await BillingService.changePlan(req.params.orgId, body.data.plan);
      return reply.send({ success: true, data: sub });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.get<{ Params: { orgId: string }; Querystring: { year?: string; month?: string } }>(
    '/orgs/:orgId/billing/summary',
    async (req: any, reply) => {
      const now   = new Date();
      const year  = parseInt(req.query.year  ?? now.getUTCFullYear(),  10);
      const month = parseInt(req.query.month ?? now.getUTCMonth() + 1, 10);
      const summary = await BillingService.getBillingSummary(req.params.orgId, year, month);
      return reply.send({ success: true, data: summary });
    },
  );
}

async function jwtGuard(request: any, reply: any): Promise<void> {
  const [, token] = (request.headers['authorization'] ?? '').split(' ');
  if (!token) return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Authentication required' } });
  try { request.jwtUser = verifyAccessToken(token); }
  catch { return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Invalid or expired token' } }); }
}
