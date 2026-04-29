/**
 * Webhooks Routes
 * ---------------
 * POST   /orgs/:orgId/webhooks                     — Register endpoint
 * GET    /orgs/:orgId/webhooks                     — List endpoints
 * DELETE /orgs/:orgId/webhooks/:id                 — Delete endpoint
 * POST   /orgs/:orgId/webhooks/:id/rotate-secret   — Regenerate signing secret
 */

import type { FastifyInstance } from 'fastify';
import { z }                    from 'zod';
import { verifyAccessToken }    from '../auth/strategies/jwt.strategy.js';
import * as WebhookService      from './webhooks.service.js';

const CreateWebhookSchema = z.object({
  url:    z.string().url(),
  events: z.array(z.enum(WebhookService.VALID_EVENTS)).min(1),
});

export async function webhookRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.addHook('preHandler', jwtGuard);

  fastify.post<{ Params: { orgId: string } }>('/orgs/:orgId/webhooks', async (req: any, reply) => {
    const body = CreateWebhookSchema.safeParse(req.body);
    if (!body.success) {
      return reply.code(400).send({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Invalid input', details: body.error.flatten() } });
    }
    const wh = await WebhookService.createWebhook(req.params.orgId, body.data.url, body.data.events);
    return reply.code(201).send({ success: true, data: wh });
  });

  fastify.get<{ Params: { orgId: string } }>('/orgs/:orgId/webhooks', async (req: any, reply) => {
    const whs = await WebhookService.listWebhooks(req.params.orgId);
    return reply.send({ success: true, data: whs });
  });

  fastify.delete<{ Params: { orgId: string; id: string } }>('/orgs/:orgId/webhooks/:id', async (req: any, reply) => {
    try {
      await WebhookService.deleteWebhook(req.params.id, req.params.orgId);
      return reply.code(204).send();
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.post<{ Params: { orgId: string; id: string } }>('/orgs/:orgId/webhooks/:id/rotate-secret', async (req: any, reply) => {
    try {
      const result = await WebhookService.rotateWebhookSecret(req.params.id, req.params.orgId);
      return reply.send({ success: true, data: result });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });
}

async function jwtGuard(request: any, reply: any): Promise<void> {
  const [, token] = (request.headers['authorization'] ?? '').split(' ');
  if (!token) return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Authentication required' } });
  try { request.jwtUser = verifyAccessToken(token); }
  catch { return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Invalid or expired token' } }); }
}
