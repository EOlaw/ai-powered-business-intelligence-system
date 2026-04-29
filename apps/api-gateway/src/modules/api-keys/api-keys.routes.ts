/**
 * API Keys Routes
 * ---------------
 * All routes require a valid JWT. The org ID is sourced from the user's membership.
 *
 * POST   /orgs/:orgId/api-keys           — Create a new key
 * GET    /orgs/:orgId/api-keys           — List all keys
 * GET    /orgs/:orgId/api-keys/:id       — Get single key metadata
 * POST   /orgs/:orgId/api-keys/:id/revoke  — Revoke
 * POST   /orgs/:orgId/api-keys/:id/rotate  — Rotate (revoke + new)
 * DELETE /orgs/:orgId/api-keys/:id       — Hard delete
 */

import type { FastifyInstance } from 'fastify';
import { verifyAccessToken }    from '../auth/strategies/jwt.strategy.js';
import { getOrgOrFail }         from '../organizations/organizations.service.js';
import * as KeyService          from './api-keys.service.js';
import { CreateApiKeySchema }   from './api-keys.schema.js';

export async function apiKeyRoutes(fastify: FastifyInstance): Promise<void> {
  // All routes in this plugin require a valid JWT
  fastify.addHook('preHandler', jwtGuard);

  // ── POST /orgs/:orgId/api-keys ──────────────────────────────────────────
  fastify.post<{ Params: { orgId: string } }>(
    '/orgs/:orgId/api-keys',
    async (request, reply) => {
      await assertOrgAccess(request, reply);
      const orgId = request.params.orgId;

      const body = CreateApiKeySchema.safeParse(request.body);
      if (!body.success) {
        return reply.code(400).send({
          success: false,
          error: { code: 'VALIDATION_ERROR', message: 'Invalid input', details: body.error.flatten() },
        });
      }

      const result = await KeyService.createApiKey(orgId, body.data);

      // Raw key returned once — client must store it
      return reply.code(201).send({ success: true, data: result });
    },
  );

  // ── GET /orgs/:orgId/api-keys ────────────────────────────────────────────
  fastify.get<{ Params: { orgId: string } }>(
    '/orgs/:orgId/api-keys',
    async (request, reply) => {
      await assertOrgAccess(request, reply);
      const keys = await KeyService.listApiKeys(request.params.orgId);
      return reply.send({ success: true, data: keys });
    },
  );

  // ── GET /orgs/:orgId/api-keys/:id ───────────────────────────────────────
  fastify.get<{ Params: { orgId: string; id: string } }>(
    '/orgs/:orgId/api-keys/:id',
    async (request, reply) => {
      await assertOrgAccess(request, reply);
      try {
        const key = await KeyService.getApiKey(request.params.id, request.params.orgId);
        return reply.send({ success: true, data: key });
      } catch (err: any) {
        return reply.code(err.statusCode ?? 500).send({
          success: false,
          error: { code: err.code ?? 'ERROR', message: err.message },
        });
      }
    },
  );

  // ── POST /orgs/:orgId/api-keys/:id/revoke ───────────────────────────────
  fastify.post<{ Params: { orgId: string; id: string } }>(
    '/orgs/:orgId/api-keys/:id/revoke',
    async (request, reply) => {
      await assertOrgAccess(request, reply);
      try {
        await KeyService.revokeApiKey(request.params.id, request.params.orgId);
        return reply.send({ success: true, data: { message: 'API key revoked' } });
      } catch (err: any) {
        return reply.code(err.statusCode ?? 500).send({
          success: false,
          error: { code: err.code ?? 'ERROR', message: err.message },
        });
      }
    },
  );

  // ── POST /orgs/:orgId/api-keys/:id/rotate ───────────────────────────────
  fastify.post<{ Params: { orgId: string; id: string } }>(
    '/orgs/:orgId/api-keys/:id/rotate',
    async (request, reply) => {
      await assertOrgAccess(request, reply);
      try {
        const result = await KeyService.rotateApiKey(request.params.id, request.params.orgId);
        return reply.code(201).send({ success: true, data: result });
      } catch (err: any) {
        return reply.code(err.statusCode ?? 500).send({
          success: false,
          error: { code: err.code ?? 'ERROR', message: err.message },
        });
      }
    },
  );

  // ── DELETE /orgs/:orgId/api-keys/:id ───────────────────────────────────
  fastify.delete<{ Params: { orgId: string; id: string } }>(
    '/orgs/:orgId/api-keys/:id',
    async (request, reply) => {
      await assertOrgAccess(request, reply);
      try {
        await KeyService.deleteApiKey(request.params.id, request.params.orgId);
        return reply.code(204).send();
      } catch (err: any) {
        return reply.code(err.statusCode ?? 500).send({
          success: false,
          error: { code: err.code ?? 'ERROR', message: err.message },
        });
      }
    },
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Guards
// ─────────────────────────────────────────────────────────────────────────────

async function jwtGuard(request: any, reply: any): Promise<void> {
  const [, token] = (request.headers['authorization'] ?? '').split(' ');
  if (!token) {
    return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Authentication required' } });
  }
  try {
    request.jwtUser = verifyAccessToken(token);
  } catch {
    return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Invalid or expired token' } });
  }
}

async function assertOrgAccess(request: any, reply: any): Promise<void> {
  const { orgId } = request.params as { orgId: string };
  const userId    = request.jwtUser?.sub as string;

  if (!userId) return;

  const { prisma } = await import('../../db/client.js');
  const membership = await prisma.orgMembership.findUnique({
    where: { userId_orgId: { userId, orgId } },
  });

  if (!membership) {
    reply.code(403).send({
      success: false,
      error: { code: 'FORBIDDEN', message: 'You do not have access to this organisation' },
    });
    throw new Error('Access denied');
  }
}
