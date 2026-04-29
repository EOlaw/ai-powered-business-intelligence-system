/**
 * Organizations Routes
 * ---------------------
 * GET    /orgs                          — List user's orgs
 * POST   /orgs                          — Create new org
 * GET    /orgs/:orgId                   — Get org details
 * PATCH  /orgs/:orgId                   — Update org name
 * GET    /orgs/:orgId/members           — List members
 * POST   /orgs/:orgId/members           — Add member by email
 * DELETE /orgs/:orgId/members/:userId   — Remove member
 * PATCH  /orgs/:orgId/members/:userId   — Update member role
 */

import type { FastifyInstance } from 'fastify';
import { verifyAccessToken }    from '../auth/strategies/jwt.strategy.js';
import * as OrgService          from './organizations.service.js';
import { CreateOrgSchema, UpdateOrgSchema, InviteMemberSchema, UpdateMemberRoleSchema } from './organizations.schema.js';

export async function organizationRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.addHook('preHandler', jwtGuard);

  fastify.get('/orgs', async (req: any, reply) => {
    const orgs = await OrgService.listUserOrgs(req.jwtUser.sub);
    return reply.send({ success: true, data: orgs });
  });

  fastify.post('/orgs', async (req: any, reply) => {
    const body = CreateOrgSchema.safeParse(req.body);
    if (!body.success) return validationError(reply, body.error);
    const org = await OrgService.createOrg(req.jwtUser.sub, body.data.name);
    return reply.code(201).send({ success: true, data: org });
  });

  fastify.get<{ Params: { orgId: string } }>('/orgs/:orgId', async (req: any, reply) => {
    try {
      const { org, role } = await OrgService.getOrgOrFail(req.params.orgId, req.jwtUser.sub);
      return reply.send({ success: true, data: { ...org, myRole: role } });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.patch<{ Params: { orgId: string } }>('/orgs/:orgId', async (req: any, reply) => {
    const body = UpdateOrgSchema.safeParse(req.body);
    if (!body.success) return validationError(reply, body.error);
    try {
      const org = await OrgService.updateOrg(req.params.orgId, req.jwtUser.sub, body.data.name!);
      return reply.send({ success: true, data: org });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.get<{ Params: { orgId: string } }>('/orgs/:orgId/members', async (req: any, reply) => {
    const members = await OrgService.listMembers(req.params.orgId);
    return reply.send({ success: true, data: members });
  });

  fastify.post<{ Params: { orgId: string } }>('/orgs/:orgId/members', async (req: any, reply) => {
    const body = InviteMemberSchema.safeParse(req.body);
    if (!body.success) return validationError(reply, body.error);
    try {
      const m = await OrgService.addMember(req.params.orgId, req.jwtUser.sub, body.data.email, body.data.role);
      return reply.code(201).send({ success: true, data: m });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.delete<{ Params: { orgId: string; userId: string } }>('/orgs/:orgId/members/:userId', async (req: any, reply) => {
    try {
      await OrgService.removeMember(req.params.orgId, req.jwtUser.sub, req.params.userId);
      return reply.code(204).send();
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.patch<{ Params: { orgId: string; userId: string } }>('/orgs/:orgId/members/:userId', async (req: any, reply) => {
    const body = UpdateMemberRoleSchema.safeParse(req.body);
    if (!body.success) return validationError(reply, body.error);
    try {
      await OrgService.updateMemberRole(req.params.orgId, req.jwtUser.sub, req.params.userId, body.data.role);
      return reply.send({ success: true, data: { message: 'Role updated' } });
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

function validationError(reply: any, error: any) {
  return reply.code(400).send({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Invalid input', details: error.flatten() } });
}
