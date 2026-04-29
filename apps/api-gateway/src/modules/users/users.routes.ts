/**
 * Users Routes
 * ------------
 * GET    /users/me                — Profile + org memberships
 * POST   /users/me/password       — Change password
 * DELETE /users/me                — Delete account
 */

import type { FastifyInstance } from 'fastify';
import { z }                    from 'zod';
import { verifyAccessToken }    from '../auth/strategies/jwt.strategy.js';
import * as UserService         from './users.service.js';

const ChangePasswordSchema = z.object({
  currentPassword: z.string().min(1),
  newPassword:     z.string().min(8).max(128),
});

export async function userRoutes(fastify: FastifyInstance): Promise<void> {
  fastify.addHook('preHandler', jwtGuard);

  fastify.get('/users/me', async (req: any, reply) => {
    try {
      const profile = await UserService.getProfile(req.jwtUser.sub);
      return reply.send({ success: true, data: profile });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.post('/users/me/password', async (req: any, reply) => {
    const body = ChangePasswordSchema.safeParse(req.body);
    if (!body.success) {
      return reply.code(400).send({ success: false, error: { code: 'VALIDATION_ERROR', message: 'Invalid input' } });
    }
    try {
      await UserService.changePassword(req.jwtUser.sub, body.data.currentPassword, body.data.newPassword);
      return reply.send({ success: true, data: { message: 'Password changed. All sessions have been revoked.' } });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 500).send({ success: false, error: { code: err.code, message: err.message } });
    }
  });

  fastify.delete('/users/me', async (req: any, reply) => {
    await UserService.deleteAccount(req.jwtUser.sub);
    return reply.code(204).send();
  });
}

async function jwtGuard(request: any, reply: any): Promise<void> {
  const [, token] = (request.headers['authorization'] ?? '').split(' ');
  if (!token) return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Authentication required' } });
  try { request.jwtUser = verifyAccessToken(token); }
  catch { return reply.code(401).send({ success: false, error: { code: 'UNAUTHORIZED', message: 'Invalid or expired token' } }); }
}
