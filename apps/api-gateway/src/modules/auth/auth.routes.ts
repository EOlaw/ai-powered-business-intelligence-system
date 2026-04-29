/**
 * InsightSerenity API Gateway — Auth Routes
 * ==========================================
 * POST /auth/register   — Create account + org + FREE subscription
 * POST /auth/login      — Email/password → access + refresh tokens
 * POST /auth/refresh    — Refresh token → new access token
 * POST /auth/logout     — Revoke refresh token (requires valid access token)
 * GET  /auth/me         — Get current user profile (requires JWT)
 */

import type { FastifyInstance } from 'fastify';
import { z }                    from 'zod';
import { verifyAccessToken }    from './strategies/jwt.strategy.js';
import * as AuthService         from './auth.service.js';
import { RegisterSchema, LoginSchema, RefreshSchema } from './auth.schema.js';
import { getLogger } from '../../observability/logger.js';
import { prisma }    from '../../db/client.js';

const log = getLogger('auth-routes');

export async function authRoutes(fastify: FastifyInstance): Promise<void> {

  // ── POST /auth/register ──────────────────────────────────────────────────
  fastify.post('/auth/register', async (request, reply) => {
    const body = RegisterSchema.safeParse(request.body);
    if (!body.success) {
      return reply.code(400).send({
        success: false,
        error: { code: 'VALIDATION_ERROR', message: 'Invalid input', details: body.error.flatten() },
      });
    }

    const { email, password, orgName } = body.data;

    try {
      const result = await AuthService.register(email, password, orgName);
      return reply.code(201).send({
        success: true,
        data: {
          accessToken:  result.tokens.accessToken,
          refreshToken: result.tokens.refreshToken,
          expiresIn:    result.tokens.expiresIn,
          userId:       result.userId,
          orgId:        result.orgId,
        },
      });
    } catch (err: any) {
      const status = err.statusCode ?? 500;
      return reply.code(status).send({
        success: false,
        error: { code: err.code ?? 'REGISTRATION_FAILED', message: err.message },
      });
    }
  });

  // ── POST /auth/login ────────────────────────────────────────────────────
  fastify.post('/auth/login', async (request, reply) => {
    const body = LoginSchema.safeParse(request.body);
    if (!body.success) {
      return reply.code(400).send({
        success: false,
        error: { code: 'VALIDATION_ERROR', message: 'Invalid input' },
      });
    }

    const { email, password } = body.data;

    try {
      const result = await AuthService.login(email, password);
      return reply.send({
        success: true,
        data: {
          accessToken:  result.tokens.accessToken,
          refreshToken: result.tokens.refreshToken,
          expiresIn:    result.tokens.expiresIn,
          userId:       result.userId,
        },
      });
    } catch (err: any) {
      const status = err.statusCode ?? 500;
      return reply.code(status).send({
        success: false,
        error: { code: err.code ?? 'LOGIN_FAILED', message: err.message },
      });
    }
  });

  // ── POST /auth/refresh ───────────────────────────────────────────────────
  fastify.post('/auth/refresh', async (request, reply) => {
    const body = RefreshSchema.safeParse(request.body);
    if (!body.success) {
      return reply.code(400).send({
        success: false,
        error: { code: 'VALIDATION_ERROR', message: 'refreshToken is required' },
      });
    }

    try {
      const result = await AuthService.refresh(body.data.refreshToken);
      return reply.send({
        success:  true,
        data: { accessToken: result.accessToken, expiresIn: result.expiresIn },
      });
    } catch (err: any) {
      return reply.code(err.statusCode ?? 401).send({
        success: false,
        error: { code: err.code ?? 'REFRESH_FAILED', message: err.message },
      });
    }
  });

  // ── POST /auth/logout ────────────────────────────────────────────────────
  fastify.post('/auth/logout', {
    preHandler: [jwtGuard],
  }, async (request, reply) => {
    const user = request.jwtUser!;
    if (user.jti) {
      await AuthService.logout(user.jti);
    }
    return reply.send({ success: true, data: { message: 'Logged out successfully' } });
  });

  // ── GET /auth/me ─────────────────────────────────────────────────────────
  fastify.get('/auth/me', {
    preHandler: [jwtGuard],
  }, async (request, reply) => {
    const user = request.jwtUser!;

    const dbUser = await prisma.user.findUnique({
      where:  { id: user.sub },
      select: {
        id:          true,
        email:       true,
        role:        true,
        createdAt:   true,
        memberships: {
          select: {
            role: true,
            org: { select: { id: true, name: true, slug: true, plan: true } },
          },
        },
      },
    });

    if (!dbUser) {
      return reply.code(404).send({
        success: false,
        error: { code: 'NOT_FOUND', message: 'User not found' },
      });
    }

    return reply.send({ success: true, data: dbUser });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// JWT guard (shared preHandler for protected routes in this module)
// ─────────────────────────────────────────────────────────────────────────────

async function jwtGuard(request: any, reply: any): Promise<void> {
  const authHeader = request.headers['authorization'] as string | undefined;
  if (!authHeader) {
    return reply.code(401).send({
      success: false,
      error: { code: 'UNAUTHORIZED', message: 'Missing Authorization header' },
    });
  }

  const [, token] = authHeader.split(' ');
  if (!token) {
    return reply.code(401).send({
      success: false,
      error: { code: 'UNAUTHORIZED', message: 'Invalid Authorization format' },
    });
  }

  try {
    request.jwtUser = verifyAccessToken(token);
  } catch {
    return reply.code(401).send({
      success: false,
      error: { code: 'UNAUTHORIZED', message: 'Invalid or expired token' },
    });
  }
}
