/**
 * InsightSerenity API Gateway — Fastify Application Factory
 * ==========================================================
 * Builds and configures the Fastify instance with all plugins and routes.
 * Kept separate from server.ts so the app can be imported in tests without
 * binding to a port.
 *
 * Plugin registration order matters in Fastify:
 *   1. Core infrastructure plugins (helmet, cors, content-parser)
 *   2. Observability (request logging)
 *   3. Feature route plugins
 *   4. Error handler
 *
 * Route prefixes:
 *   /health*           — liveness + readiness probes (no auth)
 *   /auth/*            — registration, login, token refresh
 *   /users/*           — self-service user management (JWT)
 *   /orgs/*            — org + member + key management (JWT)
 *   /admin/*           — platform admin (JWT + ADMIN role)
 *   /v1/*              — AI engine proxy (INSIGHTSERENITY_API_KEY)
 */

import Fastify from 'fastify';
import cors    from '@fastify/cors';
import helmet  from '@fastify/helmet';

import { corsOrigins, config } from './config/settings.js';
import { logger }              from './observability/logger.js';

import { healthRoutes }       from './modules/health/health.routes.js';
import { authRoutes }         from './modules/auth/auth.routes.js';
import { userRoutes }         from './modules/users/users.routes.js';
import { organizationRoutes } from './modules/organizations/organizations.routes.js';
import { apiKeyRoutes }       from './modules/api-keys/api-keys.routes.js';
import { billingRoutes }      from './modules/billing/billing.routes.js';
import { usageRoutes }        from './modules/usage/usage.routes.js';
import { webhookRoutes }      from './modules/webhooks/webhooks.routes.js';
import { adminRoutes }        from './modules/admin/admin.routes.js';
import { proxyRoutes }        from './modules/proxy/proxy.routes.js';
import { metricsOnRequest, metricsOnResponse } from './middleware/prometheus.js';
import { getMetrics, PROMETHEUS_CONTENT_TYPE } from './observability/metrics.js';

// ─────────────────────────────────────────────────────────────────────────────
// Build
// ─────────────────────────────────────────────────────────────────────────────

export async function buildApp() {
  const app = Fastify({
    loggerInstance: logger,
    trustProxy:     true,                  // Required when behind a load balancer
    requestTimeout: 150_000,               // 2.5 min — long enough for streamed AI responses
    bodyLimit:      1_048_576,             // 1 MB request body limit
    ajv: {
      customOptions: {
        strict:         'log',
        coerceTypes:    true,
        removeAdditional: true,
      },
    },
  });

  // ── Core security headers ─────────────────────────────────────────────────
  await app.register(helmet, {
    contentSecurityPolicy: false,  // We're an API, not serving HTML
    crossOriginEmbedderPolicy: false,
  });

  // ── CORS ──────────────────────────────────────────────────────────────────
  await app.register(cors, {
    origin:      corsOrigins,
    methods:     ['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: [
      'Authorization',
      'Content-Type',
      'X-Org-Id',
      'X-Request-Id',
    ],
    credentials: corsOrigins[0] !== '*',
  });

  // ── Request ID + Prometheus request timing ───────────────────────────────
  app.addHook('onRequest', async (request) => {
    const reqId = request.headers['x-request-id'] as string | undefined
      ?? `gw-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    (request as any).reqId = reqId;
    await metricsOnRequest(request);
  });

  app.addHook('onResponse', metricsOnResponse);

  // ── Prometheus /metrics endpoint (internal scrape target, no auth) ────────
  app.get('/metrics', async (_request, reply) => {
    const content = await getMetrics();
    return reply
      .header('Content-Type', PROMETHEUS_CONTENT_TYPE)
      .send(content);
  });

  // ── Route plugins ─────────────────────────────────────────────────────────
  await app.register(healthRoutes);
  await app.register(authRoutes);
  await app.register(userRoutes);
  await app.register(organizationRoutes);
  await app.register(apiKeyRoutes);
  await app.register(billingRoutes);
  await app.register(usageRoutes);
  await app.register(webhookRoutes);
  await app.register(adminRoutes);
  await app.register(proxyRoutes);

  // ── Global error handler ──────────────────────────────────────────────────
  app.setErrorHandler(async (error, request, reply) => {
    const err = error instanceof Error ? error : new Error(String(error));
    const maybeStatus = (error as { statusCode?: unknown }).statusCode;
    const statusCode = typeof maybeStatus === 'number' ? maybeStatus : 500;

    if (statusCode >= 500) {
      app.log.error({ err, reqId: (request as any).reqId }, 'Unhandled error');
    }

    return reply.code(statusCode).send({
      success: false,
      error: {
        code:    'INTERNAL_ERROR',
        message: config.NODE_ENV === 'production' ? 'An unexpected error occurred' : err.message,
      },
    });
  });

  // ── 404 handler ───────────────────────────────────────────────────────────
  app.setNotFoundHandler(async (_request, reply) => {
    return reply.code(404).send({
      success: false,
      error: { code: 'NOT_FOUND', message: 'Route not found' },
    });
  });

  return app;
}
