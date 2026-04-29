/**
 * InsightSerenity API Gateway — Prometheus Request Metrics Middleware
 * ====================================================================
 * Fastify hook that records per-request metrics after every response.
 *
 * Records:
 *   - gatewayRequestsTotal   (counter)   — all requests by endpoint/method/status
 *   - gatewayLatency         (histogram) — wall-clock request duration
 *   - gatewayActiveRequests  (gauge)     — in-flight request count
 *   - tokenProxiedTotal      (counter)   — prompt+completion tokens, set by proxy route
 *
 * Token counts are populated by the proxy route handler, which reads them from
 * the AI engine response and attaches them to the Fastify request object via:
 *   request.metricsContext = { promptTokens: N, completionTokens: N, orgPlan: 'FREE' }
 *
 * Path normalisation prevents cardinality explosion:
 *   /orgs/clxyz123/api-keys/clkeyabc → /orgs/{orgId}/api-keys/{keyId}
 */

import type { FastifyRequest, FastifyReply } from 'fastify';
import {
  gatewayRequestsTotal,
  gatewayLatency,
  gatewayActiveRequests,
  tokenProxiedTotal,
} from '../observability/metrics.js';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface MetricsContext {
  promptTokens:     number;
  completionTokens: number;
  orgPlan:          string;
}

declare module 'fastify' {
  interface FastifyRequest {
    metricsContext?: MetricsContext;
    _metricsStart?: [number, number];  // process.hrtime() tuple
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hooks
// ─────────────────────────────────────────────────────────────────────────────

/** onRequest hook: record start time and increment active counter. */
export async function metricsOnRequest(
  request: FastifyRequest,
): Promise<void> {
  const path = request.url;

  // Skip the /metrics endpoint itself to avoid recursive counting
  if (path === '/metrics' || path.startsWith('/health')) return;

  request._metricsStart = process.hrtime();
  gatewayActiveRequests.inc();
}

/** onResponse hook: record latency and decrement active counter. */
export async function metricsOnResponse(
  request: FastifyRequest,
  reply:   FastifyReply,
): Promise<void> {
  const path = request.url;
  if (path === '/metrics' || path.startsWith('/health')) return;
  if (!request._metricsStart) return;

  gatewayActiveRequests.dec();

  const [sec, ns] = process.hrtime(request._metricsStart);
  const latencyS  = sec + ns / 1e9;
  const endpoint  = normalisePath(request.routerPath ?? path);
  const method    = request.method;
  const status    = String(reply.statusCode);

  gatewayRequestsTotal.labels(endpoint, method, status).inc();
  gatewayLatency.labels(endpoint, method).observe(latencyS);

  // Token counts from proxy route (only set for /v1/* routes)
  const ctx = request.metricsContext;
  if (ctx) {
    const plan = ctx.orgPlan || 'UNKNOWN';
    if (ctx.promptTokens > 0) {
      tokenProxiedTotal.labels('prompt', plan).inc(ctx.promptTokens);
    }
    if (ctx.completionTokens > 0) {
      tokenProxiedTotal.labels('completion', plan).inc(ctx.completionTokens);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Path normalisation — prevents label cardinality explosion
// ─────────────────────────────────────────────────────────────────────────────

function normalisePath(path: string): string {
  return path
    // CUID / UUID patterns in path segments
    .replace(/\/c[a-z0-9]{20,}(?=\/|$)/g, '/{id}')
    .replace(/\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=\/|$)/gi, '/{uuid}')
    // Model IDs  /v1/models/insightserenity-1:v0.1.0
    .replace(/\/v1\/models\/[^/]+/g, '/v1/models/{model_id}')
    // Org-scoped routes  /orgs/:orgId/...
    .replace(/\/orgs\/[^/]+/g, '/orgs/{orgId}');
}
