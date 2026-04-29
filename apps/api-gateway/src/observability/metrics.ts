/**
 * InsightSerenity API Gateway — Prometheus Metrics
 * ==================================================
 * All metrics are defined once here and exported for use in middleware
 * and route handlers. prom-client auto-registers them in the default registry.
 *
 * Metric naming:
 *   insightserenity_gateway_<name>_<unit>
 *
 * Scraped by Prometheus at GET /metrics (no auth — internal network only).
 *
 * Labels:
 *   endpoint    — normalised request path (/v1/chat/completions, /auth/login, …)
 *   method      — HTTP method (POST, GET, …)
 *   status_code — response HTTP status code string ("200", "429", "500", …)
 *   org_plan    — organisation plan tier (FREE, STARTER, PRO, ENTERPRISE)
 */

import {
  Counter, Histogram, Gauge,
  collectDefaultMetrics, register,
} from 'prom-client';

// Collect default Node.js runtime metrics (event loop lag, heap, GC, etc.)
collectDefaultMetrics({ prefix: 'insightserenity_gateway_nodejs_' });

// ─────────────────────────────────────────────────────────────────────────────
// HTTP layer
// ─────────────────────────────────────────────────────────────────────────────

export const gatewayRequestsTotal = new Counter({
  name:       'insightserenity_gateway_requests_total',
  help:       'Total HTTP requests handled by the API gateway',
  labelNames: ['endpoint', 'method', 'status_code'] as const,
});

export const gatewayLatency = new Histogram({
  name:       'insightserenity_gateway_request_duration_seconds',
  help:       'End-to-end gateway request latency (auth + proxy + response)',
  labelNames: ['endpoint', 'method'] as const,
  buckets:    [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
});

export const gatewayActiveRequests = new Gauge({
  name: 'insightserenity_gateway_active_requests',
  help: 'Number of requests currently being proxied to the AI engine',
});

// ─────────────────────────────────────────────────────────────────────────────
// Auth & rate limiting
// ─────────────────────────────────────────────────────────────────────────────

export const rateLimitedTotal = new Counter({
  name:       'insightserenity_gateway_rate_limited_total',
  help:       'Requests rejected by the rate limiter or daily quota',
  labelNames: ['reason', 'org_plan'] as const,
  // reason: "rpm_exceeded" | "token_quota_exceeded"
});

export const authFailuresTotal = new Counter({
  name:       'insightserenity_gateway_auth_failures_total',
  help:       'API key authentication failures',
  labelNames: ['reason'] as const,
  // reason: "missing_header" | "invalid_key" | "revoked" | "expired"
});

// ─────────────────────────────────────────────────────────────────────────────
// Token usage
// ─────────────────────────────────────────────────────────────────────────────

export const tokenProxiedTotal = new Counter({
  name:       'insightserenity_gateway_tokens_proxied_total',
  help:       'Total AI tokens proxied through the gateway, split by type',
  labelNames: ['type', 'org_plan'] as const,
  // type: "prompt" | "completion"
});

// ─────────────────────────────────────────────────────────────────────────────
// Prometheus registry export (for the /metrics endpoint)
// ─────────────────────────────────────────────────────────────────────────────

export { register };

/**
 * Generate the Prometheus text exposition format for all registered metrics.
 * Used by the GET /metrics route handler.
 */
export async function getMetrics(): Promise<string> {
  return register.metrics();
}

export const PROMETHEUS_CONTENT_TYPE = register.contentType;
