/**
 * InsightSerenity API Gateway — AI Proxy Routes
 * ===============================================
 * These routes are the bridge between external clients and the Python AI engine.
 *
 * All routes under /v1/* are proxied after:
 *   1. API key validation (authenticate middleware)
 *   2. Rate limit + quota checks (rate-limit middleware)
 *   3. Scope enforcement (checked here per endpoint)
 *
 * Streaming (SSE) support:
 *   When request body contains `"stream": true`, we pipe the AI engine's
 *   `text/event-stream` response directly to the client using Node.js streams.
 *   This avoids buffering the entire generation in memory — critical for
 *   long outputs and real-time UX.
 *
 * Forwarded routes:
 *   POST /v1/completions         → completions:create scope
 *   POST /v1/chat/completions    → chat:create scope
 *   POST /v1/embeddings          → embeddings:create scope
 *   POST /v1/agents/run          → agents:run scope
 *   GET  /v1/models              → no scope required (metadata only)
 *   GET  /v1/models/:id          → no scope required
 */

import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { authenticateApiKey }          from '../../middleware/authenticate.js';
import { checkRateLimit }              from '../../middleware/rate-limit.js';
import * as ProxyService               from './proxy.service.js';
import { config }                      from '../../config/settings.js';
import { getLogger }                   from '../../observability/logger.js';

const log = getLogger('proxy-routes');

// ─────────────────────────────────────────────────────────────────────────────
// Plugin registration
// ─────────────────────────────────────────────────────────────────────────────

export async function proxyRoutes(fastify: FastifyInstance): Promise<void> {
  // Auth + rate limit run on every /v1/* route
  fastify.addHook('preHandler', authenticateApiKey);
  fastify.addHook('preHandler', checkRateLimit);

  // ── POST routes — generation endpoints ──────────────────────────────────
  for (const route of ['/v1/completions', '/v1/chat/completions', '/v1/embeddings', '/v1/agents/run']) {
    fastify.post(route, { config: { rawBody: true } }, makeProxyHandler(route));
  }

  // ── GET /v1/models — no auth scope required ────────────────────────────
  fastify.get('/v1/models', async (request, reply) => {
    return proxyGet(request, reply, '/v1/models');
  });

  fastify.get<{ Params: { '*': string } }>('/v1/models/*', async (request, reply) => {
    return proxyGet(request, reply, `/v1/models/${(request.params as any)['*']}`);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Proxy handler factory
// ─────────────────────────────────────────────────────────────────────────────

function makeProxyHandler(endpoint: string) {
  return async function handler(request: FastifyRequest, reply: FastifyReply): Promise<void> {
    const ctx = request.apiKeyContext!;

    // ── Scope check ──────────────────────────────────────────────────────
    const requiredScope = ProxyService.getRequiredScope(endpoint);
    if (requiredScope && !ProxyService.hasScope(ctx, requiredScope)) {
      return reply.code(403).send({
        success: false,
        error: {
          code:    'INSUFFICIENT_SCOPE',
          message: `API key does not have the '${requiredScope}' scope`,
        },
      }) as unknown as void;
    }

    const body        = request.body as Record<string, unknown>;
    const isStreaming = body?.stream === true;

    // ── Streaming path ───────────────────────────────────────────────────
    if (isStreaming) {
      return handleStreamingProxy(request, reply, endpoint, body, ctx);
    }

    // ── Non-streaming path ───────────────────────────────────────────────
    const start = Date.now();
    const result = await ProxyService.forwardRequest('POST', endpoint, body, ctx);

    // Forward all headers from the engine response
    for (const [key, value] of Object.entries(result.headers)) {
      if (!['transfer-encoding', 'connection'].includes(key.toLowerCase())) {
        reply.header(key, value);
      }
    }

    ProxyService.recordUsageAsync(ctx, endpoint, 'POST', result);

    // Expose token counts to the Prometheus response hook
    (request as any).metricsContext = {
      promptTokens:     result.promptTokens,
      completionTokens: result.completionTokens,
      orgPlan:          (ctx.org as any).plan ?? 'UNKNOWN',
    };

    return reply.code(result.statusCode).send(result.body) as unknown as void;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming proxy — pipes SSE from engine directly to client
// ─────────────────────────────────────────────────────────────────────────────

async function handleStreamingProxy(
  request:  FastifyRequest,
  reply:    FastifyReply,
  endpoint: string,
  body:     Record<string, unknown>,
  ctx:      ReturnType<typeof request.apiKeyContext extends infer T ? () => T : never> extends never
              ? NonNullable<typeof request.apiKeyContext>
              : NonNullable<typeof request.apiKeyContext>,
): Promise<void> {
  const url   = `${config.AI_ENGINE_URL}${endpoint}`;
  const start = Date.now();

  const engineResponse = await fetch(url, {
    method:  'POST',
    headers: ProxyService.buildEngineHeaders(ctx),
    body:    JSON.stringify(body),
    signal:  AbortSignal.timeout(300_000),  // 5-minute max for long streams
  });

  if (!engineResponse.ok || !engineResponse.body) {
    const errText = await engineResponse.text();
    log.warn({ status: engineResponse.status, endpoint }, 'Engine returned error for stream');
    return reply.code(engineResponse.status).send({
      success: false,
      error: { code: 'ENGINE_ERROR', message: errText },
    }) as unknown as void;
  }

  // Set SSE headers before streaming begins
  reply.raw.setHeader('Content-Type',  'text/event-stream');
  reply.raw.setHeader('Cache-Control', 'no-cache');
  reply.raw.setHeader('Connection',    'keep-alive');
  reply.raw.setHeader('X-Accel-Buffering', 'no');  // Disable nginx buffering
  reply.raw.writeHead(200);

  // Track tokens from SSE chunks (best-effort — final [DONE] has no usage)
  let totalTokens = 0;

  const reader = engineResponse.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      reply.raw.write(chunk);

      // Try to extract token usage from intermediate chunks
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data: ') && !line.includes('[DONE]')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data?.usage?.total_tokens) totalTokens = data.usage.total_tokens;
          } catch { /* Non-JSON data line */ }
        }
      }
    }
  } finally {
    reader.releaseLock();
    reply.raw.end();

    const latencyMs = Date.now() - start;
    ProxyService.recordUsageAsync(ctx, endpoint, 'POST', {
      statusCode:       200,
      promptTokens:     0,
      completionTokens: totalTokens,
      totalTokens,
      latencyMs,
      model:            'insightserenity-1',
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// GET proxy (models listing)
// ─────────────────────────────────────────────────────────────────────────────

async function proxyGet(request: FastifyRequest, reply: FastifyReply, path: string): Promise<void> {
  const ctx = request.apiKeyContext!;
  const url = `${config.AI_ENGINE_URL}${path}`;

  const response = await fetch(url, {
    method:  'GET',
    headers: ProxyService.buildEngineHeaders(ctx),
  });

  const body = await response.arrayBuffer();
  const contentType = response.headers.get('content-type') ?? 'application/json';

  reply.header('content-type', contentType);
  return reply.code(response.status).send(Buffer.from(body)) as unknown as void;
}
