/**
 * InsightSerenity API Gateway — AI Engine Proxy Service
 * ======================================================
 * Forwards authenticated requests from external clients to the Python AI engine.
 *
 * What this layer does on EVERY proxied request:
 *   1. Verified API key is already validated by authenticate middleware
 *   2. Rate limits already checked by rate-limit middleware
 *   3. Build the forwarding URL: AI_ENGINE_URL + request path
 *   4. Inject internal service headers:
 *        Authorization: Bearer <SERVING_INTERNAL_API_SECRET>
 *        X-Org-Id:     <org_id>
 *        X-Key-Id:     <api_key_id>
 *        X-Scopes:     <scope1,scope2,...>
 *   5. Stream or buffer the AI engine response back to the client
 *   6. After response: enqueue a UsageRecord asynchronously
 *   7. After response: update daily token quota in Redis
 *
 * Streaming:
 *   If the request body contains `"stream": true`, the AI engine returns
 *   Content-Type: text/event-stream. We pipe it through directly so the
 *   SSE connection reaches the client with zero buffering overhead.
 *
 * Scope enforcement:
 *   Each endpoint requires a specific scope.
 *   The scope map is defined here and checked before forwarding.
 */

import { config }             from '../../config/settings.js';
import { getLogger }          from '../../observability/logger.js';
import { enqueueUsage }       from '../../queue/worker.js';
import { incrementTokenQuota } from '../../middleware/rate-limit.js';
import type { ApiKeyContext }  from '../../types/index.js';

const log = getLogger('proxy-service');

// ─────────────────────────────────────────────────────────────────────────────
// Scope requirements per endpoint prefix
// ─────────────────────────────────────────────────────────────────────────────

const SCOPE_MAP: Record<string, string> = {
  '/v1/completions':         'completions:create',
  '/v1/chat/completions':    'chat:create',
  '/v1/embeddings':          'embeddings:create',
  '/v1/agents/run':          'agents:run',
};

export function getRequiredScope(path: string): string | null {
  for (const [prefix, scope] of Object.entries(SCOPE_MAP)) {
    if (path.startsWith(prefix)) return scope;
  }
  return null;
}

export function hasScope(ctx: ApiKeyContext, scope: string): boolean {
  return ctx.scopes.includes(scope) || ctx.scopes.includes('admin:all');
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward request to AI engine
// ─────────────────────────────────────────────────────────────────────────────

export interface ProxyResult {
  statusCode:       number;
  headers:          Record<string, string>;
  body:             Buffer | null;
  isStream:         boolean;
  promptTokens:     number;
  completionTokens: number;
  totalTokens:      number;
  latencyMs:        number;
  model:            string;
}

/**
 * Forward a request to the AI engine and return the full response.
 * Used for non-streaming responses.
 */
export async function forwardRequest(
  method:  string,
  path:    string,
  body:    unknown,
  ctx:     ApiKeyContext,
): Promise<ProxyResult> {
  const url   = `${config.AI_ENGINE_URL}${path}`;
  const start = Date.now();

  const response = await fetch(url, {
    method,
    headers: buildEngineHeaders(ctx),
    ...(body ? { body: JSON.stringify(body) } : {}),
    signal:  AbortSignal.timeout(120_000),  // 2-minute max for long generations
  });

  const latencyMs = Date.now() - start;
  const respBody  = await response.arrayBuffer();
  const bodyBuf   = Buffer.from(respBody);

  // Parse token usage from the response JSON (best-effort)
  let promptTokens = 0, completionTokens = 0, totalTokens = 0, model = 'insightserenity-1';
  try {
    const parsed = JSON.parse(bodyBuf.toString());
    promptTokens     = parsed?.usage?.prompt_tokens     ?? 0;
    completionTokens = parsed?.usage?.completion_tokens ?? 0;
    totalTokens      = parsed?.usage?.total_tokens      ?? (promptTokens + completionTokens);
    model            = parsed?.model ?? model;
  } catch {
    // Not JSON or no usage field — fine for error responses
  }

  const headers: Record<string, string> = {};
  response.headers.forEach((value, key) => { headers[key] = value; });

  return {
    statusCode: response.status,
    headers,
    body:       bodyBuf,
    isStream:   false,
    promptTokens,
    completionTokens,
    totalTokens,
    latencyMs,
    model,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Record usage after response
// ─────────────────────────────────────────────────────────────────────────────

export function recordUsageAsync(
  ctx:        ApiKeyContext,
  endpoint:   string,
  method:     string,
  result:     Pick<ProxyResult, 'statusCode' | 'promptTokens' | 'completionTokens' | 'totalTokens' | 'latencyMs' | 'model'>,
): void {
  enqueueUsage({
    orgId:            ctx.org.id,
    apiKeyId:         ctx.apiKey.id,
    endpoint,
    method,
    promptTokens:     result.promptTokens,
    completionTokens: result.completionTokens,
    totalTokens:      result.totalTokens,
    latencyMs:        result.latencyMs,
    statusCode:       result.statusCode,
    model:            result.model,
  });

  if (result.totalTokens > 0) {
    incrementTokenQuota(ctx.org.id, result.totalTokens).catch((err: Error) => {
      log.warn({ err }, 'Failed to increment token quota');
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Build headers for the AI engine
// ─────────────────────────────────────────────────────────────────────────────

export function buildEngineHeaders(ctx: ApiKeyContext): Record<string, string> {
  return {
    'Content-Type':   'application/json',
    'Authorization':  `Bearer ${config.SERVING_INTERNAL_API_SECRET}`,
    'X-Org-Id':       ctx.org.id,
    'X-Key-Id':       ctx.apiKey.id,
    'X-Scopes':       ctx.scopes.join(','),
  };
}
