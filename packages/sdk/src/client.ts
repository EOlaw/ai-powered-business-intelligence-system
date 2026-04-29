/**
 * InsightSerenity SDK — HTTP Client Core
 * ========================================
 * Base HTTP client used by all resource classes.
 *
 * Features:
 *   - Automatic retry with exponential backoff on 429 and 5xx
 *   - Request timeout via AbortSignal
 *   - Structured error handling with typed APIError
 *   - Streaming support (returns ReadableStream for SSE endpoints)
 *   - Zero runtime dependencies — uses native fetch only
 */

import type { InsightSerenityClientOptions, APIErrorBody } from './types/index.js';

const DEFAULT_BASE_URL   = 'https://api.insightserenity.com';
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_RETRIES    = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Error class
// ─────────────────────────────────────────────────────────────────────────────

export class InsightSerenityError extends Error {
  readonly status:  number;
  readonly code:    string;
  readonly headers: Record<string, string>;

  constructor(status: number, body: APIErrorBody, headers: Record<string, string>) {
    super(body.error.message);
    this.name    = 'InsightSerenityError';
    this.status  = status;
    this.code    = body.error.code;
    this.headers = headers;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP client
// ─────────────────────────────────────────────────────────────────────────────

export class HttpClient {
  protected readonly baseURL:     string;
  protected readonly apiKey:      string;
  protected readonly timeout:     number;
  protected readonly maxRetries:  number;

  constructor(options: InsightSerenityClientOptions) {
    this.baseURL    = (options.baseURL ?? DEFAULT_BASE_URL).replace(/\/$/, '');
    this.apiKey     = options.apiKey;
    this.timeout    = options.timeout    ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = options.maxRetries ?? DEFAULT_RETRIES;
  }

  // ── JSON request ──────────────────────────────────────────────────────────

  async request<T>(
    method: string,
    path:   string,
    body?:  unknown,
  ): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      if (attempt > 0) {
        await sleep(Math.min(1000 * 2 ** (attempt - 1), 8000));
      }

      try {
        const response = await fetch(`${this.baseURL}${path}`, {
          method,
          headers: this.buildHeaders(),
          body:    body ? JSON.stringify(body) : undefined,
          signal:  AbortSignal.timeout(this.timeout),
        });

        // Retry on 429 or 5xx
        if (response.status === 429 || response.status >= 500) {
          const retryAfter = response.headers.get('retry-after');
          if (retryAfter && attempt < this.maxRetries) {
            await sleep(parseInt(retryAfter, 10) * 1000);
          }
          lastError = new Error(`HTTP ${response.status}`);
          continue;
        }

        if (!response.ok) {
          const errBody = await response.json() as APIErrorBody;
          const headers: Record<string, string> = {};
          response.headers.forEach((v, k) => { headers[k] = v; });
          throw new InsightSerenityError(response.status, errBody, headers);
        }

        return response.json() as Promise<T>;
      } catch (err) {
        if (err instanceof InsightSerenityError) throw err;
        lastError = err as Error;
      }
    }

    throw lastError ?? new Error('Request failed after retries');
  }

  // ── Streaming request (SSE) ───────────────────────────────────────────────

  async *stream<T extends object>(
    method: string,
    path:   string,
    body?:  unknown,
  ): AsyncGenerator<T, void, unknown> {
    const response = await fetch(`${this.baseURL}${path}`, {
      method,
      headers: this.buildHeaders(),
      body:    body ? JSON.stringify(body) : undefined,
      signal:  AbortSignal.timeout(300_000),   // 5-min timeout for streams
    });

    if (!response.ok || !response.body) {
      const errBody = await response.json() as APIErrorBody;
      const headers: Record<string, string> = {};
      response.headers.forEach((v, k) => { headers[k] = v; });
      throw new InsightSerenityError(response.status, errBody, headers);
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          try { yield JSON.parse(data) as T; } catch { /* Malformed chunk */ }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private buildHeaders(): Record<string, string> {
    return {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
      'User-Agent':    'insightserenity-sdk/1.0.0',
    };
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
