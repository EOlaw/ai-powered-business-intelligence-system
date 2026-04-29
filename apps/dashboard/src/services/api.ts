/**
 * InsightSerenity Dashboard — API Service
 * ========================================
 * HTTP client for the API Gateway management endpoints (auth, keys, usage, billing).
 * Separate from the AI SDK — this talks to the management API, not the AI inference API.
 *
 * Auth: reads JWT access token from Zustand store / localStorage.
 * All requests attach Authorization: Bearer <accessToken>.
 */

const GATEWAY_URL = process.env['NEXT_PUBLIC_GATEWAY_URL'] ?? 'http://localhost:3000';

// ─────────────────────────────────────────────────────────────────────────────
// Base request helper
// ─────────────────────────────────────────────────────────────────────────────

async function request<T>(
  method:  string,
  path:    string,
  body?:   unknown,
  token?:  string,
): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${GATEWAY_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  const json = await res.json();

  if (!res.ok) {
    throw Object.assign(new Error(json?.error?.message ?? 'Request failed'), {
      status: res.status,
      code:   json?.error?.code,
    });
  }

  return (json as { data: T }).data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth API
// ─────────────────────────────────────────────────────────────────────────────

export interface AuthTokens {
  accessToken:  string;
  refreshToken: string;
  expiresIn:    number;
  userId:       string;
  orgId?:       string;
}

export const authApi = {
  register: (email: string, password: string, orgName: string): Promise<AuthTokens> =>
    request('POST', '/auth/register', { email, password, orgName }),

  login: (email: string, password: string): Promise<AuthTokens> =>
    request('POST', '/auth/login', { email, password }),

  refresh: (refreshToken: string): Promise<{ accessToken: string; expiresIn: number }> =>
    request('POST', '/auth/refresh', { refreshToken }),

  me: (token: string) =>
    request<any>('GET', '/auth/me', undefined, token),
};

// ─────────────────────────────────────────────────────────────────────────────
// API Keys API
// ─────────────────────────────────────────────────────────────────────────────

export const apiKeysApi = {
  list: (orgId: string, token: string) =>
    request<any[]>('GET', `/orgs/${orgId}/api-keys`, undefined, token),

  create: (orgId: string, body: { name: string; scopes: string[]; expiresAt?: string }, token: string) =>
    request<any>('POST', `/orgs/${orgId}/api-keys`, body, token),

  revoke: (orgId: string, keyId: string, token: string) =>
    request<any>('POST', `/orgs/${orgId}/api-keys/${keyId}/revoke`, undefined, token),

  rotate: (orgId: string, keyId: string, token: string) =>
    request<any>('POST', `/orgs/${orgId}/api-keys/${keyId}/rotate`, undefined, token),

  delete: (orgId: string, keyId: string, token: string) =>
    request<void>('DELETE', `/orgs/${orgId}/api-keys/${keyId}`, undefined, token),
};

// ─────────────────────────────────────────────────────────────────────────────
// Usage API
// ─────────────────────────────────────────────────────────────────────────────

export const usageApi = {
  overview: (orgId: string, token: string, start?: string, end?: string) => {
    const qs = new URLSearchParams();
    if (start) qs.set('start', start);
    if (end)   qs.set('end', end);
    return request<any>('GET', `/orgs/${orgId}/usage?${qs}`, undefined, token);
  },

  timeline: (orgId: string, token: string, start?: string, end?: string) => {
    const qs = new URLSearchParams();
    if (start) qs.set('start', start);
    if (end)   qs.set('end', end);
    return request<any[]>('GET', `/orgs/${orgId}/usage/timeline?${qs}`, undefined, token);
  },

  requests: (orgId: string, token: string, page = 1, limit = 50) =>
    request<any>('GET', `/orgs/${orgId}/usage/requests?page=${page}&limit=${limit}`, undefined, token),
};

// ─────────────────────────────────────────────────────────────────────────────
// Billing API
// ─────────────────────────────────────────────────────────────────────────────

export const billingApi = {
  subscription: (orgId: string, token: string) =>
    request<any>('GET', `/orgs/${orgId}/billing`, undefined, token),

  summary: (orgId: string, token: string, year?: number, month?: number) => {
    const qs = new URLSearchParams();
    if (year)  qs.set('year',  String(year));
    if (month) qs.set('month', String(month));
    return request<any>('GET', `/orgs/${orgId}/billing/summary?${qs}`, undefined, token);
  },

  changePlan: (orgId: string, plan: string, token: string) =>
    request<any>('POST', `/orgs/${orgId}/billing/plan`, { plan }, token),
};
