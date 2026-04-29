/**
 * InsightSerenity API Gateway — Shared TypeScript Types
 * ======================================================
 * Central type definitions shared across all modules.
 * Augments Fastify's request/reply objects with our custom fields.
 */

import type { FastifyRequest } from 'fastify';
import type { ApiKey, Organization, User, OrgRole } from '@prisma/client';

// ─────────────────────────────────────────────────────────────────────────────
// Auth context — attached to request by authenticate middleware
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Set on `request.apiKeyContext` after successful INSIGHTSERENITY_API_KEY validation.
 * Available in all proxy and AI-engine-facing routes.
 */
export interface ApiKeyContext {
  apiKey:       ApiKey;
  org:          Organization;
  scopes:       string[];
}

/**
 * Set on `request.jwtUser` after successful JWT validation.
 * Available in all user-facing dashboard/management routes.
 */
export interface JwtUser {
  sub:   string;   // User ID (cuid)
  email: string;
  role:  'USER' | 'ADMIN';
  jti:   string;   // JWT ID — checked against sessions table for revocation
}

// ─────────────────────────────────────────────────────────────────────────────
// Fastify module augmentation
// ─────────────────────────────────────────────────────────────────────────────

declare module 'fastify' {
  interface FastifyRequest {
    /** Populated by authenticateApiKey hook — present on all /v1/* routes. */
    apiKeyContext?: ApiKeyContext;
    /** Populated by verifyJwt hook — present on all /dashboard/* routes. */
    jwtUser?: JwtUser;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// API key
// ─────────────────────────────────────────────────────────────────────────────

/** Available scopes for INSIGHTSERENITY_API_KEY. */
export const API_KEY_SCOPES = [
  'completions:create',
  'chat:create',
  'embeddings:create',
  'agents:run',
  'admin:all',
] as const;

export type ApiKeyScope = (typeof API_KEY_SCOPES)[number];

// ─────────────────────────────────────────────────────────────────────────────
// Pagination
// ─────────────────────────────────────────────────────────────────────────────

export interface PaginatedResult<T> {
  data:    T[];
  total:   number;
  page:    number;
  limit:   number;
  hasMore: boolean;
}

export interface PaginationParams {
  page:  number;
  limit: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard API response envelope
// ─────────────────────────────────────────────────────────────────────────────

export interface ApiSuccess<T> {
  success: true;
  data:    T;
}

export interface ApiError {
  success: false;
  error:   {
    code:    string;
    message: string;
    details?: unknown;
  };
}

export type ApiResponse<T> = ApiSuccess<T> | ApiError;

// ─────────────────────────────────────────────────────────────────────────────
// Usage
// ─────────────────────────────────────────────────────────────────────────────

export interface UsageStats {
  promptTokens:     number;
  completionTokens: number;
  totalTokens:      number;
  requests:         number;
  avgLatencyMs:     number;
}
