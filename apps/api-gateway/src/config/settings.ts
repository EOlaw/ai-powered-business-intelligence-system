/**
 * InsightSerenity API Gateway — Configuration
 * =============================================
 * Zod-validated settings parsed from environment variables.
 * All modules import the exported `config` singleton — nothing reads
 * process.env directly outside this file.
 *
 * Validation happens at startup: the process crashes immediately with a
 * clear error message if a required variable is missing or malformed.
 * This prevents silent misconfiguration in production.
 */

import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────────────
// Schema
// ─────────────────────────────────────────────────────────────────────────────

const schema = z.object({
  // Server
  NODE_ENV: z.enum(['development', 'staging', 'production', 'test']).default('development'),
  PORT:     z.coerce.number().int().min(1024).max(65535).default(3000),
  HOST:     z.string().default('0.0.0.0'),

  // Database
  DATABASE_URL: z.string().url(),

  // Redis
  REDIS_URL: z.string().default('redis://localhost:6379'),

  // JWT
  JWT_ACCESS_SECRET:     z.string().min(32),
  JWT_REFRESH_SECRET:    z.string().min(32),
  JWT_ACCESS_EXPIRES_IN: z.string().default('15m'),
  JWT_REFRESH_EXPIRES_IN: z.string().default('7d'),

  // AI Engine (internal service-to-service)
  AI_ENGINE_URL:              z.string().url().default('http://localhost:8001'),
  SERVING_INTERNAL_API_SECRET: z.string().min(16),

  // Security
  BCRYPT_ROUNDS:  z.coerce.number().int().min(8).max(20).default(12),
  API_KEY_BYTES:  z.coerce.number().int().min(16).max(64).default(24),

  // CORS: comma-separated list of allowed origins
  CORS_ORIGINS: z.string().default('*'),

  // Logging
  LOG_LEVEL:  z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace']).default('info'),
  LOG_PRETTY: z
    .string()
    .transform((v) => v === 'true' || v === '1')
    .default('false'),
});

// ─────────────────────────────────────────────────────────────────────────────
// Parse & export
// ─────────────────────────────────────────────────────────────────────────────

function parseConfig() {
  const result = schema.safeParse(process.env);

  if (!result.success) {
    const formatted = result.error.issues
      .map((i) => `  ${i.path.join('.')}: ${i.message}`)
      .join('\n');
    throw new Error(`[config] Invalid environment variables:\n${formatted}`);
  }

  return result.data;
}

export const config = parseConfig();

// ─────────────────────────────────────────────────────────────────────────────
// Derived helpers
// ─────────────────────────────────────────────────────────────────────────────

export const isProd        = config.NODE_ENV === 'production';
export const isTest        = config.NODE_ENV === 'test';
export const isDev         = config.NODE_ENV === 'development';
export const corsOrigins   = config.CORS_ORIGINS === '*'
  ? ['*']
  : config.CORS_ORIGINS.split(',').map((s) => s.trim());

/** Plan-level rate limits: tokens/day and requests/minute. */
export const PLAN_LIMITS = {
  FREE:       { tokensPerDay: 1_000,     requestsPerMin: 10 },
  STARTER:    { tokensPerDay: 100_000,   requestsPerMin: 60 },
  PRO:        { tokensPerDay: 1_000_000, requestsPerMin: 300 },
  ENTERPRISE: { tokensPerDay: Infinity,  requestsPerMin: 1_000 },
} as const;
