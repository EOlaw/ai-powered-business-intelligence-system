/**
 * InsightSerenity API Gateway — Structured Logger
 * ================================================
 * Pino-based logger with:
 *   - JSON output in production (machine-readable, shipped to log aggregator)
 *   - Pretty-printed output in development (human-readable)
 *   - Automatic redaction of sensitive fields (passwords, tokens, secrets)
 *   - Child logger factory for per-module context binding
 *
 * Usage:
 *   import { logger } from '@/observability/logger';
 *   logger.info({ userId: '…' }, 'User registered');
 *
 *   const routeLogger = logger.child({ module: 'auth' });
 *   routeLogger.error({ err }, 'Login failed');
 */

import pino from 'pino';
import { config } from '../config/settings.js';

// ─────────────────────────────────────────────────────────────────────────────
// Redaction paths — pino replaces values at these dot-paths with "[Redacted]"
// ─────────────────────────────────────────────────────────────────────────────

const REDACTED_PATHS = [
  'password',
  'passwordHash',
  'newPassword',
  'currentPassword',
  'authorization',
  'req.headers.authorization',
  'body.password',
  'body.currentPassword',
  'body.newPassword',
  'token',
  'accessToken',
  'refreshToken',
  'apiKey',
  'secret',
  'internalSecret',
];

// ─────────────────────────────────────────────────────────────────────────────
// Transport (pretty-print in dev, JSON in prod)
// ─────────────────────────────────────────────────────────────────────────────

const transport = config.LOG_PRETTY
  ? pino.transport({
      target: 'pino-pretty',
      options: {
        colorize:        true,
        translateTime:   'SYS:HH:MM:ss.l',
        ignore:          'pid,hostname',
        messageFormat:   '{module} — {msg}',
      },
    })
  : undefined;

// ─────────────────────────────────────────────────────────────────────────────
// Logger instance
// ─────────────────────────────────────────────────────────────────────────────

export const logger = pino(
  {
    level:   config.LOG_LEVEL,
    redact:  { paths: REDACTED_PATHS, censor: '[Redacted]' },
    base:    { service: 'api-gateway', env: config.NODE_ENV },
    timestamp: pino.stdTimeFunctions.isoTime,
    formatters: {
      level: (label) => ({ level: label }),
    },
  },
  transport,
);

/**
 * Create a child logger bound to a named module.
 * Adds `module` field to every log line for structured querying.
 */
export function getLogger(module: string): pino.Logger {
  return logger.child({ module });
}
