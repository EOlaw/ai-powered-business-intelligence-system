export const DATABASE_PACKAGE_NAME = '@business-intelligence/database';

export type DatabaseHealth = {
  ok: boolean;
  provider: 'postgresql';
  checkedAt: string;
};

export type AuditEventInput = {
  organizationId: string;
  actorId?: string;
  action: string;
  resource: string;
  resourceId?: string;
  metadata?: Record<string, unknown>;
};

export type UsageRecordInput = {
  organizationId: string;
  apiKeyId?: string;
  endpoint: string;
  model?: string;
  promptTokens?: number;
  outputTokens?: number;
  latencyMs: number;
  statusCode: number;
};

export function normalizeSlug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export function buildDatabaseHealth(): DatabaseHealth {
  return {
    ok: true,
    provider: 'postgresql',
    checkedAt: new Date().toISOString(),
  };
}
