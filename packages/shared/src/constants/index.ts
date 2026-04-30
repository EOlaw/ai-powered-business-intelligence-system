export const SERVICE_NAMES = {
  API_GATEWAY: 'api-gateway',
  AI_ENGINE: 'ai-engine',
  DASHBOARD: 'dashboard',
  WORKER: 'worker',
} as const;

export const DEFAULT_MODELS = {
  CHAT: 'business-intelligence-ai',
  EMBEDDING: 'business-intelligence-embedding',
  ANALYTICS: 'business-intelligence-analytics',
} as const;

export const API_SCOPES = {
  CHAT_CREATE: 'chat:create',
  COMPLETION_CREATE: 'completion:create',
  EMBEDDING_CREATE: 'embedding:create',
  AGENT_RUN: 'agent:run',
  USAGE_READ: 'usage:read',
  ADMIN_READ: 'admin:read',
  ADMIN_WRITE: 'admin:write',
} as const;

export const RATE_LIMITS = {
  FREE_REQUESTS_PER_MINUTE: 30,
  TEAM_REQUESTS_PER_MINUTE: 120,
  BUSINESS_REQUESTS_PER_MINUTE: 600,
  ENTERPRISE_REQUESTS_PER_MINUTE: 1200,
} as const;

export const HTTP_HEADERS = {
  API_KEY: 'x-api-key',
  REQUEST_ID: 'x-request-id',
  MODEL_VERSION: 'x-model-version',
} as const;
