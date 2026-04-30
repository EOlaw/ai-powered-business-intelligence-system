export type ServiceName =
  | 'api-gateway'
  | 'ai-engine'
  | 'dashboard'
  | 'worker';

export type Plan = 'FREE' | 'TEAM' | 'BUSINESS' | 'ENTERPRISE';

export type UserRole = 'OWNER' | 'ADMIN' | 'ANALYST' | 'VIEWER';

export type ApiScope =
  | 'chat:create'
  | 'completion:create'
  | 'embedding:create'
  | 'agent:run'
  | 'usage:read'
  | 'admin:read'
  | 'admin:write';

export type ApiResponse<T> = {
  success: true;
  data: T;
  requestId: string;
};

export type ApiErrorResponse = {
  success: false;
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
  requestId: string;
};

export type ModelVersion = {
  name: string;
  version: string;
  promotedAt: string;
  status: 'candidate' | 'active' | 'previous' | 'failed';
  metrics?: {
    perplexity?: number;
    validationLoss?: number;
    accuracy?: number;
  };
};

export type UsageSummary = {
  requests: number;
  promptTokens: number;
  outputTokens: number;
  totalTokens: number;
  errorRate: number;
  p95LatencyMs: number;
};
