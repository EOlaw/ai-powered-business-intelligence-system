/**
 * API Keys module — Zod request schemas
 */

import { z } from 'zod';
import { API_KEY_SCOPES } from '../../types/index.js';

export const CreateApiKeySchema = z.object({
  name:      z.string().min(1).max(100).trim(),
  scopes:    z.array(z.enum(API_KEY_SCOPES)).min(1).default(['completions:create', 'chat:create']),
  expiresAt: z.string().datetime().optional(),
});

export const UpdateApiKeySchema = z.object({
  name:   z.string().min(1).max(100).trim().optional(),
  scopes: z.array(z.enum(API_KEY_SCOPES)).min(1).optional(),
});

export type CreateApiKeyInput = z.infer<typeof CreateApiKeySchema>;
export type UpdateApiKeyInput = z.infer<typeof UpdateApiKeySchema>;
