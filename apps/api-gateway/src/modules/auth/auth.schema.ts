/**
 * Auth module — Zod request/response schemas
 */

import { z } from 'zod';

export const RegisterSchema = z.object({
  email:    z.string().email(),
  password: z.string().min(8).max(128),
  orgName:  z.string().min(2).max(80).trim(),
});

export const LoginSchema = z.object({
  email:    z.string().email(),
  password: z.string().min(1),
});

export const RefreshSchema = z.object({
  refreshToken: z.string().min(1),
});

export type RegisterInput = z.infer<typeof RegisterSchema>;
export type LoginInput    = z.infer<typeof LoginSchema>;
export type RefreshInput  = z.infer<typeof RefreshSchema>;
