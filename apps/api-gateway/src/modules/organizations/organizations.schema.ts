/**
 * Organizations module — Zod schemas
 */

import { z } from 'zod';

export const CreateOrgSchema = z.object({
  name: z.string().min(2).max(80).trim(),
});

export const UpdateOrgSchema = z.object({
  name: z.string().min(2).max(80).trim().optional(),
});

export const InviteMemberSchema = z.object({
  email: z.string().email(),
  role:  z.enum(['ADMIN', 'MEMBER']).default('MEMBER'),
});

export const UpdateMemberRoleSchema = z.object({
  role: z.enum(['ADMIN', 'MEMBER']),
});

export type CreateOrgInput       = z.infer<typeof CreateOrgSchema>;
export type UpdateOrgInput       = z.infer<typeof UpdateOrgSchema>;
export type InviteMemberInput    = z.infer<typeof InviteMemberSchema>;
export type UpdateMemberRoleInput = z.infer<typeof UpdateMemberRoleSchema>;
