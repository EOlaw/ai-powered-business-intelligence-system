/**
 * Users Service — self-service profile management
 */

import { prisma }                from '../../db/client.js';
import { hashPassword, verifyPassword } from '../../security/crypto.js';
import { getLogger }             from '../../observability/logger.js';

const log = getLogger('users-service');

export async function getProfile(userId: string) {
  const user = await prisma.user.findUnique({
    where:  { id: userId },
    select: {
      id: true, email: true, role: true, createdAt: true,
      memberships: { include: { org: { select: { id: true, name: true, slug: true, plan: true } } } },
    },
  });
  if (!user) throw Object.assign(new Error('User not found'), { statusCode: 404, code: 'NOT_FOUND' });
  return user;
}

export async function changePassword(userId: string, currentPassword: string, newPassword: string) {
  const user = await prisma.user.findUnique({ where: { id: userId } });
  if (!user) throw Object.assign(new Error('User not found'), { statusCode: 404, code: 'NOT_FOUND' });

  const valid = await verifyPassword(currentPassword, user.passwordHash);
  if (!valid) throw Object.assign(new Error('Current password is incorrect'), { statusCode: 400, code: 'INVALID_PASSWORD' });

  const newHash = await hashPassword(newPassword);
  await prisma.user.update({ where: { id: userId }, data: { passwordHash: newHash } });

  // Invalidate all sessions to force re-login everywhere
  await prisma.session.deleteMany({ where: { userId } });
  log.info({ userId }, 'Password changed — all sessions revoked');
}

export async function deleteAccount(userId: string) {
  // Cascade deletes memberships and sessions
  await prisma.user.delete({ where: { id: userId } });
  log.info({ userId }, 'User account deleted');
}
